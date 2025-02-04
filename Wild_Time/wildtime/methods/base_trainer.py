import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from tdc import Evaluator

from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import prepare_data, forward_pass, get_collate_functions


class BaseTrainer:
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = 2
        self.num_classes = dataset.num_classes
        self.num_tasks = dataset.num_tasks
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(args, self.train_dataset)

        # Training hyperparameters
        self.args = args
        self.train_update_iter = args.train_update_iter
        self.lisa = args.lisa
        self.mixup = args.mixup
        self.cut_mix = args.cut_mix
        self.mix_alpha = args.mix_alpha
        self.mini_batch_size = args.mini_batch_size
        self.num_workers = args.num_workers
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.split_time = args.split_time
        self.eval_next_timestamps = args.eval_next_timestamps
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = 'accuracy'
        if str(self.eval_dataset) == 'drug':
            self.eval_metric = 'PCC'
        elif 'mimic' in str(self.eval_dataset) and self.args.prediction_type == 'mortality':
            self.eval_metric = 'ROC-AUC'

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'train_update_iter={self.train_update_iter}-lr={self.args.lr}-' \
                                f'mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}'
        if self.args.lisa:
            base_trainer_str += f'-lisa-mix_alpha={self.mix_alpha}'
        elif self.mixup:
            base_trainer_str += f'-mixup-mix_alpha={self.mix_alpha}'
        if self.cut_mix:
            base_trainer_str += f'-cut_mix'
        if self.args.eval_fix:
            base_trainer_str += f'-eval_fix'
        else:
            base_trainer_str += f'-eval_stream'
        return base_trainer_str

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_online(self):
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and timestamp == self.split_time:
                break
            if self.args.load_model and self.model_path_exists(timestamp):
                self.load_model(timestamp)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader)
                self.save_model(timestamp)
                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

    def train_offline(self):
        if self.args.method in ['simclr', 'swav']:
            self.train_dataset.ssl_training = True
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(timestamp)
                else:
                    self.train_step(train_id_dataloader)
                    self.save_model(timestamp)
                break

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                logits = self.network(x)
                if self.args.dataset in ['drug']:
                    pred = logits.reshape(-1, )
                else:
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()

        if self.args.dataset == 'drug':
            evaluator = Evaluator(name='PCC')
            metric = evaluator(y_all, pred_all)
        else:
            pred_all = np.array(pred_all)
            y_all = np.array(y_all)
            if self.args.dataset == 'mimic' and self.args.prediction_type == 'mortality':
                metric = metrics.roc_auc_score(y_all, pred_all)
            else:
                correct = (pred_all == y_all).sum().item()
                metric = correct / float(y_all.shape[0])
        self.network.train()

        return metric

    def evaluate_stream(self, start):
        self.network.eval()
        metrics = []
        for i in range(start, min(start + self.eval_next_timestamps, len(self.eval_dataset.ENV))):
            test_time = self.eval_dataset.ENV[i]
            self.eval_dataset.update_current_timestamp(test_time)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.mini_batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric = self.network_evaluation(test_time_dataloader)
            metrics.append(metric)

        avg_metric, worst_metric, best_metric = np.mean(metrics), np.min(metrics), np.max(metrics)

        print(
            f'Timestamp = {start - 1}'
            f'\t Average {self.eval_metric}: {avg_metric}'
            f'\t Worst {self.eval_metric}: {worst_metric}'
            f'\t Best {self.eval_metric}: {best_metric}'
            f'\t Performance over all timestamps: {metrics}\n'
        )
        self.network.train()

        return avg_metric, worst_metric, best_metric

    def evaluate_online(self):
        print(f'\n=================================== Results (Eval-Stream) ===================================')
        print(f'Metric: {self.eval_metric}\n')
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, timestamp in enumerate(self.eval_dataset.ENV[:end]):
            self.load_model(timestamp)
            avg_metric, worst_metric, best_metric = self.evaluate_stream(i + 1)
            self.task_accuracies[timestamp] = avg_metric
            self.worst_time_accuracies[timestamp] = worst_metric
            self.best_time_accuracies[timestamp] = best_metric

    def evaluate_offline(self):
        print(f'\n=================================== Results (Eval-Fix) ===================================')
        print(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric = self.network_evaluation(test_id_dataloader)
                print(f'ID {self.eval_metric}: \t{id_metric}\n')
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {acc}')
                metrics.append(acc)
        print(f'\nOOD Average Metric: \t{np.mean(metrics)}'
              f'\nOOD Worst Metric: \t{np.min(metrics)}'
              f'\nAll OOD Metrics: \t{metrics}\n')

    def evaluate_offline_all_timestamps(self):
        print(f'\n=================================== Results (Eval-Fix) ===================================')
        timestamps = self.train_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp <= self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric = self.network_evaluation(test_id_dataloader)
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric = self.network_evaluation(test_ood_dataloader)
            print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
            metrics.append(metric)
        print(f'\nAverage Metric Across All Timestamps: \t{np.mean(metrics)}'
              f'\nWorst Metric Across All Timestamps: \t{np.min(metrics)}'
              f'\nMetrics Across All Timestamps: \t{metrics}\n')

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        if self.args.method in ['agem', 'ewc', 'ft', 'si']:
            self.train_online()
        else:
            self.train_offline()
        if self.args.eval_all_timestamps:
            self.evaluate_offline_all_timestamps()
        else:
            self.evaluate_offline()

    def run_task_difficulty(self):
        print('==========================================================================================')
        print("Running Task Difficulty...\n")
        timestamps = self.train_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            self.train_dataset.mode = 0
            self.train_dataset.update_current_timestamp(timestamp)
            if i < len(timestamps) - 1:
                self.train_dataset.update_historical(i + 1)
            else:
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(timestamp)
                else:
                    self.train_step(train_id_dataloader)
                    self.save_model(timestamp)

        for i, timestamp in enumerate(timestamps):
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(timestamp)
            test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                 batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric = round(self.network_evaluation(test_ood_dataloader), 2)
            print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
            metrics.append(metric)
        print(f'Average Metric: {np.mean(metrics)}')
        print(f'Worst timestamp accuracy: {np.min(metrics)}')
        print(f'All timestamp accuracies: {metrics}')

    def run_eval_stream(self):
        print('==========================================================================================')
        print("Running Eval-Stream...\n")
        if not self.args.load_model:
            self.train_online()
        self.evaluate_online()

    def run(self):
        torch.cuda.empty_cache()
        start_time = time.time()
        if self.args.difficulty:
            self.run_task_difficulty()
        elif self.args.eval_fix:
            self.run_eval_fix()
        else:
            self.run_eval_stream()
        runtime = time.time() - start_time
        print(f'Runtime: {runtime:.2f}\n')

    def get_model_path(self, timestamp):
        model_str = f'{str(self.train_dataset)}_{str(self)}_time={timestamp}'
        path = os.path.join(self.args.log_dir, model_str)
        return path

    def model_path_exists(self, timestamp):
        return os.path.exists(self.get_model_path(timestamp))

    def save_model(self, timestamp):
        path = self.get_model_path(timestamp)
        torch.save(self.network.state_dict(), path)
        print(f'Saving model at timestamp {timestamp} to path {path}...\n')

    def load_model(self, timestamp):
        path = self.get_model_path(timestamp)
        self.network.load_state_dict(torch.load(path), strict=False)
