import copy
import os

import torch
from ..dataloaders import InfiniteDataLoader
from ..base_trainer import BaseTrainer
from torchcontrib.optim import SWA as SWA_optimizer


class SWA(BaseTrainer):
    """
    Stochastic Weighted Averaging

    Original paper:
        @article{izmailov2018averaging,
            title={Averaging weights leads to wider optima and better generalization},
            author={Izmailov, Pavel and Podoprikhin, Dmitrii and Garipov, Timur and Vetrov, Dmitry and Wilson, Andrew Gordon},
            journal={arXiv preprint arXiv:1803.05407},
            year={2018}
        }
    """
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        base_opt = self.optimizer
        self.optimizer = SWA_optimizer(base_opt)
        self.optimizer.defaults = base_opt.defaults
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def train_offline(self):
        for i, t in enumerate(self.train_dataset.ENV):
            if t < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(t)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif t == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(t)
                train_id_dataloader = InfiniteDataLoader(
                    dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size, num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(t)
                else:
                    self.train_step(train_id_dataloader)
                    self.save_model(t)
                    self.optimizer.update_swa()
                break
        self.optimizer.swap_swa_sgd()

    def save_swa_model(self, timestamp):
        backup_state_dict = self.network.state_dict()

        self.optimizer.swap_swa_sgd()
        swa_model_path = self.get_model_path(timestamp) + "_swa"
        torch.save(self.network.state_dict(), swa_model_path)

        self.network.load_state_dict(backup_state_dict)

    def load_swa_model(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa"
        self.network.load_state_dict(torch.load(swa_model_path), strict=False)

    def train_online(self):
        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and t == self.split_time:
                break
            if self.args.load_model and self.model_path_exists(t):
                self.load_model(t)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(t)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader)
                self.optimizer.update_swa()
                self.save_swa_model(t)
                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

    def get_swa_model_copy(self, timestamp):
        swa_model_path = self.get_model_path(timestamp) + "_swa_copy"
        torch.save(self.network, swa_model_path)
        return torch.load(swa_model_path)

    def evaluate_online(self):
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, t in enumerate(self.eval_dataset.ENV[:end]):
            model_checkpoint = copy.deepcopy(self.network)
            self.load_swa_model(t)

            avg_acc, worst_acc, best_acc = self.evaluate_stream(i + 1)
            self.task_accuracies[t] = avg_acc
            self.worst_time_accuracies[t] = worst_acc
            self.best_time_accuracies[t] = best_acc

            self.network = model_checkpoint

    def __str__(self):
        return f'SWA-{self.base_trainer_str}'
