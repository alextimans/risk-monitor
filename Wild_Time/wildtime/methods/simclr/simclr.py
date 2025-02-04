import os

from ..dataloaders import InfiniteDataLoader
from lightly.loss import NTXentLoss
from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass


class SimCLR(BaseTrainer):
    """
    SimCLR

    Original paper:
        @inproceedings{chen2020simple,
            title={A simple framework for contrastive learning of visual representations},
            author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
            booktitle={International conference on machine learning},
            pages={1597--1607},
            year={2020},
            organization={PMLR}
        }

    Code uses Lightly, a Python library for self-supervised learning on images: https://github.com/lightly-ai/lightly.
    """

    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        assert str(dataset) == 'yearbook' or str(dataset) == 'fmow', \
            'SimCLR on image classification datasets only'
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.network.ssl_training = True
        self.ssl_criterion = NTXentLoss()
        self.ssl_finetune_iter = args.ssl_finetune_iter
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'SimCLR-{self.base_trainer_str}-ssl_finetune_iter={self.ssl_finetune_iter}'

    def finetune_classifier(self):
        self.network.ssl_training = False
        self.train_dataset.ssl_training = False
        finetune_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                 batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
        self.network.train()
        loss_all = []

        for step, (x, y) in enumerate(finetune_dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == self.ssl_finetune_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (x, y, _) in enumerate(dataloader):
            (x0, x1), y = prepare_data(x, y, str(self.train_dataset))

            z0 = self.network(x0)
            z1 = self.network(x1)
            loss = self.ssl_criterion(z0, z1)

            loss_all.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.finetune_classifier()
                break