import os

from lightly.loss import SwaVLoss

from ..base_trainer import BaseTrainer
from ..dataloaders import InfiniteDataLoader
from ..utils import prepare_data, forward_pass


class SwaV(BaseTrainer):
    """
    SwaV

    Original paper:
        @article{caron2020unsupervised,
            title={Unsupervised learning of visual features by contrasting cluster assignments},
            author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
            journal={Advances in Neural Information Processing Systems},
            volume={33},
            pages={9912--9924},
            year={2020}
        }

    Code uses Lightly, a Python library for self-supervised learning on images: https://github.com/lightly-ai/lightly.
    """
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        assert str(dataset) == 'yearbook' or str(dataset) == 'fmow', \
            'SimCLR on image classification datasets only'
        super().__init__(args, dataset, network, criterion, optimizer, scheduler)
        self.network.ssl_training = True
        self.ssl_criterion = SwaVLoss()
        self.ssl_finetune_iter = args.ssl_finetune_iter
        self.results_file = os.path.join(args.results_dir, f'{str(dataset)}-{str(self)}.pkl')

    def __str__(self):
        return f'SwaV-{self.base_trainer_str}-ssl_finetune_iter={self.ssl_finetune_iter}'

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
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step == self.ssl_finetune_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        for step, (batch, _, _) in enumerate(dataloader):
            self.network.prototypes.normalize()

            multi_crop_features = [self.network(x.cuda()) for x in batch]
            high_resolution = multi_crop_features[:2]
            low_resolution = multi_crop_features[2:]
            loss = self.ssl_criterion(high_resolution, low_resolution)

            loss_all.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.finetune_classifier()
                break