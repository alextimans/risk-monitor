import torch
import torch.nn as nn
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.heads import SimCLRProjectionHead


class YearbookNetwork(nn.Module):
    def __init__(self, args, num_input_channels, num_classes, ssl_training=False):
        super(YearbookNetwork, self).__init__()
        self.args = args
        self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.hid_dim = 32
        self.classifier = nn.Linear(32, num_classes)

        # SimCLR: projection head
        if self.args.method == 'simclr':
            self.projection_head = SimCLRProjectionHead(32, 32, 128)
        # SwaV: projection head and prototypes
        elif self.args.method == 'swav':
            self.projection_head = SwaVProjectionHead(32, 32, 128)
            self.prototypes = SwaVPrototypes(128, n_prototypes=32)
        self.ssl_training = ssl_training

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=(2, 3))

        if self.args.method == 'simclr' and self.ssl_training:
            return self.projection_head(x)
        elif self.args.method == 'swav' and self.ssl_training:
            x = self.projection_head(x)
            x = nn.functional.normalize(x, dim=1, p=2)
            return self.prototypes(x)
        else:
            return self.classifier(x)