import numpy as np
import torch
import torch.nn as nn
import random

from .networks.article import ArticleNetwork
from .networks.drug import DTI_Encoder, DTI_Classifier
from .networks.fmow import FMoWNetwork
from .networks.mimic import Transformer
from .networks.yearbook import YearbookNetwork
from functools import partial
from .methods.agem.agem import AGEM
from .methods.coral.coral import DeepCORAL
from .methods.erm.erm import ERM
from .methods.ewc.ewc import EWC
from .methods.ft.ft import FT
from .methods.groupdro.groupdro import GroupDRO
from .methods.irm.irm import IRM
from .methods.si.si import SI
from .methods.simclr.simclr import SimCLR
from .methods.swa.swa import SWA
from .methods.swav.swav import SwaV

scheduler = None
group_datasets = ['coral', 'groupdro', 'irm']
print = partial(print, flush=True)


def _yearbook_init(args):
    if args.method in group_datasets:
        from .data.yearbook import YearbookGroup
        dataset = YearbookGroup(args)
    else:
        from .data.yearbook import Yearbook
        dataset = Yearbook(args)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    network = YearbookNetwork( args, num_input_channels=3, num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _fmow_init(args):
    if args.method in group_datasets:
        from .data.fmow import FMoWGroup
        dataset = FMoWGroup(args)
    else:
        from .data.fmow import FMoW
        dataset = FMoW(args)

    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    network = FMoWNetwork(args).cuda()
    optimizer = torch.optim.Adam((network.parameters()), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True,
                                 betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)
    return dataset, criterion, network, optimizer, scheduler


def _drug_init(args):
    if args.method in group_datasets:
        from .data.drug import TdcDtiDgGroup
        dataset = TdcDtiDgGroup(args)
    else:
        from .data.drug import TdcDtiDg
        dataset = TdcDtiDg(args)

    scheduler = None
    criterion = nn.MSELoss(reduction=args.reduction).cuda()
    featurizer = DTI_Encoder()
    classifier = DTI_Classifier(featurizer.n_outputs, 1)
    network = nn.Sequential(featurizer, classifier).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler

def _mimic_init(args):
    if args.method in group_datasets:
        from .data.mimic import MIMICGroup
        dataset = MIMICGroup(args)
    else:
        from .data.mimic import MIMIC
        dataset = MIMIC(args)

    scheduler = None
    network = Transformer(args, embedding_size=128, dropout=0.5, layers=2, heads=2).cuda()
    class_weight = None
    if args.prediction_type == 'readmission':
        class_weight = torch.FloatTensor(np.array([0.26, 0.74])).cuda()
    elif args.prediction_type == 'mortality':
        if args.lisa:
            class_weight = torch.FloatTensor(np.array([0.03, 0.97])).cuda()
        else:
            class_weight = torch.FloatTensor(np.array([0.05, 0.95])).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weight, reduction=args.reduction).cuda()
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)  # use lr = 5e-4
    return dataset, criterion, network, optimizer, scheduler

def _arxiv_init(args):
    if args.method in group_datasets:
        from .data.arxiv import ArXivGroup
        dataset = ArXivGroup(args)
    else:
        from .data.arxiv import ArXiv
        dataset = ArXiv(args)
    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    network = ArticleNetwork(num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler

def _huffpost_init(args):
    if args.method in group_datasets:
        from .data.huffpost import HuffPostGroup
        dataset = HuffPostGroup(args)
    else:
        from .data.huffpost import HuffPost
        dataset = HuffPost(args)
    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    network = ArticleNetwork(num_classes=dataset.num_classes).cuda()
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def trainer_init(args):

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.device)
    if args.method in ['groupdro', 'irm']:
        args.reduction = 'none'
    else:
        args.reduction = 'mean'
    return globals()[f'_{args.dataset}_init'](args)

def init(args):
    dataset, criterion, network, optimizer, scheduler = trainer_init(args)
    method_dict = {'groupdro': 'GroupDRO', 'coral': 'DeepCORAL', 'irm': 'IRM', 'ft': 'FT', 'erm': 'ERM', 'ewc': 'EWC',
                  'agem': 'AGEM', 'si': 'SI', 'simclr': 'SimCLR', 'swav': 'SwaV', 'swa': 'SWA'}
    trainer = globals()[method_dict[args.method]](args, dataset, network, criterion, optimizer, scheduler)
    return trainer

def train(args):
    trainer = init(args)
    trainer.run()
