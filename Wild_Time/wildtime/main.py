import argparse
import os
import random

import numpy as np
import torch
from torch import cuda

from baseline_trainer import trainer_init
from config import config
from methods.agem.agem import AGEM
from methods.coral.coral import DeepCORAL
from methods.erm.erm import ERM
from methods.ewc.ewc import EWC
from methods.ft.ft import FT
from methods.groupdro.groupdro import GroupDRO
from methods.irm.irm import IRM
from methods.si.si import SI
from methods.simclr.simclr import SimCLR
from methods.swa.swa import SWA
from methods.swav.swav import SwaV

device = 'cuda' if cuda.is_available() else 'cpu'


if __name__ == '__main__':
    configs = argparse.Namespace(**config)
    print(configs)

    random.seed(configs.random_seed)
    np.random.seed(configs.random_seed)
    torch.cuda.manual_seed(configs.random_seed)
    torch.manual_seed(configs.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.empty_cache()

    if not os.path.isdir(configs.data_dir):
        raise ValueError(f'Data directory {configs.data_dir} does not exist!')
    if configs.load_model and not os.path.isdir(configs.log_dir):
        raise ValueError(f'Model checkpoint directory {configs.log_dir} does not exist!')
    if not os.path.isdir(configs.results_dir):
        raise ValueError(f'Results directory {configs.results_dir} does not exist!')

    if configs.method in ['groupdro', 'irm']:
        configs.reduction = 'none'

    dataset, criterion, network, optimizer, scheduler = trainer_init(configs)

    if   configs.method == 'groupdro': trainer = GroupDRO(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'coral': trainer = DeepCORAL(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'irm': trainer = IRM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'ft': trainer = FT(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'erm': trainer = ERM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'ewc': trainer = EWC(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'agem': trainer = AGEM(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'si': trainer = SI(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'simclr': trainer = SimCLR(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'swav': trainer = SwaV(configs, dataset, network, criterion, optimizer, scheduler)
    elif configs.method == 'swa': trainer = SWA(configs, dataset, network, criterion, optimizer, scheduler)
    else: raise ValueError

    trainer.run()

    # todo: When using a dictionary to store classes, each class will be instantiated and there will be incompatible datasets and methods
    # if configs.method in ['coral', 'groupdro', 'irm']:
    #     trainer_dict = {
    #         'groupdro': GroupDRO(*param),
    #         'coral':    DeepCORAL(*param),
    #         'irm':      IRM(*param),
    #     }
    #
    # else:
    #     trainer_dict = {
    #                         'ft':     FT(*param),
    #                         'erm':    ERM(*param),
    #                         'ewc':    EWC(*param),
    #                         'agem':   AGEM(*param),
    #                         'si':     SI(*param),
    #                         'simclr': SimCLR(*param),
    #                         'swav':   SwaV(*param),
    #                         'swa':    SWA(*param),
    #     }




