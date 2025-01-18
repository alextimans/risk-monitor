import sys
import os

# add import paths

code_dir = os.path.dirname(os.path.dirname(__file__))  # erc/
sys.path.append(code_dir)

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import binom
from tqdm import tqdm

import torch
import torch.nn as nn
# from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from PyTorch_CIFAR10.cifar10_models.resnet import resnet50
from config.cfg_exp import get_cfg_defaults, update_from_args
from util import io_file, misc


class ExpOOD:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = self.cfg.MODEL.DEVICE
        
    def load_model(self):
        if self.cfg.MODEL.TYPE == "resnet50":
            model = resnet50(pretrained=True)
            self.data_mean = [0.4914, 0.4822, 0.4465]
            self.data_std = [0.2471, 0.2435, 0.2616]
        else:
            raise ValueError(f"Model type '{self.cfg.MODEL.TYPE}' not supported.")
        model.to(self.device)
        model.eval()
        self.logger.info(f"Loaded model '{self.cfg.MODEL.TYPE}'.")
        return model
        
    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.data_mean, self.data_std)
        ])
        
        if self.cfg.EXP.DATA_ID == "cifar10":
            id_dataset = datasets.CIFAR10(root=self.cfg.DATASET.DIR, train=False, transform=transform)
            id_labels = torch.tensor(id_dataset.targets)
            id_loader = DataLoader(id_dataset, batch_size=self.cfg.DATASET.BATCH_PRED, shuffle=False)
        else:
            raise ValueError(f"Dataset '{self.cfg.EXP.DATA_ID}' for ID not supported.")
        
        if self.cfg.EXP.DATA_OOD == "svhn":
            ood_dataset = datasets.SVHN(root=self.cfg.DATASET.DIR, split='test', transform=transform)
            ood_labels = torch.tensor(ood_dataset.labels)
            ood_loader = DataLoader(ood_dataset, batch_size=self.cfg.DATASET.BATCH_PRED, shuffle=False)
        else:
            raise ValueError(f"Dataset '{self.cfg.EXP.DATA_OOD}' for OOD not supported.")
        
        self.logger.info("Loaded datasets")
        self.logger.info(f"ID dataset: {self.cfg.EXP.DATA_ID}, {len(id_loader.dataset), id_loader.dataset.__getitem__(0)[0].shape}")
        self.logger.info(f"OOD dataset: {self.cfg.EXP.DATA_OOD}, {len(ood_loader.dataset), ood_loader.dataset.__getitem__(0)[0].shape}")
        return id_dataset, id_labels, id_loader, ood_dataset, ood_labels, ood_loader
        
    def get_pred(self, model, loader):
        all_preds, all_conf = [], []

        with torch.no_grad():
            for images, _ in tqdm(loader, desc="Batch"):
                images = images.to(self.device)
                out = model(images).to('cpu')
                probs = nn.functional.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                conf = self.outlier_score(probs)
                
                all_preds.append(preds)
                all_conf.append(conf)
        return (
            torch.tensor(all_preds).to(torch.int32),
            torch.tensor(all_conf).to(torch.float32)
        )
    
    def outlier_score(self, probs):
        if self.cfg.EXP.OUT_SCORE == "top1":
            score = 1 - torch.max(probs, axis=1)[0]
        elif self.cfg.EXP.OUT_SCORE == "entropy":
            score = - torch.sum(probs * torch.log(probs + 1e-10), axis=1) / torch.log(torch.tensor(probs.shape[1]).float())
        else:
            raise ValueError(f"Outlier score '{self.cfg.EXP.OUT_SCORE}' not supported.")
        return score
    
    def update_ood_prob(self, ood_probs, ood_prob_idx, t):
        """
        check if the ood probability should be updated and return the new value
        """
        if (t > 0) and (t % self.cfg.EXP.NR_OOD_TIMESTEPS == 0) and (t < len(ood_probs) * self.cfg.EXP.NR_OOD_TIMESTEPS):
            ood_prob_idx += 1
        return ood_probs[ood_prob_idx]
    
    def draw_samples(self, ood_prob, id_labels, id_preds, id_conf, ood_labels, ood_preds, ood_conf):
        """
        draw a batch of samples from id and ood data with a given ood probability
        """
        bern_batch = torch.from_numpy(
            np.random.binomial(1, ood_prob, self.cfg.EXP.NR_POINT_RISK_SAMP)
            )  # (NR_POINT_RISK_SAMP,)
        
        nr_ood = bern_batch.sum().item()
        nr_id = len(bern_batch) - nr_ood
        ood_idx = np.random.randint(0, len(self.ood_labels), nr_ood)
        id_idx = np.random.randint(0, len(self.id_labels), nr_id)
        
        return (
            bern_batch,
            torch.cat([id_labels[id_idx], ood_labels[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
            torch.cat([id_preds[id_idx], ood_preds[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
            torch.cat([id_conf[id_idx], ood_conf[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
        )
        
    def compute_sample_losses(self, psi_cand, bern_batch, lab_batch, pred_batch, conf_batch):
        """
        get the loss for all psi candidates for a batch of samples
        """
        if self.cfg.EXP.RISK == "outlier_fpr_fnr":
            psi_loss = torch.where(
                bern_batch.unsqueeze(1) == 0,  # if bern == 0 (ID sample)
                conf_batch.unsqueeze(1) > psi_cand,  # false positive
                conf_batch.unsqueeze(1) <= psi_cand  # false negative
            )
        elif self.cfg.EXP.RISK == "outlier_fnr":
            psi_loss = torch.where(
                bern_batch.unsqueeze(1) == 0,
                torch.zeros_like(psi_cand),  # no penalty for false positive
                conf_batch.unsqueeze(1) <= psi_cand  # false negative
            )
        elif self.cfg.EXP.RISK == "outlier_fpr":
            psi_loss = torch.where(
                bern_batch.unsqueeze(1) == 0,
                conf_batch.unsqueeze(1) > psi_cand,  # false positive
                torch.zeros_like(psi_cand)  # no penalty for false negative
            )
        else:
            raise ValueError(f"Unknown risk type {self.cfg.EXP.RISK}.")
        return psi_loss.to(torch.float32)
    
    def get_valid_psi(self, psi_cand, stop_time, psi_select="min"):
        """
        get the set of valid psi candidates for one time step and return the selected psi
        """
        assert psi_cand.shape[0] == stop_time.shape[0], "Number of psi cand and stop times must match."
        valid_psi = psi_cand[torch.where(stop_time == -1)[0]]  # not yet stopped
        
        if len(valid_psi) > 0:
            if psi_select == "min":
                select_psi = valid_psi.min()
            elif psi_select == "max":
                select_psi = valid_psi.max()
            elif psi_select == "median":
                select_psi = valid_psi.median()
            else:
                raise ValueError(f"Unknown psi selection criterion {psi_select}.")
        else:  # default to a 'trivial' safe zone
            select_psi = torch.ones(1)
        return select_psi, valid_psi.tolist()
    
    def count_false_alarm(self, psi_size, stop_time, true_stop_time):
        """
        count the false alarms for all psi candidates for one trial
        """
        false_alarms = torch.zeros(psi_size)
        early_stops = torch.where((stop_time - true_stop_time) < 0)[0]
        false_alarms[early_stops] = 1
        return false_alarms


def create_parser():
    """
    hierarchy: CLI > cfg > cfg default
    """
    parser = argparse.ArgumentParser(
        description="Parser for CLI arguments to run model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        default=None,
        required=False,
        help="Config file name to get settings to use for current run.",
    )
    parser.add_argument(
        "--cfg_dir",
        type=str,
        default=None,
        required=False,
        help="Config dir to get settings to use for current run.",
    )
    parser.add_argument(
        "--load_dir",
        type=str,
        default=None,
        required=False,
        help="Path to load files from for current run.",
    )
    parser.add_argument(
        "--exp_suffix",
        type=str,
        default=None,
        required=False,
        help="Experiment folder suffix to use for current run.",
    )
    parser.add_argument(
        "--get_pred",
        action=argparse.BooleanOptionalAction,
        default=None,
        required=False,
        help="If run pred loop to do inference and compute preds (bool).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        required=False,
        help="Tolerated risk upper bound.",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        required=False,
        help="Risk bound probability (1 - delta).",
    )
    parser.add_argument(
        "--risk",
        type=str,
        default=None,
        required=False,
        choices=["fpr_fnr", "fnr", "fpr"],
        help="Desired loss function to compute associated risks.",
    )
    parser.add_argument(
        "--out_score",
        type=str,
        default=None,
        required=False,
        choices=["top1", "entropy"],
        help="Type of outlier score to compute from class prob.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
        required=False,
        choices=["point_risk", "running_risk", "eprocess"],
        help="Type of risk tracker to use.",
    )
    parser.add_argument(
        "--bet_type",
        type=str,
        default=None,
        required=False,
        choices=["unit_bet", "approx_grapa", "grapa"],
        help="Type of betting function to use for eprocess.",
    )
    parser.add_argument(
        "--batch_ts",
        type=float,
        default=None,
        required=False,
        help="How many samples are received per timestep.",
    )
    # parser.add_argument(
    #     "--lookback",
    #     type=float,
    #     default=None,
    #     required=False,
    #     help="Lookback window of time steps for tracker params.",
    # )
    parser.add_argument(  # --lookback 10 20 30
        "--lookback",
        type=int,
        nargs='+',  # Accepts one or more values as a list
        default=None,
        required=False,
        help="Lookback window of time steps for tracker params (provide space-separated values).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        required=False,
        choices=["cpu", "cuda"],
        help="Device to run code on.",
    )
    return parser


def set_dirs(cfg):
    # Determine experiment names
    exp_type = cfg.PROJECT.CONFIG_DIR.split("/")[-1]
    if cfg.RUN.SUB_DIR == "auto":
        cfg.RUN.SUB_DIR = f"{cfg.MODEL.TYPE}_{cfg.EXP.DATA_ID}_{cfg.EXP.DATA_OOD}_{cfg.EXP.OUT_SCORE}"
    if cfg.RUN.EXP_DIR == "auto":
        cfg.RUN.EXP_DIR = f"erc_{cfg.EXP.EPS}_{cfg.EXP.DELTA}_{cfg.EXP.RISK}_{cfg.EXP.TRACKER}_bt{cfg.EXP.BATCH_TIMESTEP}_lk{cfg.EXP.LOOKBACK}"
    
    # Create full experiment dir
    full_dir = os.path.join(
        cfg.PROJECT.OUTPUT_DIR,  # .../output_erc/
        exp_type,  # .../exp_ood/
        cfg.RUN.SUB_DIR,  # .../resnet50_cifar10_svhn_entropy/
        f"{cfg.RUN.EXP_DIR}{cfg.RUN.SUFFIX}"  # .../erc_0.1_0.05_fpr_fnr_point_risk_bt100_lk100/
    )
    Path(full_dir).mkdir(exist_ok=True, parents=True)
    cfg.RUN.FULL_DIR = full_dir
    
    # Create plot subdir in full_dir
    if cfg.RUN.PLOT:
        plot_dir = os.path.join(full_dir, cfg.RUN.PLOT_DIR)
        Path(plot_dir).mkdir(exist_ok=True, parents=True)
        cfg.RUN.PLOT_DIR = plot_dir
    
    # Point loading dir to sub_dir if not specified
    if cfg.RUN.LOAD_DIR == "auto":
        cfg.RUN.LOAD_DIR = os.path.join(cfg.PROJECT.OUTPUT_DIR, exp_type, cfg.RUN.SUB_DIR)
   
    return cfg


def main():
    parser = create_parser()
    args = parser.parse_args()

    # cfg default -> override opts with cfg exp -> override opts with CLI -> device check
    cfg = get_cfg_defaults()
    cfg_exp_file = cfg.PROJECT.CONFIG_FILE if args.cfg_file is None else args.cfg_file
    cfg_exp_dir = cfg.PROJECT.CONFIG_DIR if args.cfg_dir is None else args.cfg_dir
    cfg_exp = io_file.load_yaml(cfg_exp_file, cfg_exp_dir, to_yacs=True)
    cfg.merge_from_other_cfg(cfg_exp)  # override cfg with cfg_exp
    cfg, _ = update_from_args(cfg, args)  # override cfg with args
    cfg.MODEL.DEVICE = misc.set_device(cfg.MODEL.DEVICE)

    # Set folder dirs and freeze
    cfg = set_dirs(cfg)
    cfg.freeze()
    
    # Set up logger & seed
    logger = misc.get_logger(cfg.RUN.FULL_DIR, "log.txt")
    misc.set_seed(cfg.PROJECT.SEED, logger)
    
    # EXPERIMENT START ########################
    
    logger.info("===== EXPERIMENT START =====")
    logger.info(f"Using config file '{cfg.PROJECT.CONFIG_FILE}'.")
    logger.info(f"Saving experiment files to '{cfg.RUN.FULL_DIR}'.")
    logger.info(f"Loading experiment files from '{cfg.RUN.LOAD_DIR}'.")

    # Init experiment
    exp = ExpOOD(cfg, logger)
    model = exp.load_model()
    id_dataset, id_labels, id_loader, ood_dataset, ood_labels, ood_loader = exp.load_data()
    
    if cfg.RUN.GET_PRED:
        logger.info("Getting predictions and outlier scores...")
        id_preds, id_conf = exp.get_pred(model, id_loader)
        ood_preds, ood_conf = exp.get_pred(model, ood_loader)
        logger.info("Saving to file.")
        io_file.save_tensor(id_preds, "id_preds", cfg.RUN.FULL_DIR)
        io_file.save_tensor(id_conf, "id_conf", cfg.RUN.FULL_DIR)
        io_file.save_tensor(ood_preds, "ood_preds", cfg.RUN.FULL_DIR)
        io_file.save_tensor(ood_conf, "ood_conf", cfg.RUN.FULL_DIR)
    else:
        logger.info("Loading existing predictions and outlier scores...")
        id_preds = io_file.load_tensor("id_preds", cfg.RUN.LOAD_DIR)
        id_conf = io_file.load_tensor("id_conf", cfg.RUN.LOAD_DIR)
        ood_preds = io_file.load_tensor("ood_preds", cfg.RUN.LOAD_DIR)
        ood_conf = io_file.load_tensor("ood_conf", cfg.RUN.LOAD_DIR)
        logger.info("Loaded to file.")
    
    # AVAILABLE VARIABLES
    # logger, cfg
    # model, id_dataset, id_labels, id_loader, ood_dataset, ood_labels, ood_loader
    # id_preds, id_conf, ood_preds, ood_conf
    
    logger.info("Starting test stream setting...")
    
    # Init stream variables valid for all trackers
    ood_probs = torch.arange(cfg.EXP.OOD_START, cfg.EXP.OOD_END, cfg.EXP.OOD_STEP)
    psi_cand = torch.arange(cfg.EXP.PSI_START, cfg.EXP.PSI_END, cfg.EXP.PSI_STEP)
    psi_size = len(psi_cand)
    stream_bern = torch.zeros((cfg.EXP.NR_TRIALS,cfg.EXP.NR_TIMESTEPS, cfg.EXP.BATCH_TIMESTEP))
    stream_losses = torch.zeros((cfg.EXP.NR_TRIALS, cfg.EXP.NR_TIMESTEPS, psi_size))

    # Initialize tracker objects
    # PointRiskTracker
        # vars: risk, stop_time, psi_select, psi_cs, psi_cs_size, false_alarms, detection_delay
        # func: get_risk, check_stop_time, get_valid_psi, count_false_alarm
    # RunningRiskTracker
        # vars: risk, stop_time, psi_select, psi_cs, psi_cs_size, false_alarms, detection_delay
        # func: get_risk, check_stop_time, get_valid_psi, count_false_alarm
    # EProcessTracker
        # vars: bets, eval, eprocess, stop_time, psi_select, psi_cs, psi_cs_size, false_alarms, detection_delay
        # func: get_bet, get_eval, get_eprocess, check_stop_time, get_valid_psi, count_false_alarm  
    
    # exp trial loop with if/else for each tracker (or just all of them)
    
    update_ood_prob(ood_probs, ood_prob_idx, t) # NOTE: RANDOMLY SAMPLE FOR SUB-BATCH INSTEAD OF INDEXING
    draw_samples(ood_prob, id_labels, id_preds, id_conf, ood_labels, ood_preds, ood_conf)
    compute_sample_losses(psi_cand, bern_batch, lab_batch, pred_batch, conf_batch)
    get_valid_psi(psi_cand, stop_time, psi_select="min")
    
    # eval
    count_false_alarm(psi_size, stop_time, true_stop_time)
     # make simple func for detection delay
     # make simple func for 
    
    


    # Initialize risk control object (controller)
    logger.info(f"Init risk control procedure with '{args.risk_control}'...")
    if args.risk_control == "std_conf":
        controller = std_conformal.StdConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    elif args.risk_control == "ens_conf":
        controller = ens_conformal.EnsConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    elif args.risk_control == "cqr_conf":
        controller = cqr_conformal.CQRConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    elif args.risk_control == "base_conf":
        controller = baseline_conformal.BaselineConformal(
            cfg, args, nr_class, filedir, log=wandb_run, logger=logger
        )
    else:
        raise ValueError("Risk control procedure not specified.")
    
    
    logger.info("===== EXPERIMENT END =====")


if __name__ == "__main__":
    main()
    
    
