import sys
import os

# add import paths
code_dir = os.path.dirname(os.path.dirname(__file__))  # erc/
sys.path.append(code_dir)

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from PyTorch_CIFAR10.cifar10_models.resnet import resnet50
from config.cfg_exp import get_cfg_defaults, update_from_args
from util import io_file, misc
from exp.tracker_ood import PointRiskTracker, RunningRiskTracker, EProcessTracker, NaiveEProcessTracker, PMEBProcessTracker
from plots.plot_auto_ood import plot_auto


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
        # Concatenate all tensors along the batch dimension
        return (
            torch.cat(all_preds, dim=0),
            torch.cat(all_conf, dim=0)
        )
    
    def outlier_score(self, probs):
        if self.cfg.EXP.OUT_SCORE == "top1":
            score = 1 - torch.max(probs, axis=1)[0]
        elif self.cfg.EXP.OUT_SCORE == "entropy":
            score = - torch.sum(probs * torch.log(probs + 1e-10), axis=1) / torch.log(torch.tensor(probs.shape[1]).float())
        else:
            raise ValueError(f"Outlier score '{self.cfg.EXP.OUT_SCORE}' not supported.")
        return score
    
    def update_ood_prob(self, ood_probs, ood_prob_idx, ts):
        """
        check if the ood probability should be updated and return the new value
        """
        if (ts > 0) and (ts % self.cfg.EXP.NR_OOD_TIMESTEPS == 0) and (ts < len(ood_probs) * self.cfg.EXP.NR_OOD_TIMESTEPS):
            ood_prob_idx += 1
        return ood_probs[ood_prob_idx], ood_prob_idx
    
    def draw_samples(self, ood_prob, id_labels, id_preds, id_conf, ood_labels, ood_preds, ood_conf):
        """
        draw a batch of samples from id and ood data with a given ood probability
        """
        bern_batch = torch.from_numpy(
            np.random.binomial(1, ood_prob, self.cfg.EXP.NR_POINT_RISK_SAMP)
        )  # (NR_POINT_RISK_SAMP,)
        
        nr_ood = bern_batch.sum().item()
        nr_id = len(bern_batch) - nr_ood
        ood_idx = np.random.randint(0, len(ood_labels), nr_ood)
        id_idx = np.random.randint(0, len(id_labels), nr_id)
        
        return (
            bern_batch,
            torch.cat([id_labels[id_idx], ood_labels[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
            torch.cat([id_preds[id_idx], ood_preds[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
            torch.cat([id_conf[id_idx], ood_conf[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
        )
    
    def reduce_batch(self, loss_batch, bern_batch):
        reduce_idx = torch.randperm(self.cfg.EXP.NR_POINT_RISK_SAMP)[:self.cfg.EXP.BATCH_TIMESTEP]
        return loss_batch[reduce_idx], bern_batch[reduce_idx]
        
    def compute_sample_losses(self, psi_cand, bern_batch, lab_batch, pred_batch, conf_batch):
        """
        get the loss for all psi candidates for a batch of samples
        """
        if self.cfg.EXP.RISK == "fpr_fnr":
            psi_loss = torch.where(
                bern_batch.unsqueeze(1) == 0,  # if bern == 0 (ID sample)
                conf_batch.unsqueeze(1) > psi_cand,  # false positive
                conf_batch.unsqueeze(1) <= psi_cand  # false negative
            )
        elif self.cfg.EXP.RISK == "fnr":
            psi_loss = torch.where(
                bern_batch.unsqueeze(1) == 0,
                torch.zeros_like(psi_cand),  # no penalty for false positive
                conf_batch.unsqueeze(1) <= psi_cand  # false negative
            )
        elif self.cfg.EXP.RISK == "fpr":
            psi_loss = torch.where(
                bern_batch.unsqueeze(1) == 0,
                conf_batch.unsqueeze(1) > psi_cand,  # false positive
                torch.zeros_like(psi_cand)  # no penalty for false negative
            )
        else:
            raise ValueError(f"Unknown risk type {self.cfg.EXP.RISK}.")
        return psi_loss.to(torch.float32) # (NR_POINT_RISK_SAMP, psi_size)
    
    def get_valid_psi(self, psi_cand, stop_time, psi_select="min"):
        """
        get the set of valid psi candidates and the selected psi for one time step
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
    
    def get_psi_cs_size(self, valid_psi):
        """
        return the size of the set of valid psi candidates
        """
        return len(valid_psi)
    
    def get_detection_delay_false_alarm(self, stop_time, true_stop_time):
        """
        measure detection delay and count the false alarms for all psi candidates for one trial
        """
        detection_delay = (stop_time - true_stop_time)
        false_alarms = torch.zeros(stop_time.shape[-1]) # (psi_size,)
        false_alarms[torch.where(detection_delay < 0)[0]] = 1
        return detection_delay, false_alarms


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
        "--save_file",
        action=argparse.BooleanOptionalAction,
        default=None,
        required=False,
        help="If shall save tracking results to file.",
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
        "--bet_type",
        type=str,
        default=None,
        required=False,
        choices=["unit_bet", "approx_grapa", "grapa"],
        help="Type of betting function to use for eprocess.",
    )
    parser.add_argument(
        "--batch_ts",
        type=int,
        default=None,
        required=False,
        help="How many samples are received per timestep.",
    )
    parser.add_argument(  # --tracker_window 10 20 30
        "--tracker_window",
        type=int,
        nargs='+',  # Accepts one or more values as a list
        default=None,
        required=False,
        help="Tracker window of time steps for running trackers (provide space-separated values).",
    )
    parser.add_argument(  # --stop_counter 10 20 30
        "--stop_counter",
        type=int,
        nargs='+',  # Accepts one or more values as a list
        default=None,
        required=False,
        help="Condition counter of time steps for robust stopping (provide space-separated values).",
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
        cfg.RUN.EXP_DIR = f"erc_{cfg.EXP.EPS}_{cfg.EXP.DELTA}_{cfg.EXP.RISK}_ts{cfg.EXP.NR_TIMESTEPS}_bts{cfg.EXP.BATCH_TIMESTEP}_ots{cfg.EXP.NR_OOD_TIMESTEPS}_tw{cfg.EXP.TRACKER_WINDOW[1]}"
    if cfg.RUN.LOAD_DIR == "auto": # Point loading dir to sub_dir if not specified
        cfg.RUN.LOAD_DIR = os.path.join(cfg.PROJECT.OUTPUT_DIR, exp_type, cfg.RUN.SUB_DIR)
    
    # Create experiment dir in sub_dir
    full_dir = os.path.join(
        cfg.PROJECT.OUTPUT_DIR,  # .../output_erc/
        exp_type,  # .../exp_ood/
        cfg.RUN.SUB_DIR,  # .../resnet50_cifar10_svhn_entropy/
        f"{cfg.RUN.EXP_DIR}{cfg.RUN.SUFFIX}"  # .../erc_0.1_0.05_fpr_fnr_ts1000_bts10/
    )
    Path(full_dir).mkdir(exist_ok=True, parents=True)
    cfg.RUN.FULL_DIR = full_dir
    
    # Create plot subdir in full_dir
    if cfg.RUN.PLOT:
        plot_dir = os.path.join(full_dir, cfg.RUN.PLOT_DIR)
        Path(plot_dir).mkdir(exist_ok=True, parents=True)
        cfg.RUN.PLOT_DIR = plot_dir
   
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
    logger.info(f"Loading experiment files from above or '{cfg.RUN.LOAD_DIR}'.")

    # Init experiment
    exp = ExpOOD(cfg, logger)
    model = exp.load_model()
    _, id_labels, id_loader, _, ood_labels, ood_loader = exp.load_data()
    
    if cfg.RUN.GET_PRED:
        logger.info("Getting predictions and outlier scores...")
        id_preds, id_conf = exp.get_pred(model, id_loader)
        ood_preds, ood_conf = exp.get_pred(model, ood_loader)
        io_file.save_tensor(id_preds, "id_preds", cfg.RUN.LOAD_DIR)
        io_file.save_tensor(id_conf, "id_conf", cfg.RUN.LOAD_DIR)
        io_file.save_tensor(ood_preds, "ood_preds", cfg.RUN.LOAD_DIR)
        io_file.save_tensor(ood_conf, "ood_conf", cfg.RUN.LOAD_DIR)
        logger.info("Saved to file.")
    else:
        logger.info("Loading existing predictions and outlier scores...")
        del model
        id_preds = io_file.load_tensor("id_preds", cfg.RUN.LOAD_DIR)
        id_conf = io_file.load_tensor("id_conf", cfg.RUN.LOAD_DIR)
        ood_preds = io_file.load_tensor("ood_preds", cfg.RUN.LOAD_DIR)
        ood_conf = io_file.load_tensor("ood_conf", cfg.RUN.LOAD_DIR)
        logger.info("Loaded to file.")
    
    logger.info("Starting test stream setting...")
    
    # Init stream vars valid for all trackers
    ood_probs = torch.arange(cfg.EXP.OOD_START, cfg.EXP.OOD_END + cfg.EXP.OOD_STEP, cfg.EXP.OOD_STEP)
    psi_cand = torch.arange(cfg.EXP.PSI_START, cfg.EXP.PSI_END + cfg.EXP.PSI_STEP, cfg.EXP.PSI_STEP)
    psi_size = len(psi_cand)
    stream_bern = torch.zeros((cfg.EXP.NR_TRIALS, cfg.EXP.NR_TIMESTEPS, cfg.EXP.BATCH_TIMESTEP))
    stream_losses = torch.zeros((cfg.EXP.NR_TRIALS, cfg.EXP.NR_TIMESTEPS, psi_size))

    logger.info("Initializing trackers...")
    point_risk = PointRiskTracker(cfg, logger, psi_cand)
    running_risk = RunningRiskTracker(cfg, logger, psi_cand)
    eprocess = EProcessTracker(cfg, logger, psi_cand)
    naive_eprocess = NaiveEProcessTracker(cfg, logger, psi_cand)
    pmeb_eprocess = PMEBProcessTracker(cfg, logger, psi_cand)
    
    logger.info("Running experiment loop...")
    for tr in range(cfg.EXP.NR_TRIALS):
        ood_prob_idx = 0
        ood_prob = ood_probs[ood_prob_idx]
        
        for ts in tqdm(range(cfg.EXP.NR_TIMESTEPS), desc=f"Trial {tr+1}, Time step", leave=False):
            # update ood prob if necessary
            ood_prob, ood_prob_idx = exp.update_ood_prob(ood_probs, ood_prob_idx, ts)
            # draw samples
            bern_batch, lab_batch, pred_batch, conf_batch = exp.draw_samples(
                ood_prob, id_labels, id_preds, id_conf, ood_labels, ood_preds, ood_conf
            )
            
            # get sample losses & point risk for full batch
            loss_batch = exp.compute_sample_losses(psi_cand, bern_batch, lab_batch, pred_batch, conf_batch)
            point_risk.risk[tr, ts, :] = point_risk.get_risk(loss_batch, bern_batch, explicit=False)
            
            # reduce batch and update stream vars
            loss_batch, bern_batch = exp.reduce_batch(loss_batch, bern_batch)
            stream_bern[tr, ts, :] = bern_batch # (BATCH_TIMESTEP,)
            stream_losses[tr, ts, :] = loss_batch.mean(dim=0) # (psi_size,)
            
            # get running risk
            running_risk.risk[tr, ts, :] = running_risk.get_risk(stream_losses[tr], stream_bern[tr], ts)

            # get eprocess and update storage vars
            eprocess.bets[tr, ts, :] = eprocess.get_bets(stream_losses[tr], ts)
            eprocess.evalues[tr, ts, :] = eprocess.get_evalues(loss_batch, eprocess.bets[tr, ts, :], reduction="mean")
            eprocess.eprocess[tr, ts, :] = eprocess.get_eprocess(eprocess.evalues[tr, ts, :], tr, ts)
            
            naive_eprocess.bets[tr, ts, :] = naive_eprocess.get_bets(stream_losses[tr], ts)
            naive_eprocess.evalues[tr, ts, :] = naive_eprocess.get_evalues(loss_batch, naive_eprocess.bets[tr, ts, :], reduction="mean")
            naive_eprocess.eprocess[tr, ts, :] = naive_eprocess.get_eprocess(naive_eprocess.evalues[tr, ts, :], tr, ts)
            
            pmeb_eprocess.bets[tr, ts, :] = pmeb_eprocess.get_bets(stream_losses[tr], ts)
            pmeb_eprocess.evalues[tr, ts, :] = pmeb_eprocess.get_evalues(stream_losses[tr], ts, loss_batch, pmeb_eprocess.bets[tr, ts, :], reduction="mean")
            pmeb_eprocess.eprocess[tr, ts, :] = pmeb_eprocess.get_eprocess(pmeb_eprocess.evalues[tr, ts, :], tr, ts)
            
            # print("+++ PMEB EPROCESS +++")
            # print(pmeb_eprocess.bets[tr, ts, :])
            # print(pmeb_eprocess.evalues[tr, ts, :])
            # print(pmeb_eprocess.eprocess[tr, ts, :])
            
            # more trackers here...

            # UPDATE PER-STEP METRICS
            # check stopping times
            point_risk.stop_time[tr] = point_risk.check_stop_time(
                point_risk.stop_time[tr], point_risk.risk[tr, ts, :], ts
            )
            running_risk.stop_time[tr] = running_risk.check_stop_time(
                running_risk.stop_time[tr], running_risk.risk[tr, ts, :], ts
            )
            eprocess.stop_time[tr] = eprocess.check_stop_time(
                eprocess.stop_time[tr], eprocess.eprocess[tr, ts, :], ts
            )
            naive_eprocess.stop_time[tr] = naive_eprocess.check_stop_time(
                naive_eprocess.stop_time[tr], naive_eprocess.eprocess[tr, ts, :], ts
            )
            pmeb_eprocess.stop_time[tr] = pmeb_eprocess.check_stop_time(
                pmeb_eprocess.stop_time[tr], pmeb_eprocess.eprocess[tr, ts, :], ts
            )
            # get psi-CIs and store
            point_risk.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, point_risk.stop_time[tr], psi_select="min"
            )
            point_risk.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            point_risk.psi_cs[tr][ts].append(valid_psi)
            
            running_risk.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, running_risk.stop_time[tr], psi_select="min"
            )
            running_risk.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            running_risk.psi_cs[tr][ts].append(valid_psi)
            
            eprocess.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, eprocess.stop_time[tr], psi_select="min"
            )
            eprocess.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            eprocess.psi_cs[tr][ts].append(valid_psi)
            
            naive_eprocess.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, naive_eprocess.stop_time[tr], psi_select="min"
            )
            naive_eprocess.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            naive_eprocess.psi_cs[tr][ts].append(valid_psi)
            
            pmeb_eprocess.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, pmeb_eprocess.stop_time[tr], psi_select="min"
            )
            pmeb_eprocess.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            pmeb_eprocess.psi_cs[tr][ts].append(valid_psi)
            
            # print("STOP TIME:", pmeb_eprocess.stop_time[tr])
        
        # UPDATE PER-TRIAL METRICS
        point_risk.detection_delay[tr], point_risk.false_alarms[tr] = exp.get_detection_delay_false_alarm(
            stop_time=point_risk.stop_time[tr], true_stop_time=point_risk.stop_time[tr]
        )
        running_risk.detection_delay[tr], running_risk.false_alarms[tr] = exp.get_detection_delay_false_alarm(
            stop_time=running_risk.stop_time[tr], true_stop_time=point_risk.stop_time[tr]
        )
        eprocess.detection_delay[tr], eprocess.false_alarms[tr] = exp.get_detection_delay_false_alarm(
            stop_time=eprocess.stop_time[tr], true_stop_time=point_risk.stop_time[tr]
        )
        naive_eprocess.detection_delay[tr], naive_eprocess.false_alarms[tr] = exp.get_detection_delay_false_alarm(
            stop_time=naive_eprocess.stop_time[tr], true_stop_time=point_risk.stop_time[tr]
        )
        pmeb_eprocess.detection_delay[tr], pmeb_eprocess.false_alarms[tr] = exp.get_detection_delay_false_alarm(
            stop_time=pmeb_eprocess.stop_time[tr], true_stop_time=point_risk.stop_time[tr]
        )
    logger.info("Experiment loop finished.")
    
    if cfg.RUN.SAVE_FILE:
        logger.info("Saving experiment results to file...")
        point_risk.save_to_file("point_risk", cfg.RUN.FULL_DIR)
        running_risk.save_to_file("running_risk", cfg.RUN.FULL_DIR)
        eprocess.save_to_file("eprocess", cfg.RUN.FULL_DIR)
        naive_eprocess.save_to_file("naive_eprocess", cfg.RUN.FULL_DIR)
        pmeb_eprocess.save_to_file("pmeb_eprocess", cfg.RUN.FULL_DIR)
    
    if cfg.RUN.PLOT:
        logger.info("Plotting experiment results...")
        plot_auto(
            cfg,
            logger,
            psi_cand,
            stream_losses,
            point_risk,
            running_risk,
            eprocess,
            naive_eprocess,
            pmeb_eprocess,
        )
        
    logger.info("===== EXPERIMENT END =====")


if __name__ == "__main__":
    main()
    
    
