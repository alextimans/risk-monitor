import sys
import os

# add import paths
code_dir = os.path.dirname(os.path.dirname(__file__))  # erc/
sys.path.append(code_dir)

import argparse
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from config.cfg_exp import get_cfg_defaults, update_from_args
from util import io_file, misc
from exp.tracker_uci import PointRiskTracker, RunningRiskTracker, EProcessTracker, NaiveEProcessTracker, PMEBProcessTracker
from plots.plot_auto_cp import plot_auto


class ExpCP:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.device = self.cfg.MODEL.DEVICE
        
    def load_model(self):
        if self.cfg.MODEL.TYPE == "rf":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
        else:
            raise ValueError(f"Model type '{self.cfg.MODEL.TYPE}' not supported.")
        self.logger.info(f"Loaded model '{self.cfg.MODEL.TYPE}'.")
        return model
        
    def load_data(self, data_name="uci"):
        if data_name == "uci":
            path = os.path.join(self.cfg.DATASET.DIR, "uci-cbm", "data_aug.csv")
            data = np.loadtxt(path, delimiter=',')
            # data = torch.tensor(data, dtype=torch.float32)
        else:
            raise ValueError(f"Data '{data_name}' not supported.")
        self.logger.info("Loaded dataset")
        return data
        
    def get_pred(self, model, data):
        # split by idx (augm. train vs. test)
        X_train, y_train = data[:self.cfg.EXP.SPLIT_TIME, :-1], data[:self.cfg.EXP.SPLIT_TIME, -1]
        X_test, y_test = data[self.cfg.EXP.SPLIT_TIME:, :-1], data[self.cfg.EXP.SPLIT_TIME:, -1]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return (
            torch.tensor((y_test), dtype=torch.float32),
            torch.tensor((y_pred), dtype=torch.float32)
        )
    
    def set_score(self, psi_cand, pred_batch):
        if self.cfg.EXP.SET_SCORE == "abs_res":
            pi = torch.stack((
                (pred_batch.unsqueeze(1) - psi_cand.unsqueeze(0)), # lower
                (pred_batch.unsqueeze(1) + psi_cand.unsqueeze(0)) # upper
            ), dim=2).clamp(0, 1) # (nr_samples, nr_psi, 2)
        else:
            raise ValueError(f"Set score '{self.cfg.EXP.OUT_SCORE}' not supported.")
        return pi
    
    def draw_samples(self, data_idx, labs, preds):
        """
        draw a batch of samples from the specified data time range
        """
        rand_idx = torch.randint(data_idx[0], data_idx[1], (self.cfg.EXP.NR_POINT_RISK_SAMP,))
        return labs[rand_idx], preds[rand_idx]
    
    def reduce_batch(self, loss_batch):
        reduce_idx = torch.randperm(self.cfg.EXP.NR_POINT_RISK_SAMP)[:self.cfg.EXP.BATCH_TIMESTEP]
        return loss_batch[reduce_idx]
    
    def compute_sample_losses(self, psi_cand, lab_batch, pred_batch):
        """
        get the loss for all psi candidates for a batch of samples
        """
        if self.cfg.EXP.RISK == "miscover":
            pred_int = self.set_score(psi_cand, pred_batch)
            cover = (lab_batch.unsqueeze(1) >= pred_int[:, :, 0]) * (lab_batch.unsqueeze(1) <= pred_int[:, :, 1]) # (nr_samples, nr_psi)
            psi_loss = (1.0 - cover.to(torch.float32))  # miscoverage loss
        else:
            raise ValueError(f"Unknown risk type {self.cfg.EXP.RISK}.")
        return psi_loss # (NR_POINT_RISK_SAMP, psi_size)
    
    def get_valid_psi(self, psi_cand, stop_time, psi_select="max"):
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
            select_psi = torch.zeros(1)  # here psi=0 is always valid (full label set)
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
        choices=["miscover"],
        help="Desired loss function to compute associated risks.",
    )
    parser.add_argument(
        "--set_score",
        type=str,
        default=None,
        required=False,
        choices=["abs_res"],
        help="Type of interval score to use.",
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
        cfg.RUN.SUB_DIR = f"uci_cbm_{cfg.EXP.SET_SCORE}_{cfg.EXP.SPLIT_TIME}"
    if cfg.RUN.EXP_DIR == "auto":
        cfg.RUN.EXP_DIR = f"erc_{cfg.EXP.EPS}_{cfg.EXP.DELTA}_{cfg.EXP.RISK}_ts{cfg.EXP.NR_TIMESTEPS}_bts{cfg.EXP.BATCH_TIMESTEP}_tw{cfg.EXP.TRACKER_WINDOW[1]}"
    if cfg.RUN.LOAD_DIR == "auto": # Point loading dir to sub_dir if not specified
        cfg.RUN.LOAD_DIR = os.path.join(cfg.PROJECT.OUTPUT_DIR, exp_type, cfg.RUN.SUB_DIR)
    
    # Create experiment dir in sub_dir
    full_dir = os.path.join(
        cfg.PROJECT.OUTPUT_DIR,  # .../output_erc/
        exp_type,  # .../exp_cp/
        cfg.RUN.SUB_DIR,  # .../erm_fmow_top1_13/
        f"{cfg.RUN.EXP_DIR}{cfg.RUN.SUFFIX}"  # .../erc_0.1_0.05_miscover_ts1000_bts10/
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
    exp = ExpCP(cfg, logger)
    model = exp.load_model()
    data = exp.load_data()
    
    if cfg.RUN.GET_PRED:
        logger.info("Getting labels and predictions...")
        labs, preds = exp.get_pred(model, data)
        io_file.save_tensor(labs, "labs", cfg.RUN.LOAD_DIR)
        io_file.save_tensor(preds, "preds", cfg.RUN.LOAD_DIR)
        logger.info("Saved to file.")
    else:
        logger.info("Loading existing labels and predictions...")
        del model
        labs = io_file.load_tensor("labs", cfg.RUN.LOAD_DIR)
        preds = io_file.load_tensor("preds", cfg.RUN.LOAD_DIR)
        logger.info("Loaded to file.")
    
    logger.info("Starting test stream setting...")
     
    assert len(labs) == len(preds), "Mismatch labs vs. preds size"
    assert cfg.EXP.NR_TIMESTEPS <= len(labs), "Not enough time steps to run series"
     
    # Init stream vars valid for all trackers
    psi_cand = torch.arange(cfg.EXP.PSI_START, cfg.EXP.PSI_END + cfg.EXP.PSI_STEP, cfg.EXP.PSI_STEP)
    psi_size = len(psi_cand)
    stream_losses = torch.zeros((cfg.EXP.NR_TRIALS, cfg.EXP.NR_TIMESTEPS, psi_size))
    time_idx = 200 # sample from closest X time steps to simulate current P_t

    logger.info("Initializing trackers...")
    point_risk = PointRiskTracker(cfg, logger, psi_cand)
    running_risk = RunningRiskTracker(cfg, logger, psi_cand)
    eprocess = EProcessTracker(cfg, logger, psi_cand)
    naive_eprocess = NaiveEProcessTracker(cfg, logger, psi_cand)
    pmeb_eprocess = PMEBProcessTracker(cfg, logger, psi_cand)
    
    logger.info("Running experiment loop...")
    for tr in range(cfg.EXP.NR_TRIALS):
        for ts in tqdm(range(cfg.EXP.NR_TIMESTEPS), desc=f"Trial {tr+1}, Time step", leave=False):
            
            # update data range for sampling based on time
            if (ts + time_idx <= cfg.EXP.NR_TIMESTEPS):
                data_idx = (ts, ts + time_idx)
            
            # draw samples
            lab_batch, pred_batch = exp.draw_samples(data_idx, labs, preds)
            
            # get sample losses & point risk for full batch
            loss_batch = exp.compute_sample_losses(psi_cand, lab_batch, pred_batch)
            point_risk.risk[tr, ts, :] = point_risk.get_risk(loss_batch, explicit=False)
            
            # reduce batch and update stream vars
            loss_batch = exp.reduce_batch(loss_batch)
            stream_losses[tr, ts, :] = loss_batch.mean(dim=0) # (psi_size,)
            
            # get running risk
            running_risk.risk[tr, ts, :] = running_risk.get_risk(stream_losses[tr], ts)

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
                psi_cand, point_risk.stop_time[tr], psi_select="max"
            )
            point_risk.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            point_risk.psi_cs[tr][ts].append(valid_psi)
            
            running_risk.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, running_risk.stop_time[tr], psi_select="max"
            )
            running_risk.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            running_risk.psi_cs[tr][ts].append(valid_psi)
            
            eprocess.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, eprocess.stop_time[tr], psi_select="max"
            )
            eprocess.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            eprocess.psi_cs[tr][ts].append(valid_psi)
            
            naive_eprocess.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, naive_eprocess.stop_time[tr], psi_select="max"
            )
            naive_eprocess.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            naive_eprocess.psi_cs[tr][ts].append(valid_psi)
            
            pmeb_eprocess.psi_select[tr, ts], valid_psi = exp.get_valid_psi(
                psi_cand, pmeb_eprocess.stop_time[tr], psi_select="max"
            )
            pmeb_eprocess.psi_cs_size[tr, ts] = exp.get_psi_cs_size(valid_psi)
            pmeb_eprocess.psi_cs[tr][ts].append(valid_psi)
        
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
    
    
