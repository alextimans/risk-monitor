import os

import numpy as np
from scipy.stats import binom
from tqdm import tqdm

import torch
import torch.nn as nn

from util import io_file, misc


# class Tracker:
#     def __init__(
#         self, 
#         cfg, 
#         logger, 
#         id_labels, 
#         id_preds, 
#         id_conf, 
#         ood_labels, 
#         ood_preds, 
#         ood_conf,
#         ood_probs,
#         psi_cand
#     ):
#         self.cfg = cfg
#         self.logger = logger
#         self.device = self.cfg.MODEL.DEVICE
        
#         self.id_labels = id_labels
#         self.id_preds = id_preds
#         self.id_conf = id_conf
#         self.ood_labels = ood_labels
#         self.ood_preds = ood_preds
#         self.ood_conf = ood_conf
        
#         self.ood_probs = ood_probs
#         self.psi_cand = psi_cand
    
#     def update_ood_prob(self, ood_prob_idx, t):
#         """
#         check if the ood probability should be updated and return the new value
#         """
#         if (t > 0) and (t % self.cfg.EXP.NR_OOD_TIMESTEPS == 0) and (t < len(self.ood_probs) * self.cfg.EXP.NR_OOD_TIMESTEPS):
#             ood_prob_idx += 1
#         return self.ood_probs[ood_prob_idx]
    
#     def draw_samples(self, ood_prob):
#         """
#         draw a batch of samples from id and ood data with a given ood probability
#         """
#         bern_batch = torch.from_numpy(
#             np.random.binomial(1, ood_prob, self.cfg.EXP.NR_POINT_RISK_SAMP)
#             )  # (NR_POINT_RISK_SAMP,)
        
#         nr_ood = bern_batch.sum().item()
#         nr_id = len(bern_batch) - nr_ood
#         ood_idx = np.random.randint(0, len(self.ood_labels), nr_ood)
#         id_idx = np.random.randint(0, len(self.id_labels), nr_id)
        
#         return (
#             torch.cat([self.id_labels[id_idx], self.ood_labels[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
#             torch.cat([self.id_preds[id_idx], self.ood_preds[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
#             torch.cat([self.id_conf[id_idx], self.ood_conf[ood_idx]]),  # (NR_POINT_RISK_SAMP,)
#         )
        
#     def compute_sample_losses(self, bern_batch, lab_batch, pred_batch, conf_batch):
#         """
#         get the loss for all psi candidates for a batch of samples
#         """
#         if self.cfg.EXP.RISK == "outlier_fpr_fnr":
#             psi_loss = torch.where(
#                 bern_batch.unsqueeze(1) == 0,  # if bern == 0 (ID sample)
#                 conf_batch.unsqueeze(1) > self.psi_cand,  # false positive
#                 conf_batch.unsqueeze(1) <= self.psi_cand  # false negative
#             )
#         elif self.cfg.EXP.RISK == "outlier_fnr":
#             psi_loss = torch.where(
#                 bern_batch.unsqueeze(1) == 0,
#                 torch.zeros_like(self.psi_cand),  # no penalty for false positive
#                 conf_batch.unsqueeze(1) <= self.psi_cand  # false negative
#             )
#         elif self.cfg.EXP.RISK == "outlier_fpr":
#             psi_loss = torch.where(
#                 bern_batch.unsqueeze(1) == 0,
#                 conf_batch.unsqueeze(1) > self.psi_cand,  # false positive
#                 torch.zeros_like(self.psi_cand)  # no penalty for false negative
#             )
#         else:
#             raise ValueError(f"Unknown risk type {self.cfg.EXP.RISK}.")
        
#         return psi_loss.to(torch.float32)
    
#     def get_valid_psi(self, stop_time, psi_select="min"):
#         """
#         get the set of valid psi candidates for one time step and return the selected psi
#         """
#         assert self.psi_cand.shape[0] == stop_time.shape[0], "Number of psi cand and stop times must match."
#         valid_psi = self.psi_cand[torch.where(stop_time == -1)[0]]  # not yet stopped
        
#         if len(valid_psi) > 0:
#             if psi_select == "min":
#                 select_psi = valid_psi.min()
#             elif psi_select == "max":
#                 select_psi = valid_psi.max()
#             elif psi_select == "median":
#                 select_psi = valid_psi.median()
#             else:
#                 raise ValueError(f"Unknown psi selection criterion {psi_select}.")
#         else:  # default to a 'trivial' safe zone
#             select_psi = torch.ones(1)
        
#         return select_psi, valid_psi.tolist()
    
#     def count_false_alarm(self, stop_time, true_stop_time):
#         """
#         count the false alarms for all psi candidates for one trial
#         """
#         false_alarms = torch.zeros(self.psi_size)
#         early_stops = torch.where((stop_time - true_stop_time) < 0)[0]
#         false_alarms[early_stops] = 1
#         return false_alarms


class PointRiskTracker():
    def __init__(self, cfg, logger, psi_cand):
        self.cfg = cfg
        self.logger = logger
        
        self.psi_size = len(psi_cand)
        self.burn_in = int(self.cfg.EXP.NR_BURNIN / self.cfg.EXP.BATCH_TIMESTEP)
        self.tracker_window = self.cfg.EXP.TRACKER_WINDOW[0] # not used
        self.stop_counter = self.cfg.EXP.STOP_COUNTER[0] # used
        
        # init storage vars
        self.risk = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS, self.psi_size))
        self.stop_time = torch.full((self.cfg.EXP.NR_TRIALS, self.psi_size), -1.0)
        self.psi_cs = [[[] for _ in range(self.cfg.EXP.NR_TIMESTEPS)] for _ in range(self.cfg.EXP.NR_TRIALS)]
        self.psi_cs_size = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.psi_select = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.false_alarms = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        self.detection_delay = torch.zeros((self.cfg.EXP_NR_TRIALS, self.psi_size))
        
    def get_risk(self, loss_batch, bern_batch, explicit: bool = False):
        """
        get point risk using fully sampled batch or explicit formula
        """
        if self.cfg.EXP.RISK == "fpr_fnr":
            if explicit:
                pass
            else:
                risk = loss_batch.mean(dim=0)
        elif self.cfg.EXP.RISK == "fnr":
            if explicit:
                pass
            else:
                nr_out = bern_batch.sum() + 1e-10
                risk = loss_batch.mean(dim=0) / nr_out
        elif self.cfg.EXP.RISK == "fpr":
            if explicit:
                pass
            else:
                nr_in = ((bern_batch.shape[0] - bern_batch.sum()) + 1e-10)
                risk = loss_batch.mean(dim=0) / nr_in
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.EXP.RISK}")
        return risk
        
    def check_stop_time(self, stop_time, risk, ts):
        """
        check stopping condition of the point risk with robust counter for all psi candidates for one time step
        """
        null_rejections = torch.where(risk >= self.cfg.EXP.EPS)[0]
        
        if len(null_rejections) > 0:
            self.stop_time_robust_counter[null_rejections] += 1
            mask = (self.stop_time_robust_counter >= self.stop_counter) * (stop_time == -1)
            null_rejections_new = torch.where(mask)[0]  # new rejections (not yet stopped)
            
            if len(null_rejections_new) > 0:
                stop_time[null_rejections_new] = ts
        
        if ts == self.cfg.EXP.NR_TIMESTEPS:  # At the final time step, assign T to any index still marked as "not stopped"
            stop_time[stop_time == -1] = self.cfg.EXP.NR_TIMESTEPS
        
        return stop_time
    
    def save_to_file(self, filename, filedir):
        save_dict = {
            "risk": self.risk,
            "stop_time": self.stop_time,
            "psi_cs": self.psi_cs,
            "psi_cs_size": self.psi_cs_size,
            "psi_select": self.psi_select,
            "false_alarms": self.false_alarms,
            "detection_delay": self.detection_delay
        }
        io_file.save_tensor(save_dict, filename, filedir)
        

class RunningRiskTracker():
    def __init__(self, cfg, logger, psi_cand):
        self.cfg = cfg
        self.logger = logger
        
        self.psi_size = len(psi_cand)
        self.burn_in = int(self.cfg.EXP.NR_BURNIN / self.cfg.EXP.BATCH_TIMESTEP)
        self.tracker_window = self.cfg.EXP.TRACKER_WINDOW[1] # used
        self.stop_counter = self.cfg.EXP.STOP_COUNTER[1] # used
        
        # init storage vars
        self.risk = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS, self.psi_size))
        self.stop_time = torch.full((self.cfg.EXP.NR_TRIALS, self.psi_size), -1.0)
        self.stop_time_robust_counter = torch.zeros((self.psi_size))
        self.psi_cs = [[[] for _ in range(self.cfg.EXP.NR_TIMESTEPS)] for _ in range(self.cfg.EXP.NR_TRIALS)]
        self.psi_cs_size = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.psi_select = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.false_alarms = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        self.detection_delay = torch.zeros((self.cfg.EXP_NR_TRIALS, self.psi_size))
        
    def get_risk(self, losses, bern, ts):
        """
        get running risk for all psi candidates for one time step
        """
        window = self.tracker_window if self.tracker_window > 0 else ts
        ts_start = max(ts - window, 0)
        if self.cfg.EXP.RISK == "fpr_fnr":
            risk = losses[ts_start:(ts + 1), :].mean(dim=0)
        elif self.cfg.EXP.RISK == "fnr":
            nr_out = bern[ts_start:(ts + 1), :].sum() + 1e-10
            risk = losses[ts_start:(ts + 1), :].mean(dim=0) / nr_out
        elif self.cfg.EXP.RISK == "fpr":
            nr_in = bern[ts_start:(ts + 1), :].eq(0).sum() + 1e-10
            risk = losses[ts_start:(ts + 1), :].mean(dim=0) / nr_in
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.EXP.RISK}")
        return risk
    
    def check_stop_time(self, stop_time, risk, ts):
        """
        check stopping condition of the running risk with robust counter for all psi candidates for one time step
        """
        null_rejections = torch.where(risk >= self.cfg.EXP.EPS)[0]
        
        if len(null_rejections) > 0:
            self.stop_time_robust_counter[null_rejections] += 1
            mask = (self.stop_time_robust_counter >= self.stop_counter) * (stop_time == -1)
            null_rejections_new = torch.where(mask)[0]  # new rejections (not yet stopped)
            
            if len(null_rejections_new) > 0:
                stop_time[null_rejections_new] = ts
        
        if ts == self.cfg.EXP.NR_TIMESTEPS:  # At the final time step, assign T to any index still marked as "not stopped"
            stop_time[stop_time == -1] = self.cfg.EXP.NR_TIMESTEPS
        
        return stop_time
                
    # robust_null_rejections = torch.where(self.stop_time_robust_counter >= self.stop_counter)[0]
    
    # if len(robust_null_rejections) > 0:
    #     null_rejections_new = robust_null_rejections[stop_time[robust_null_rejections] == -1] # new rejections (not yet stopped)
        
    #     if len(null_rejections_new) > 0:
    #         stop_time[null_rejections_new] = ts
    
    # if ts == self.cfg.EXP.NR_TIMESTEPS:  # At the final time step, assign T to any index still marked as "not stopped"
    #     stop_time[stop_time == -1] = self.cfg.EXP.NR_TIMESTEPS
    
    # return stop_time
    
    # def check_stop_time(self, risk):
    #     stop_time = torch.zeros(self.psi_size)
    #     for psi_idx in range(self.psi_size):
            
    #         # Identify crossings after the burn-in period
    #         null_rejections = torch.where(risk[:, psi_idx] >= self.cfg.EXP.EPS)[0]
    #         burnin_null_rejections = null_rejections[null_rejections >= 0] # hardcode burn_in=0
    #         nr_rejections = len(burnin_null_rejections)
            
    #         if nr_rejections > 0:  # At least one crossing
    #             if nr_rejections >= self.stop_lookback:  # Check for robust crossings
    #                 # Sliding window check for `stop_lookback` consecutive integers
    #                 for i in range(nr_rejections - self.stop_lookback + 1):
    #                     start = burnin_null_rejections[i]
    #                     if torch.all(burnin_null_rejections[i:i + self.stop_lookback] == torch.arange(start, start + self.stop_lookback)):
    #                         stop_time[psi_idx] = burnin_null_rejections[i + self.stop_lookback]
    #                         break
    #                 else:
    #                     # No robust crossings, use the first crossing
    #                     stop_time[psi_idx] = burnin_null_rejections[0]
    #             else:
    #                 # Not enough crossings, use the first crossing
    #                 stop_time[psi_idx] = burnin_null_rejections[0]
    #         else:
    #             # No crossings, stop at the final time step
    #             stop_time[psi_idx] = self.cfg.EXP.NR_TIMESTEPS
        
    #     return stop_time
    
    def save_to_file(self, filename, filedir):
        save_dict = {
            "risk": self.risk,
            "stop_time": self.stop_time,
            "psi_cs": self.psi_cs,
            "psi_cs_size": self.psi_cs_size,
            "psi_select": self.psi_select,
            "false_alarms": self.false_alarms,
            "detection_delay": self.detection_delay
        }
        io_file.save_tensor(save_dict, filename, filedir)
        

class EProcessTracker():
    def __init__(self, cfg, logger, psi_cand):
        self.cfg = cfg
        self.logger = logger

        self.psi_size = len(psi_cand)
        self.burn_in = int(self.cfg.EXP.NR_BURNIN / self.cfg.EXP.BATCH_TIMESTEP)
        self.tracker_window = self.cfg.EXP.TRACKER_WINDOW[2] # used
        self.stop_counter = self.cfg.EXP.STOP_COUNTER[2] # not used
        
        # init storage vars
        self.bets = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS, self.psi_size))
        self.evalues = torch.ones((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS, self.psi_size))
        self.eprocess = torch.ones((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS, self.psi_size))
        
        self.stop_time = torch.full((self.cfg.EXP.NR_TRIALS, self.psi_size), -1.0)
        self.psi_cs = [[[] for _ in range(self.cfg.EXP.NR_TIMESTEPS)] for _ in range(self.cfg.EXP.NR_TRIALS)]
        self.psi_cs_size = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.psi_select = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.false_alarms = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        self.detection_delay = torch.zeros((self.cfg.EXP_NR_TRIALS, self.psi_size))
        
    def get_bets(self, losses, ts):
        """
        get bet for e-process for one time step
        """
        window = self.tracker_window if self.tracker_window > 0 else ts
        ts_start = max(ts - window, 0)
        
        if self.cfg.EXP.BET_TYPE == "unit_bet":
            bet = torch.ones_like(losses[ts])
        
        elif self.cfg.EXP.BET_TYPE == "approx_grapa":  # see https://arxiv.org/pdf/2010.09686, sec B.3
            mu = losses[ts_start:ts].mean(dim=0) + 1e-10
            var = losses[ts_start:ts].var(dim=0) + 1e-10
            c = 0.5  # constant, recommended 0.5 or 0.75
            term = (mu - self.cfg.EXP.EPS) / (var + (mu - self.cfg.EXP.EPS)**2) + 1e-10
            bet = torch.max(torch.tensor(0), torch.min(term, torch.tensor(c / self.cfg.EXP.EPS)))
        
        elif self.cfg.EXP.BET_TYPE == "grapa":  # see https://arxiv.org/pdf/2010.09686, sec B.2
            lam_cand = torch.arange(0.1, (1.0 / self.cfg.EXP.EPS), 0.1)
            lam_growth = torch.ones((len(lam_cand), self.psi_size))
            for lam_idx, lam in enumerate(lam_cand):
                growth = torch.log(torch.ones(self.psi_size) + lam * (losses[ts_start:ts] - self.cfg.EXP.EPS))
                lam_growth[lam_idx, :] = torch.mean(growth, dim=0)
            bet = lam_cand[lam_growth.argmax(dim=0)]
        
        else:
            raise ValueError(f"Unknown betting type: {self.cfg.EXP.BET_TYPE}")
        
        if ts < self.burn_in:
            bet = torch.zeros_like(bet)
        
        return bet # (psi_size,)
    
    def get_evalues(self, loss_batch, bets, reduction="mean"):
        """
        get batch-wise e-values for one time step and reduce
        """
        e_val = 1.0 + bets * (loss_batch - self.cfg.EXP.EPS)
        if reduction == "mean":
            e_val = e_val.mean(dim=0)
        elif reduction == "prod":
            e_val = e_val.prod(dim=0)
        return e_val
    
    def get_eprocess(self, e_val, tr, ts):
        if ts == 0:
            e_process = torch.ones_like(e_val)
        else:
            e_process = self.eprocess[tr, ts - 1, :] * e_val
        return e_process
    
    def check_stop_time(self, stop_time, e_process, ts):
        """
        check stopping condition of the e-process for all psi candidates for one time step
        """
        null_rejections = torch.where(e_process >= 1 / self.cfg.EXP.DELTA)[0]
        
        if len(null_rejections) > 0:
            null_rejections_new = null_rejections[stop_time[null_rejections] == -1] # new rejections (not yet stopped)

            if len(null_rejections_new) > 0:
                stop_time[null_rejections_new] = ts
        
        if ts == self.cfg.EXP.NR_TIMESTEPS:  # At the final time step, assign T to any index still marked as "not stopped"
            stop_time[stop_time == -1] = self.cfg.EXP.NR_TIMESTEPS
        
        return stop_time    
    
    def save_to_file(self, filename, filedir):
        save_dict = {
            "bets": self.bets,
            "evalues": self.evalues,
            "eprocess": self.eprocess,
            "stop_time": self.stop_time,
            "psi_cs": self.psi_cs,
            "psi_cs_size": self.psi_cs_size,
            "psi_select": self.psi_select,
            "false_alarms": self.false_alarms,
            "detection_delay": self.detection_delay
        }
        io_file.save_tensor(save_dict, filename, filedir)
