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
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
    
    