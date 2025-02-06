import torch

from util import io_file


class PointRiskTracker():
    def __init__(self, cfg, logger, psi_cand):
        self.cfg = cfg
        self.logger = logger
        
        self.psi_size = len(psi_cand)
        self.burn_in = 0 # hardcode no burn-in since oracle
        self.tracker_window = self.cfg.EXP.TRACKER_WINDOW[0] # not used
        self.stop_counter = self.cfg.EXP.STOP_COUNTER[0] # used
        
        # init storage vars
        self.risk = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS, self.psi_size))
        self.stop_time = torch.full((self.cfg.EXP.NR_TRIALS, self.psi_size), -1.0)
        self.stop_time_robust_counter = torch.zeros((self.psi_size))
        self.psi_cs = [[[] for _ in range(self.cfg.EXP.NR_TIMESTEPS)] for _ in range(self.cfg.EXP.NR_TRIALS)]
        self.psi_cs_size = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.psi_select = torch.zeros((self.cfg.EXP.NR_TRIALS, self.cfg.EXP.NR_TIMESTEPS))
        self.false_alarms = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        self.detection_delay = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        
    def get_risk(self, loss_batch, explicit: bool = False):
        """
        get point risk using fully sampled batch or explicit formula
        """
        if self.cfg.EXP.RISK == "miscover":
            if explicit:
                pass
            else:
                risk = loss_batch.mean(dim=0)
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.EXP.RISK}")
        return risk
        
    def check_stop_time(self, stop_time, risk, ts):
        """
        check stopping condition of the point risk with robust counter for all psi candidates for one time step
        """
        if ts < self.burn_in:
            return stop_time
        
        null_rejections = torch.where(risk >= self.cfg.EXP.EPS)[0]
        
        if len(null_rejections) > 0:
            self.stop_time_robust_counter[null_rejections] += 1
            mask = (self.stop_time_robust_counter >= self.stop_counter) * (stop_time == -1)
            null_rejections_new = torch.where(mask)[0]  # new rejections (not yet stopped)
            
            if len(null_rejections_new) > 0:
                stop_time[null_rejections_new] = ts
        
        if ts == (self.cfg.EXP.NR_TIMESTEPS - 1):  # At the final time step, assign T to any index still marked as "not stopped"
            stop_time[stop_time == -1] = self.cfg.EXP.NR_TIMESTEPS
            self.stop_time_robust_counter = torch.zeros((self.psi_size)) # reset robust counter
        
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
        self.detection_delay = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        
    def get_risk(self, losses, ts):
        """
        get running risk for all psi candidates for one time step
        """
        window = self.tracker_window if self.tracker_window > 0 else ts
        ts_start = max(ts - window, 0)
        if self.cfg.EXP.RISK == "miscover":
            risk = losses[ts_start:(ts + 1), :].mean(dim=0)
        else:
            raise ValueError(f"Unknown loss type: {self.cfg.EXP.RISK}")
        return risk
    
    def check_stop_time(self, stop_time, risk, ts):
        """
        check stopping condition of the running risk with robust counter for all psi candidates for one time step
        """
        if ts < self.burn_in:
            return stop_time
        
        null_rejections = torch.where(risk >= self.cfg.EXP.EPS)[0]
        
        if len(null_rejections) > 0:
            self.stop_time_robust_counter[null_rejections] += 1
            mask = (self.stop_time_robust_counter >= self.stop_counter) * (stop_time == -1)
            null_rejections_new = torch.where(mask)[0]  # new rejections (not yet stopped)
            
            if len(null_rejections_new) > 0:
                stop_time[null_rejections_new] = ts
        
        if ts == (self.cfg.EXP.NR_TIMESTEPS - 1):  # At the final time step, assign T to any index still marked as "not stopped"
            stop_time[stop_time == -1] = self.cfg.EXP.NR_TIMESTEPS
            self.stop_time_robust_counter = torch.zeros((self.psi_size)) # reset robust counter
        
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
        self.detection_delay = torch.zeros((self.cfg.EXP.NR_TRIALS, self.psi_size))
        
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
        
        if ts == (self.cfg.EXP.NR_TIMESTEPS - 1):  # At the final time step, assign T to any index still marked as "not stopped"
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


class NaiveEProcessTracker(EProcessTracker): # # Inherit from EProcessTracker
    def __init__(self, cfg, logger, psi_cand):
        super().__init__(cfg, logger, psi_cand) 
    
    def get_evalues(self, loss_batch, bets, reduction="mean"):
        """
        get batch-wise e-values for one time step and reduce
        """
        e_val = bets * (loss_batch - self.cfg.EXP.EPS)
        if reduction == "mean":
            e_val = e_val.mean(dim=0)
        elif reduction == "prod":
            e_val = e_val.prod(dim=0)
        return e_val
    
    def get_eprocess(self, e_val, tr, ts):
        """
        Custom logic for e-process calculation, take sum instead of product
        """
        if ts == 0:
            e_process = torch.ones_like(e_val)
        else:
            e_process = self.eprocess[tr, ts - 1, :] + e_val
        return e_process


class PMEBProcessTracker(EProcessTracker): # # Inherit from EProcessTracker
    def __init__(self, cfg, logger, psi_cand):
        super().__init__(cfg, logger, psi_cand) 
    
    def get_bets(self, losses, ts):
        """
        get bet for e-process for one time step
        """
        window = self.tracker_window if self.tracker_window > 0 else ts
        ts_start = max(ts - window, 0)
        
        # Recommended plug-in for predictably-mixed Empirical Bernstein
        # see https://arxiv.org/pdf/2010.09686, sec 3.2
        # mu = losses[ts_start:ts].mean(dim=0) + 1e-10
        var = losses[ts_start:ts].var(dim=0) + 1e-10
        c = 0.5  # constant, recommended 0.5 or 0.75
        log_1 = torch.log(torch.tensor(2 / self.cfg.EXP.DELTA))
        log_2 = torch.log(torch.tensor(1 + ts))
        bet = torch.max(torch.tensor(0), torch.min(torch.sqrt((2 * log_1) / (var * ts * log_2)), torch.tensor(c)))
        
        if ts < self.burn_in:
            bet = torch.zeros_like(bet)
        
        return bet # (psi_size,)
    
    def get_evalues(self, losses, ts, loss_batch, bets, reduction="mean"):
        """
        get batch-wise e-values for one time step and reduce
        """
        window = self.tracker_window if self.tracker_window > 0 else ts
        ts_start = max(ts - window, 0)
        
        mu = losses[ts_start:ts].mean(dim=0) + 1e-10
        v = 4 * (loss_batch - mu) ** 2
        p = 0.25 * (- torch.log(1 - bets) - bets)
        e_val = torch.exp(bets * (loss_batch - self.cfg.EXP.EPS) - v * p)
        
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
    
