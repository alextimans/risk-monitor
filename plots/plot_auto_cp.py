
import os
import torch
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt


def save_fig(logger, fdir, fname):
    full_path = os.path.join(fdir, fname)
    plt.savefig(full_path, bbox_inches='tight')
    logger.info(f'Figure saved under "{full_path}".')


def plot_risk(
    cfg,
    logger,
    psi_cand,
    point_risk,
    running_risk,
    eprocess,
    naive_eprocess,
    pmeb_eprocess,
    plot_psi=5
):
    plot_psi_idx = torch.arange(
        start=int(cfg.EXP.PSI_START), 
        end=len(psi_cand), 
        step=int((len(psi_cand) - 1) / plot_psi),
        dtype=torch.int
    )
    
    # Point risk
    plt.figure(figsize=(7, 3))
    for p in plot_psi_idx:
        point_risk_mean = point_risk.risk[:, :, p].mean(dim=0).numpy()
        point_risk_std = point_risk.risk[:, :, p].std(dim=0).numpy()
        plt.plot(point_risk_mean, label=fr"$R_t(\psi = {psi_cand[p]:.2f})$")
        plt.fill_between(range(cfg.EXP.NR_TIMESTEPS), 
                         point_risk_mean - point_risk_std, 
                         point_risk_mean + point_risk_std, 
                         alpha=0.2)
    plt.axhline(cfg.EXP.EPS, color='red', linestyle='--', label=rf"$\epsilon$ = {cfg.EXP.EPS}")
    plt.axvline(cfg.EXP.NR_BURNIN, color='black', linestyle='-', label="Burn-in period")
    plt.title(f"Point risk over time ({cfg.EXP.NR_POINT_RISK_SAMP} samples, {cfg.EXP.NR_TRIALS} trials)")
    plt.xlabel("Time step")
    plt.ylabel("Risk")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0, cfg.EXP.NR_TIMESTEPS)
    plt.ylim(0, 0.25)
    plt.grid(True)
    save_fig(logger, cfg.RUN.PLOT_DIR, "point_risk_over_time")

    # Running risk
    plt.figure(figsize=(7, 3))
    for p in plot_psi_idx:
        running_risk_mean = running_risk.risk[:, :, p].mean(dim=0).numpy()
        running_risk_std = running_risk.risk[:, :, p].std(dim=0).numpy()
        plt.plot(running_risk_mean, label=fr"$R_t(\psi = {psi_cand[p]:.2f})$")
        plt.fill_between(range(cfg.EXP.NR_TIMESTEPS), 
                         running_risk_mean - running_risk_std, 
                         running_risk_mean + running_risk_std, 
                         alpha=0.2)
    plt.axhline(cfg.EXP.EPS, color='red', linestyle='--', label=rf"$\epsilon$ = {cfg.EXP.EPS}")
    plt.axvline(cfg.EXP.NR_BURNIN, color='black', linestyle='-', label="Burn-in period")
    plt.title(f"Running risk over time ({cfg.EXP.NR_TRIALS} trials)")
    plt.xlabel("Time step")
    plt.ylabel("Risk")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0, cfg.EXP.NR_TIMESTEPS)
    plt.ylim(0, 0.25)
    plt.grid(True)
    save_fig(logger, cfg.RUN.PLOT_DIR, "running_risk_over_time")
    
    # Eprocess
    plt.figure(figsize=(7, 3))
    for p in plot_psi_idx:
        eprocess_mean = eprocess.eprocess[:, :, p].mean(dim=0).numpy()
        eprocess_std = eprocess.eprocess[:, :, p].std(dim=0).numpy()
        plt.plot(eprocess_mean, label=fr"$W_t(\psi = {psi_cand[p]:.2f})$")
        plt.fill_between(range(cfg.EXP.NR_TIMESTEPS), 
                         eprocess_mean - eprocess_std, 
                         eprocess_mean + eprocess_std, 
                         alpha=0.2)
    plt.axhline(1 / cfg.EXP.DELTA, color='red', linestyle='--', label=rf"$1/\delta$ = {1 / cfg.EXP.DELTA}")
    plt.axvline(cfg.EXP.NR_BURNIN, color='black', linestyle='-', label="Burn-in period")
    plt.title(f"Wealth process over time ({cfg.EXP.NR_TRIALS} trials)")
    plt.xlabel("Time step")
    plt.ylabel("Wealth")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0, cfg.EXP.NR_TIMESTEPS)
    plt.ylim(0, 1 / cfg.EXP.DELTA + 0.5)
    plt.grid(True)
    save_fig(logger, cfg.RUN.PLOT_DIR, "eprocess_over_time")
    
    # Naive Eprocess
    plt.figure(figsize=(7, 3))
    for p in plot_psi_idx:
        naive_eprocess_mean = naive_eprocess.eprocess[:, :, p].mean(dim=0).numpy()
        naive_eprocess_std = naive_eprocess.eprocess[:, :, p].std(dim=0).numpy()
        plt.plot(naive_eprocess_mean, label=fr"$W_t(\psi = {psi_cand[p]:.2f})$")
        plt.fill_between(range(cfg.EXP.NR_TIMESTEPS), 
                         naive_eprocess_mean - naive_eprocess_std, 
                         naive_eprocess_mean + naive_eprocess_std, 
                         alpha=0.2)
    plt.axhline(1 / cfg.EXP.DELTA, color='red', linestyle='--', label=rf"$1/\delta$ = {1 / cfg.EXP.DELTA}")
    plt.axvline(cfg.EXP.NR_BURNIN, color='black', linestyle='-', label="Burn-in period")
    plt.title(f"Naive wealth process over time ({cfg.EXP.NR_TRIALS} trials)")
    plt.xlabel("Time step")
    plt.ylabel("Wealth")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0, cfg.EXP.NR_TIMESTEPS)
    plt.ylim(0, 1 / cfg.EXP.DELTA + 0.5)
    plt.grid(True)
    save_fig(logger, cfg.RUN.PLOT_DIR, "naive_eprocess_over_time")

    # PMEB Eprocess
    plt.figure(figsize=(7, 3))
    for p in plot_psi_idx:
        pmeb_eprocess_mean = pmeb_eprocess.eprocess[:, :, p].mean(dim=0).numpy()
        pmeb_eprocess_std = pmeb_eprocess.eprocess[:, :, p].std(dim=0).numpy()
        plt.plot(pmeb_eprocess_mean, label=fr"$W_t(\psi = {psi_cand[p]:.2f})$")
        plt.fill_between(range(cfg.EXP.NR_TIMESTEPS), 
                         pmeb_eprocess_mean - pmeb_eprocess_std, 
                         pmeb_eprocess_mean + pmeb_eprocess_std, 
                         alpha=0.2)
    plt.axhline(1 / cfg.EXP.DELTA, color='red', linestyle='--', label=rf"$1/\delta$ = {1 / cfg.EXP.DELTA}")
    plt.axvline(cfg.EXP.NR_BURNIN, color='black', linestyle='-', label="Burn-in period")
    plt.title(f"PMEB wealth process over time ({cfg.EXP.NR_TRIALS} trials)")
    plt.xlabel("Time step")
    plt.ylabel("Wealth")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlim(0, cfg.EXP.NR_TIMESTEPS)
    plt.ylim(0, 1 / cfg.EXP.DELTA + 0.5)
    plt.grid(True)
    save_fig(logger, cfg.RUN.PLOT_DIR, "pmeb_eprocess_over_time")


def plot_cum_losses(
    cfg,
    logger,
    psi_cand,
    stream_losses
):
    plt.figure(figsize=(7, 3))
    loss_mean = stream_losses.mean(dim=0).sum(dim=0).numpy()
    loss_std = stream_losses.std(dim=0).sum(dim=0).numpy()
    plt.plot(psi_cand, loss_mean)
    plt.fill_between(psi_cand, 
                     loss_mean - loss_std, 
                     loss_mean + loss_std, 
                     alpha=0.2)
    plt.xlabel(r"Candidate $\psi$")
    plt.ylabel("Cumulative Losses")
    plt.title(f"Cumulative Losses across Thresholds ({cfg.EXP.NR_TRIALS} trials)")
    plt.grid(True)
    save_fig(logger, cfg.RUN.PLOT_DIR, "cum_losses_over_psi")


def plot_stop_times(
    cfg,
    logger,
    psi_cand,
    point_risk,
    running_risk,
    eprocess,
    naive_eprocess,
    pmeb_eprocess
):
    fig, ax = plt.subplots(figsize=(7, 3))
    stop_point_risk_mean = point_risk.stop_time.mean(dim=0)
    stop_point_risk_std = point_risk.stop_time.std(dim=0)
    ax.plot(psi_cand, stop_point_risk_mean, label="Point Risk")
    ax.fill_between(psi_cand,
                    stop_point_risk_mean - stop_point_risk_std,
                    stop_point_risk_mean + stop_point_risk_std,
                    alpha=0.2)
    stop_running_risk_mean = running_risk.stop_time.mean(dim=0)
    stop_running_risk_std = running_risk.stop_time.std(dim=0)
    ax.plot(psi_cand, stop_running_risk_mean, label="Running Risk")
    ax.fill_between(psi_cand,
                    stop_running_risk_mean - stop_running_risk_std,
                    stop_running_risk_mean + stop_running_risk_std,
                    alpha=0.2)
    stop_eprocess_mean = eprocess.stop_time.mean(dim=0)
    stop_eprocess_std = eprocess.stop_time.std(dim=0)
    ax.plot(psi_cand, stop_eprocess_mean, label="E-process")
    ax.fill_between(psi_cand,
                    stop_eprocess_mean - stop_eprocess_std,
                    stop_eprocess_mean + stop_eprocess_std,
                    alpha=0.2)
    stop_naive_eprocess_mean = naive_eprocess.stop_time.mean(dim=0)
    stop_naive_eprocess_std = naive_eprocess.stop_time.std(dim=0)
    ax.plot(psi_cand, stop_naive_eprocess_mean, label="Naive E-process")
    ax.fill_between(psi_cand,
                    stop_naive_eprocess_mean - stop_naive_eprocess_std,
                    stop_naive_eprocess_mean + stop_naive_eprocess_std,
                    alpha=0.2)
    stop_pmeb_eprocess_mean = pmeb_eprocess.stop_time.mean(dim=0)
    stop_pmeb_eprocess_std = pmeb_eprocess.stop_time.std(dim=0)
    ax.plot(psi_cand, stop_pmeb_eprocess_mean, label="PMEB E-process")
    ax.fill_between(psi_cand,
                    stop_pmeb_eprocess_mean - stop_pmeb_eprocess_std,
                    stop_pmeb_eprocess_mean + stop_pmeb_eprocess_std,
                    alpha=0.2)
    
    ax.set_xlabel(r"Candidate $\psi$")
    ax.set_ylabel("Stopping Time")
    ax.set_title(f"Stopping Times across Thresholds ({cfg.EXP.NR_TRIALS} trials)")    
    ax.grid(True)
    ax.legend()
    save_fig(logger, cfg.RUN.PLOT_DIR, "stopping_times_over_psi")


def get_default_psi(cs_size):
    ind = torch.where(torch.isclose(cs_size, torch.tensor(0.0)))[0]
    if len(ind) > 0:
        return ind[0]
    else:
        return len(cs_size)


def plot_cs(
    cfg,
    logger,
    psi_cand,
    point_risk,
    running_risk,
    eprocess,
    naive_eprocess,
    pmeb_eprocess
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    
    # SUBFIG 1: CS sizes
    point_risk_cs_size_mean = point_risk.psi_cs_size.mean(dim=0)
    point_risk_cs_size_std = point_risk.psi_cs_size.std(dim=0)
    point_risk_psi_default = get_default_psi(point_risk_cs_size_mean)
    axes[0].plot(point_risk_cs_size_mean, label="Point Risk")
    axes[0].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         point_risk_cs_size_mean - point_risk_cs_size_std,
                         point_risk_cs_size_mean + point_risk_cs_size_std,
                         alpha=0.2)
    running_risk_cs_size_mean = running_risk.psi_cs_size.mean(dim=0)
    running_risk_cs_size_std = running_risk.psi_cs_size.std(dim=0)
    running_risk_psi_default = get_default_psi(running_risk_cs_size_mean)
    axes[0].plot(running_risk_cs_size_mean, label="Running Risk")
    axes[0].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         running_risk_cs_size_mean - running_risk_cs_size_std,
                         running_risk_cs_size_mean + running_risk_cs_size_std,
                         alpha=0.2)
    eprocess_cs_size_mean = eprocess.psi_cs_size.mean(dim=0)
    eprocess_cs_size_std = eprocess.psi_cs_size.std(dim=0)
    eprocess_psi_default = get_default_psi(eprocess_cs_size_mean)
    axes[0].plot(eprocess_cs_size_mean, label="E-process")
    axes[0].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         eprocess_cs_size_mean - eprocess_cs_size_std,
                         eprocess_cs_size_mean + eprocess_cs_size_std,
                         alpha=0.2)
    naive_eprocess_cs_size_mean = naive_eprocess.psi_cs_size.mean(dim=0)
    naive_eprocess_cs_size_std = naive_eprocess.psi_cs_size.std(dim=0)
    # naive_eprocess_psi_default = get_default_psi(naive_eprocess_cs_size_mean)
    axes[0].plot(naive_eprocess_cs_size_mean, label="Naive E-process")
    axes[0].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         naive_eprocess_cs_size_mean - naive_eprocess_cs_size_std,
                         naive_eprocess_cs_size_mean + naive_eprocess_cs_size_std,
                         alpha=0.2)
    pmeb_eprocess_cs_size_mean = pmeb_eprocess.psi_cs_size.mean(dim=0)
    pmeb_eprocess_cs_size_std = pmeb_eprocess.psi_cs_size.std(dim=0)
    # pmeb_eprocess_psi_default = get_default_psi(pmeb_eprocess_cs_size_mean)
    axes[0].plot(pmeb_eprocess_cs_size_mean, label="PMEB E-process")
    axes[0].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         pmeb_eprocess_cs_size_mean - pmeb_eprocess_cs_size_std,
                         pmeb_eprocess_cs_size_mean + pmeb_eprocess_cs_size_std,
                         alpha=0.2)
    
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel(r"Size of $\psi$-CS")
    axes[0].set_title(rf"Size of $\psi$-CS over Time ({cfg.EXP.NR_TRIALS} trials)")
    axes[0].set_xlim(0, min(max(point_risk_psi_default, running_risk_psi_default, eprocess_psi_default) + 100, cfg.EXP.NR_TIMESTEPS))
    axes[0].grid(True)
    axes[0].legend()
    
    # SUBFIG 2: Selected psi
    point_risk_psi_select_mean = point_risk.psi_select.mean(dim=0)
    point_risk_psi_select_std = point_risk.psi_select.std(dim=0)
    # axes[1].axvline(point_risk_psi_default, linestyle='-', label="No valid $\psi$")
    axes[1].plot(point_risk_psi_select_mean, label="Point Risk")
    axes[1].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         point_risk_psi_select_mean - point_risk_psi_select_std,
                         point_risk_psi_select_mean + point_risk_psi_select_std,
                         alpha=0.2)
    running_risk_psi_select_mean = running_risk.psi_select.mean(dim=0)
    running_risk_psi_select_std = running_risk.psi_select.std(dim=0)
    # axes[1].axvline(running_risk_psi_default, linestyle='-', label="No valid $\psi$")
    axes[1].plot(running_risk_psi_select_mean, label="Running Risk")
    axes[1].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         running_risk_psi_select_mean - running_risk_psi_select_std,
                         running_risk_psi_select_mean + running_risk_psi_select_std,
                         alpha=0.2)
    eprocess_psi_select_mean = eprocess.psi_select.mean(dim=0)
    eprocess_psi_select_std = eprocess.psi_select.std(dim=0)
    # axes[1].axvline(eprocess_psi_default, linestyle='-', label="No valid $\psi$")
    axes[1].plot(eprocess_psi_select_mean, label="E-process")
    axes[1].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         eprocess_psi_select_mean - eprocess_psi_select_std,
                         eprocess_psi_select_mean + eprocess_psi_select_std,
                         alpha=0.2)
    naive_eprocess_psi_select_mean = naive_eprocess.psi_select.mean(dim=0)
    naive_eprocess_psi_select_std = naive_eprocess.psi_select.std(dim=0)
    # axes[1].axvline(naive_eprocess_psi_default, linestyle='-', label="No valid $\psi$")
    axes[1].plot(naive_eprocess_psi_select_mean, label="Naive E-process")
    axes[1].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         naive_eprocess_psi_select_mean - naive_eprocess_psi_select_std,
                         naive_eprocess_psi_select_mean + naive_eprocess_psi_select_std,
                         alpha=0.2)
    pmeb_eprocess_psi_select_mean = pmeb_eprocess.psi_select.mean(dim=0)
    pmeb_eprocess_psi_select_std = pmeb_eprocess.psi_select.std(dim=0)
    # axes[1].axvline(pmeb_eprocess_psi_default, linestyle='-', label="No valid $\psi$")
    axes[1].plot(pmeb_eprocess_psi_select_mean, label="PMEB E-process")
    axes[1].fill_between(range(cfg.EXP.NR_TIMESTEPS),
                         pmeb_eprocess_psi_select_mean - pmeb_eprocess_psi_select_std,
                         pmeb_eprocess_psi_select_mean + pmeb_eprocess_psi_select_std,
                         alpha=0.2)
    
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel(r"Selected $\hat{\psi}_t$")
    axes[1].set_title(r"Selected Threshold $\hat{\psi}_t$ over Time")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_xlim(0, min(max(point_risk_psi_default, running_risk_psi_default, eprocess_psi_default) + 100, cfg.EXP.NR_TIMESTEPS))
    axes[1].set_ylim(psi_cand[0] - 0.05, psi_cand[-1] + 0.05)
    
    # SUBFIG 3: Detection delay over psi
    point_risk_detection_delay_mean = point_risk.detection_delay.mean(dim=0)
    point_risk_detection_delay_std = point_risk.detection_delay.std(dim=0)
    axes[2].plot(psi_cand, point_risk_detection_delay_mean, label="Point Risk")
    axes[2].fill_between(psi_cand,
                         point_risk_detection_delay_mean - point_risk_detection_delay_std,
                         point_risk_detection_delay_mean + point_risk_detection_delay_std,
                         alpha=0.2)
    running_risk_detection_delay_mean = running_risk.detection_delay.mean(dim=0)
    running_risk_detection_delay_std = running_risk.detection_delay.std(dim=0)
    axes[2].plot(psi_cand, running_risk_detection_delay_mean, label="Running Risk")
    axes[2].fill_between(psi_cand,
                         running_risk_detection_delay_mean - running_risk_detection_delay_std,
                         running_risk_detection_delay_mean + running_risk_detection_delay_std,
                         alpha=0.2)
    eprocess_detection_delay_mean = eprocess.detection_delay.mean(dim=0)
    eprocess_detection_delay_std = eprocess.detection_delay.std(dim=0)
    axes[2].plot(psi_cand, eprocess_detection_delay_mean, label="E-process")
    axes[2].fill_between(psi_cand,
                         eprocess_detection_delay_mean - eprocess_detection_delay_std,
                         eprocess_detection_delay_mean + eprocess_detection_delay_std,
                         alpha=0.2)
    naive_eprocess_detection_delay_mean = naive_eprocess.detection_delay.mean(dim=0)
    naive_eprocess_detection_delay_std = naive_eprocess.detection_delay.std(dim=0)
    axes[2].plot(psi_cand, naive_eprocess_detection_delay_mean, label="Naive E-process")
    axes[2].fill_between(psi_cand,
                         naive_eprocess_detection_delay_mean - naive_eprocess_detection_delay_std,
                         naive_eprocess_detection_delay_mean + naive_eprocess_detection_delay_std,
                         alpha=0.2)
    pmeb_eprocess_detection_delay_mean = pmeb_eprocess.detection_delay.mean(dim=0)
    pmeb_eprocess_detection_delay_std = pmeb_eprocess.detection_delay.std(dim=0)
    axes[2].plot(psi_cand, pmeb_eprocess_detection_delay_mean, label="PMEB E-process")
    axes[2].fill_between(psi_cand,
                         pmeb_eprocess_detection_delay_mean - pmeb_eprocess_detection_delay_std,
                         pmeb_eprocess_detection_delay_mean + pmeb_eprocess_detection_delay_std,
                         alpha=0.2)
    
    axes[2].set_xlabel(r"Candidate $\psi$")
    axes[2].set_ylabel("Detection Delay")
    axes[2].set_title("Detection Delay across Thresholds")
    axes[2].grid(True)
    axes[2].legend()
    
    save_fig(logger, cfg.RUN.PLOT_DIR, "cs_over_psi")


def plot_false_alarms(
    cfg,
    logger,
    psi_cand,
    point_risk,
    running_risk,
    eprocess,
    naive_eprocess,
    pmeb_eprocess
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    
    # SUBFIG 1: Detection delay histogram
    point_risk_detection_delay = point_risk.detection_delay.flatten().numpy()
    running_risk_detection_delay = running_risk.detection_delay.flatten().numpy()
    eprocess_detection_delay = eprocess.detection_delay.flatten().numpy()
    naive_eprocess_detection_delay = naive_eprocess.detection_delay.flatten().numpy()
    pmeb_eprocess_detection_delay = pmeb_eprocess.detection_delay.flatten().numpy()
    
    axes[0].hist(point_risk_detection_delay, bins=30, alpha=0.5, edgecolor='black', density=False, label=f"Point Risk (Avg: {point_risk_detection_delay.mean():.2f})")
    axes[0].hist(running_risk_detection_delay, bins=30, alpha=0.5, edgecolor='black', density=False, label=f"Running Risk (Avg: {running_risk_detection_delay.mean():.2f})")
    axes[0].hist(eprocess_detection_delay, bins=30, alpha=0.5, edgecolor='black', density=False, label=f"E-process (Avg: {eprocess_detection_delay.mean():.2f})")
    axes[0].hist(naive_eprocess_detection_delay, bins=30, alpha=0.5, edgecolor='black', density=False, label=f"Naive E-process (Avg: {naive_eprocess_detection_delay.mean():.2f})")
    axes[0].hist(pmeb_eprocess_detection_delay, bins=30, alpha=0.5, edgecolor='black', density=False, label=f"PMEB E-process (Avg: {pmeb_eprocess_detection_delay.mean():.2f})")
    
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlabel("Detection Delay")
    axes[0].legend()
    axes[0].set_title(f"Detection Delays across Thresholds (and {cfg.EXP.NR_TRIALS} Trials)")

    # SUBFIG 2: False Alarm histogram
    point_risk_false_alarms = point_risk.false_alarms.mean(dim=0).numpy()
    running_risk_false_alarms = running_risk.false_alarms.mean(dim=0).numpy()
    eprocess_false_alarms = eprocess.false_alarms.mean(dim=0).numpy()
    naive_eprocess_false_alarms = naive_eprocess.false_alarms.mean(dim=0).numpy()
    pmeb_eprocess_false_alarms = pmeb_eprocess.false_alarms.mean(dim=0).numpy()
    
    running_risk_false_alarms_frac = (running_risk_false_alarms > cfg.EXP.DELTA).mean() * 100
    point_risk_false_alarms_frac = (point_risk_false_alarms > cfg.EXP.DELTA).mean() * 100
    eprocess_false_alarms_frac = (eprocess_false_alarms > cfg.EXP.DELTA).mean() * 100
    naive_eprocess_false_alarms_frac = (naive_eprocess_false_alarms > cfg.EXP.DELTA).mean() * 100
    pmeb_eprocess_false_alarms_frac = (pmeb_eprocess_false_alarms > cfg.EXP.DELTA).mean() * 100
    
    axes[1].hist(point_risk_false_alarms, bins=10, alpha=0.5, edgecolor='black', density=False, label=fr"Point Risk ({point_risk_false_alarms_frac:.2f}% > $\delta$)")
    axes[1].hist(running_risk_false_alarms, bins=10, alpha=0.5, edgecolor='black', density=False, label=fr"Running Risk ({running_risk_false_alarms_frac:.2f}% > $\delta$)")
    axes[1].hist(eprocess_false_alarms, bins=10, alpha=0.5, edgecolor='black', density=False, label=fr"E-process ({eprocess_false_alarms_frac:.2f}% > $\delta$)")
    axes[1].hist(naive_eprocess_false_alarms, bins=10, alpha=0.5, edgecolor='black', density=False, label=fr"Naive E-process ({naive_eprocess_false_alarms_frac:.2f}% > $\delta$)")
    axes[1].hist(pmeb_eprocess_false_alarms, bins=10, alpha=0.5, edgecolor='black', density=False, label=fr"PMEB E-process ({pmeb_eprocess_false_alarms_frac:.2f}% > $\delta$)")
    
    axes[1].axvline(cfg.EXP.DELTA, color='red', linestyle='--', label=r"False Alarm Rate $\delta$")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlabel("False Alarms")
    axes[1].legend()
    axes[1].set_title(f"False Alarm Fraction across Thresholds ({cfg.EXP.NR_TRIALS} Trials)")
    axes[1].set_xlim(-0.1, 1.1)
    
    save_fig(logger, cfg.RUN.PLOT_DIR, "false_alarms_over_psi")


def plot_auto(
    cfg,
    logger,
    psi_cand,
    stream_losses,
    point_risk,
    running_risk,
    eprocess,
    naive_eprocess,
    pmeb_eprocess,
):
    plot_risk(cfg, logger, psi_cand, point_risk, running_risk, eprocess, naive_eprocess, pmeb_eprocess, plot_psi=5)
    plot_cum_losses(cfg, logger, psi_cand, stream_losses)
    plot_stop_times(cfg, logger, psi_cand, point_risk, running_risk, eprocess, naive_eprocess, pmeb_eprocess)
    plot_cs(cfg, logger, psi_cand, point_risk, running_risk, eprocess, naive_eprocess, pmeb_eprocess)
    plot_false_alarms(cfg, logger, psi_cand, point_risk, running_risk, eprocess, naive_eprocess, pmeb_eprocess)
    logger.info("All auto plots saved.")
