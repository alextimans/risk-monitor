from yacs.config import CfgNode

########################
### BASE CONFIG FILE ###
########################

CFG = CfgNode()

### ADMIN AND PATHS ###

CFG.PROJECT = CfgNode()
CFG.PROJECT.CODE_DIR = "erc"
# name of config file in config dir, without .py extension
CFG.PROJECT.CONFIG_FILE = "cfg"
CFG.PROJECT.CONFIG_DIR = "erc/config"
# this will be the main output dir for all experiments, with subfolders for each exp
CFG.PROJECT.OUTPUT_DIR = "your/output/directory/here" # "../../../../media/atimans/hdd/output_erc"
# used to set all seeds
CFG.PROJECT.SEED = 66666666

CFG.DATASET = CfgNode()
# this is the main dir for all datasets, with subfolders for each dataset
CFG.DATASET.DIR = "your/dataset/directory/here" # "../../../../media/atimans/hdd/datasets/erc"
# sample batch size to use for computing predictions (if RUN.GET_PRED = True)
CFG.DATASET.BATCH_PRED = 64

CFG.MODEL = CfgNode(new_allowed=True)
# cpu or cuda device to run experiments on
CFG.MODEL.DEVICE = "cpu"
CFG.MODEL.TYPE = "resnet50"
CFG.MODEL.DIR = "models"

CFG.RUN = CfgNode(new_allowed=True)
# subfolder in output_dir/exp_name, keep "auto" for automatic naming
CFG.RUN.SUB_DIR = "auto"
# subfolder in output_dir/exp_name/sub_dir, keep "auto" for automatic naming
CFG.RUN.EXP_DIR = "auto"
# suffix for output files in subfolder, e.g. for particular naming or trial runs
CFG.RUN.SUFFIX = ""
# full path to output_dir/exp_name/sub_dir/exp_dir_suffix if you want to specify manually; otherwise is auto-generated
CFG.RUN.FULL_DIR = "" 
# path to load files from, keep "auto" for automatic loading from output_dir/exp_name/sub_dir/exp_dir
CFG.RUN.LOAD_DIR = "auto"
# path to save plots, relative to output_dir/exp_name/sub_dir/exp_dir
CFG.RUN.PLOT_DIR = "plots"
# if True, compute and save predictions for datapoints before running monitoring; if False, load previously saved predictions from RUN.LOAD_DIR
CFG.RUN.GET_PRED = False
# if True, plot results at the end of the experiment; if False, skip plotting
CFG.RUN.PLOT = True
# if True, save all results to file; if False, only save final results to file
CFG.RUN.SAVE_FILE = True

### EXPERIMENTS ###

CFG.EXP = CfgNode(new_allowed=True)
# user-defined risk level
CFG.EXP.EPS = 0.1
# user-defined probability level
CFG.EXP.DELTA = 0.1
# number of trials to run the experiment
CFG.EXP.NR_TRIALS = 10
# max number of timesteps (stream length, monitoring steps)
CFG.EXP.NR_TIMESTEPS = 2000
# number of samples observed at each timestep
CFG.EXP.BATCH_TIMESTEP = 1
# decision threshold grid args
CFG.EXP.PSI_START = 0.0
CFG.EXP.PSI_END = 1.0
CFG.EXP.PSI_INIT = 1.0
CFG.EXP.PSI_STEP = 0.01

# risk to monitor
CFG.EXP.RISK = "fpr_fnr"
# betting strategy for e-process
CFG.EXP.BET_TYPE = "approx_grapa"
# number of samples to accurately estimate the true (unobserved) population risk at each step
CFG.EXP.NR_POINT_RISK_SAMP = 1000
# number of samples to use for burn-in period
CFG.EXP.NR_BURNIN = 100
# number of samples to use for sliding window (if zero, then the whole history is used)
CFG.EXP.TRACKER_WINDOW = [0, 0, 0]  # [point_risk, running_risk, eprocess]
# number of consecutive violations before stopping condition is invoked (for robust crossings)
# default '5' is used for less volatile point risk, '25' is used for more volatile running risk, '0' means e-process stops immediately when crossing 1/delta
CFG.EXP.STOP_COUNTER = [5, 25, 0]  # [point_risk, running_risk, eprocess]

### OOD EXP SPECIFIC ###

# inlier dataset
CFG.EXP.DATA_ID = "cifar10"
# OOD dataset
CFG.EXP.DATA_OOD = "svhn"
# trigger OOD probability increase every NR_OOD_TIMESTEPS steps
CFG.EXP.NR_OOD_TIMESTEPS = 200
# OOD score to threshold
CFG.EXP.OUT_SCORE = "entropy"
# OOD probability grid args
CFG.EXP.OOD_START = 0.0
CFG.EXP.OOD_END = 1.0
CFG.EXP.OOD_STEP = 0.05

### CP EXP SPECIFIC ###

# year to split between train data and deployment stream
CFG.EXP.SPLIT_TIME = 10
# confidence score to threshold
CFG.EXP.SET_SCORE = "probs"
# number of deployment timesteps to simulate for every year (daily observations)
CFG.EXP.NR_CP_TIMESTEPS = 365


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return CFG.clone()


def update_from_args(cfg, args):
    """Update config from CLI args.

    cfg: yacs.CfgNode object
    args: argparse.Namespace object
    """
    args_to_cfg = {
        "cfg_file": "PROJECT.CONFIG_FILE",
        "cfg_dir": "PROJECT.CONFIG_DIR",
        "load_dir": "RUN.LOAD_DIR",
        "exp_suffix": "RUN.SUFFIX",
        "get_pred": "RUN.GET_PRED",
        "save_file": "RUN.SAVE_FILE",
        "eps": "EXP.EPS",
        "delta": "EXP.DELTA",
        "risk": "EXP.RISK",
        "out_score": "EXP.OUT_SCORE",
        "set_score": "EXP.SET_SCORE",
        "bet_type": "EXP.BET_TYPE",
        "batch_ts": "EXP.BATCH_TIMESTEP",
        "tracker_window": "EXP.TRACKER_WINDOW",
        "stop_counter": "EXP.STOP_COUNTER",
        "device": "MODEL.DEVICE"
    }

    # Filter out entries with value None
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    # Bring into format for merge_from_list
    args_list = [
        item
        for sublist in [[args_to_cfg[k], v] for k, v in args_dict.items()]
        for item in sublist
    ]
    cfg.merge_from_list(args_list)
    return cfg, args_list
