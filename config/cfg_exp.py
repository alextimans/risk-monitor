from yacs.config import CfgNode

########################
### BASE CONFIG FILE ###
########################

CFG = CfgNode()

CFG.PROJECT = CfgNode()
CFG.PROJECT.CODE_DIR = "erc"
CFG.PROJECT.CONFIG_FILE = "cfg"
CFG.PROJECT.CONFIG_DIR = "erc/config"
CFG.PROJECT.OUTPUT_DIR = "../../../../media/atimans/hdd/output_erc"
CFG.PROJECT.SEED = 666666

CFG.DATASET = CfgNode()
CFG.DATASET.DIR = "../../../../media/atimans/hdd/datasets/erc"
CFG.DATASET.BATCH_PRED = 64

CFG.MODEL = CfgNode(new_allowed=True)
CFG.MODEL.DEVICE = "cpu"
CFG.MODEL.TYPE = "resnet50"

CFG.RUN = CfgNode(new_allowed=True)
CFG.RUN.SUB_DIR = "auto"  # subfolder in output_dir/exp_name
CFG.RUN.EXP_DIR = "auto"  # subfolder in output_dir/exp_name/sub_dir
CFG.RUN.SUFFIX = ""  # suffix for output files in subfolder
CFG.RUN.FULL_DIR = ""  # full path to output_dir/exp_name/sub_dir/exp_dir_suffix
CFG.RUN.LOAD_DIR = "auto"  # path to load files from
CFG.RUN.PLOT_DIR = "plots"  # path to save plots
CFG.RUN.GET_PRED = False
CFG.RUN.PLOT = True

### EXPERIMENTS

CFG.EXP = CfgNode(new_allowed=True)
CFG.EXP.EPS = 0.1
CFG.EXP.DELTA = 0.1
CFG.EXP.NR_TRIALS = 10

CFG.EXP.NR_TIMESTEPS = 1000
CFG.EXP.BATCH_TIMESTEP = 1
CFG.EXP.PSI_START = 0.0
CFG.EXP.PSI_END = 1.0
CFG.EXP.PSI_INIT = 1.0
CFG.EXP.PSI_STEP = 0.01

CFG.EXP.RISK = "fpr_fnr"
CFG.EXP.BET_TYPE = "approx_grapa"
CFG.EXP.NR_POINT_RISK_SAMP = 100
CFG.EXP.NR_BURNIN = 100
CFG.EXP.TRACKER_WINDOW = [0, 0, 0]  # [point_risk, running_risk, eprocess]
CFG.EXP.STOP_COUNTER = [0, 0, 0]  # [point_risk, running_risk, eprocess]

### OOD EXP SPECIFIC
CFG.EXP.DATA_ID = "cifar10"
CFG.EXP.DATA_OOD = "svhn"
CFG.EXP.NR_OOD_TIMESTEPS = 200
CFG.EXP.OUT_SCORE = "entropy"
CFG.EXP.OOD_START = 0.0
CFG.EXP.OOD_END = 1.0
CFG.EXP.OOD_STEP = 0.05

### CP EXP SPECIFIC

# CFG.FILE = CfgNode(new_allowed=True)
# CFG.FILE.SAVE_LABELS = False
# CFG.FILE.SAVE_PREDS = False
# CFG.FILE.SAVE_CONF = False

# CFG.EXPERIMENT = CfgNode(new_allowed=True)
# CFG.EXPERIMENT.LOAD_DIR = "auto" # path to load files from
# CFG.EXPERIMENT.DIR = "auto"  # subfolder in output_dir
# CFG.EXPERIMENT.SUFFIX = ""  # suffix for output files in subfolder
# CFG.EXPERIMENT.FULL_DIR = ""  # full path to output_dir/dataset/experiment_dir_suffix
# CFG.EXPERIMENT.RUN_PRED = False
# CFG.EXPERIMENT.PLOT = True


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
        "eps": "EXP.EPS",
        "delta": "EXP.DELTA",
        "risk": "EXP.RISK",
        "out_score": "EXP.OUT_SCORE",
        # "tracker": "EXP.TRACKER",
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
