"""
Generate commands.txt for running experiments, where each line is a single python run command
Then commands.txt is used by run.sh to run experiments in series.
"""

# PARAM GRIDS
configs = [
    ('exp_cp', 'cfg_natural'),
    ('exp_ood', 'cfg_step'), 
    ('exp_ood', 'cfg_noshift'), 
    ('exp_ood', 'cfg_direct'),
    # ('exp_uci', 'cfg_uci')
]
tracker_window = {
    "exp_cp": [0, 365, 50, 10],
    "exp_ood": [0, 200, 50, 10],
    # "exp_uci": [0, 200, 50, 10]
}
batch_ts = [1, 10, 50]
sc = [5, 25, 0]
suffix = "_FULL"
device = "cuda"

# python erc/exp/exp_cp.py --cfg_file=cfg_natural --cfg_dir=erc/config/exp_cp --exp_suffix=_TEST --no-get_pred --save_file --batch_ts=1 --tracker_window 0 0 0 --stop_counter 5 25 0 --device=cuda
# python erc/exp/exp_ood.py --cfg_file=cfg_step --cfg_dir=erc/config/exp_ood --exp_suffix=_TEST --no-get_pred --save_file --batch_ts=1 --tracker_window 0 0 0 --stop_counter 5 25 0 --device=cuda


def main():
    file = "erc/commands.txt"
    print(f"Writing commands to {file}")

    with open(file, "w") as f:
        for (exp, cfg) in configs:
            for bts in batch_ts:
                for tw in tracker_window[exp]:
                    s = f"python erc/exp/{exp}.py --cfg_file={cfg} --cfg_dir=erc/config/{exp} --exp_suffix={suffix} --no-get_pred --save_file --batch_ts={bts} --tracker_window 0 {tw} {tw} --stop_counter {sc[0]} {sc[1]} {sc[2]} --device={device}"
                    f.write(s + "\n")
    print("Done!")


if __name__ == "__main__":
    main()
