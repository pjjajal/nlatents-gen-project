import argparse
import logging
import os

import pandas as pd
from nlb_tools.make_tensors import make_eval_input_tensors, make_train_input_tensors
from nlb_tools.nwb_interface import NWBDataset

MCMAZE_PATH = "data/000128/sub-Jenkins"

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser("Create datasets for training.")
    parser.add_argument("--datasets", nargs="+", choices=["mcmaze"])
    parser.add_argument("--bin_width", type=int, default=1)
    return parser.parse_args()


def create_mcmaze_data(bin_width=1):
    assert os.path.exists(
        MCMAZE_PATH
    ), "MC_MAZE data not found. Please download it first. See README for instructions."

    # Load the data.
    dataset = NWBDataset(MCMAZE_PATH)
    dataset.resample(target_bin=bin_width)

    # Save train data.
    train_save_path = f"datasets/mcmaze_train_bw{bin_width}.h5"
    train_dict = make_train_input_tensors(
        dataset=dataset,
        dataset_name="mc_maze",
        trial_split="train",
        save_path=train_save_path,
        include_behavior=True,
    )

    # Save val data.
    val_save_path = f"datasets/mcmaze_val_bw{bin_width}.h5"
    val_dict = make_train_input_tensors(
        dataset=dataset,
        dataset_name="mc_maze",
        trial_split="val",
        save_path=val_save_path,
        include_behavior=True,
    )


def main():
    args = parse_args()
    if "mcmaze" in args.datasets:
        create_mcmaze_data(args.bin_width)

if __name__ == "__main__":
    main()