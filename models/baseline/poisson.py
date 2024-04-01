# This code is borrowed from nlatents basic_example.ipynb
# Link: https://github.com/neurallatents/nlb_tools/blob/main/examples/tutorials/basic_example.ipynb

import argparse

import h5py
import numpy as np
import scipy.signal as signal
from nlb_tools.evaluation import evaluate
from sklearn.linear_model import PoissonRegressor


def parse_args():
    parser = argparse.ArgumentParser("Fit a Poisson GLM.")
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to training data."
    )
    parser.add_argument(
        "--eval_data_path", type=str, required=True, help="Path to evaluation data."
    )
    parser.add_argument(
        "--target_data_path", type=str, required=True, help="Path to target data."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.0, help="Regularization strength."
    )
    parser.add_argument("--bin_width", type=int, default=5, help="Bin width of data.")
    return parser.parse_args()


def fit_poisson(train_input, eval_input, train_output, alpha=0.0):
    train_pred = []
    eval_pred = []
    # train Poisson GLM for each output column
    for chan in range(train_output.shape[1]):
        pr = PoissonRegressor(alpha=alpha, max_iter=500)
        pr.fit(train_input, train_output[:, chan])
        train_pred.append(pr.predict(train_input))
        eval_pred.append(pr.predict(eval_input))
    train_pred = np.vstack(train_pred).T
    eval_pred = np.vstack(eval_pred).T
    return train_pred, eval_pred


if __name__ == "__main__":
    args = parse_args()

    # Assign the arguments to variables.
    # (This is not necessary but makes the code more readable.)
    train_data_path = args.train_data_path
    eval_data_path = args.eval_data_path
    target_data_path = args.target_data_path
    alpha = args.alpha
    bin_width = args.bin_width

    # Load the data
    with h5py.File(train_data_path, "r") as f:
        train_behavior = f["train_behavior"][()]
        train_spikes_heldin = f["train_spikes_heldin"][()]
        train_spikes_heldout = f["train_spikes_heldout"][()]
    with h5py.File(eval_data_path, "r") as f:
        eval_spikes_heldin = f["eval_spikes_heldin"][()]
        eval_spikes_heldout = f["eval_spikes_heldout"][()]
    with h5py.File(target_data_path, "r") as f:
        target_dict = {"mc_maze": {}}
        for key in f["mc_maze"]:
            target_dict['mc_maze'][key] = f["mc_maze"][key][()]

    ## Smooth spikes

    # Assign useful variables
    tlength = train_spikes_heldin.shape[1]
    num_train = train_spikes_heldin.shape[0]
    num_eval = eval_spikes_heldin.shape[0]
    num_heldin = train_spikes_heldin.shape[2]
    num_heldout = train_spikes_heldout.shape[2]

    # Smooth spikes with 40 ms std gaussian
    kern_sd_ms = 40
    kern_sd = int(round(kern_sd_ms / bin_width))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, "same")

    train_spksmth_heldin = np.apply_along_axis(filt, 1, train_spikes_heldin)
    eval_spksmth_heldin = np.apply_along_axis(filt, 1, eval_spikes_heldin)

    ## Generate rate predictions
    # Reshape data to 2d for regression
    train_spksmth_heldin_s = train_spksmth_heldin.reshape(
        -1, train_spksmth_heldin.shape[2]
    )
    eval_spksmth_heldin_s = eval_spksmth_heldin.reshape(
        -1, eval_spksmth_heldin.shape[2]
    )
    train_spikes_heldout_s = train_spikes_heldout.reshape(
        -1, train_spikes_heldout.shape[2]
    )

    # Train Poisson regressor from log of held-in smoothed spikes to held-out spikes
    train_spksmth_heldout_s, eval_spksmth_heldout_s = fit_poisson(
        np.log(
            train_spksmth_heldin_s + 1e-4
        ),  # add constant offset to prevent taking log of 0
        np.log(eval_spksmth_heldin_s + 1e-4),
        train_spikes_heldout_s,
        alpha=0.1,
    )

    # Reshape data back to the same 3d shape as the input arrays
    train_rates_heldin = train_spksmth_heldin_s.reshape(
        (num_train, tlength, num_heldin)
    )
    train_rates_heldout = train_spksmth_heldout_s.reshape(
        (num_train, tlength, num_heldout)
    )
    eval_rates_heldin = eval_spksmth_heldin_s.reshape((num_eval, tlength, num_heldin))
    eval_rates_heldout = eval_spksmth_heldout_s.reshape(
        (num_eval, tlength, num_heldout)
    )

    ## Prepare submission data
    output_dict = {
        "mc_maze": {
            "train_rates_heldin": train_rates_heldin,
            "train_rates_heldout": train_rates_heldout,
            "eval_rates_heldin": eval_rates_heldin,
            "eval_rates_heldout": eval_rates_heldout,
        }
    }

    print(evaluate(target_dict, output_dict))