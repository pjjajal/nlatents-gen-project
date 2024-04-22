# Neural Latents - Self-Supervised Neural Data Transformers
This repository contains the code my ECE 695 - Generative Models final project.
This work will train [Neural Data Transformers](https://arxiv.org/abs/2108.01210) using self-supervised learning. 

The model will be trained and tested on the data provided from [Neural Latents](https://neurallatents.github.io/).

## Setup Instructions
This repository uses `anaconda` to create its python environment. 
I recommend you install a [miniforge](https://conda-forge.org/miniforge/) distribution.

Once you have `anaconda` installed, setup the environment using the following:
```sh
conda create -n nlatents python=3.12
conda activate nlatents
pip install -r requirements.txt
conda install ipykernel # use this if you will be looking at the notebooks.
```

(This should work, hopefully.)

## Repository Outline
- `data`: this folder is where all downloaded data should be save. 
- `dataset`: this folder contains the scripts for creating (processed) datasets used for training models, or, more generally, code related to the "data pipeline" (whatever that means).
- `models`: this folder contains model implementations. 
    - `baselines` contains all baseline models from Neural Latents.
    - `ndt_mae` contains my transformer implementations. This is a mix of publicly available code and code I have written (either for this work or for other works). Publicly available code has the copyright and source at the top of the file (I apologize if I've missed anything).

## Downloading Datasets
Datasets should be downloaded to the `data` folder. The following snippet outlines how to do this:
```sh 
cd data
dandi download DANDI:000128/0.220113.0400 # this downloads the MC_MAZE dataset, change the argument to download another one.

# For example: 
# MC_RTT: dandi download DANDI:000128/0.220113.0400
# Area2_Bump: dandi download DANDI:000127/0.220113.0359
# ...and so on.
```

### Processing Data
The data is processed using the `create_datasets.py` script in the `datasets` folder. 
To generate the data for the `MC_MAZE` dataset use the following command (from the repository root directory):
```sh
# datasets flag controls which dataset to process
# bin_width flag controls how to resample the data. (5 = 5ms bin width)
python -m datasets.create_datasets --datasets mcmaze --bin_width=5 
```


## Training Jobs

## Results
### MC_Maze Rate Predictions
| Method                                          | co-bps (higher better) | vel R2 (higher better) |
| ----------------------------------------------- | ---------------------- | ---------------------- |
| Smoothing (baseline)                            | 0.2122                 | 0.616                  |
| NDT (baseline)                                  | 0.3597                 | 0.8897                 |
| **My Method Spikes --> Spikes (supervised)**    | 0.3179                 | -                      |
| **My Method Behaviour --> Spikes (supervised)** | 0.3164                 | -                      |

*Note 1: My architecture is similar to NDT. They use hyperparameter search, this helps get better performance. I did not have time to do this but there is historical precedent that this may help.*

*Note 2: NDT values taken from EvalAI leaderboard.*
