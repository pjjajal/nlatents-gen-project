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
The data used 


## Training Jobs