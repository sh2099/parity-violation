<div align="center">

# Parity Violation

[![python](https://img.shields.io/badge/-Python_3.13-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
  
</div>

This repository contains the foundations of the code used for our paper: [An Unsupervised search for parity violation in the Large Scale Structure.](https://arxiv.org/abs/2410.16030)

## Installation

#### Clone

```bash
# clone project
git clone git@github.com:sh2099/parity-violation.git
cd parity-violation
```

#### Install using Micromamba (/Conda/Mamba)

For conda or mamba replace `micromamba` with `conda` or `mamba` below.

```bash
micromamba env create -f environment.yaml  # create mamba environment
micromamba activate parity_env                  # activate environment
pip install -e .                           # install as an editable package
```

#### Install using Pip

Not sorted yet


#### Accessing BOSS data

The LSS data from the 12th BOSS data release can be found at https://data.sdss.org/sas/dr12/boss/lss/

For now, I recommend saving any fits files into the data folder.

Environment variables may be introduced later in construction.


</div>

## Jupyter Notebook Demonstration

The Notebook ml_pv/demo.ipynb has been created to walk through the process step by step, so is a good place to start.\
The hydra setup can then be used for more efficient modular handling.


</div>

## Generating Images

```bash
python -m datagen.image_gen.run_img_gen 
```
To change the number of images, I recommend first trying with the hydra call
```bash
python -m datagen.image_gen.run_img_gen images.num_test_samples=24 images.num_train_samples=96
```
Once initial testing is done, the default values can be changed in the `configs.datagen.image_gen.yaml`.

</div>

## Training the model

```bash
python -m ml.train_nn
```
