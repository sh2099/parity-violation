<div align="center">

# Parity Violation

[![python](https://img.shields.io/badge/-Python_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
  
</div>

This repository contains the foundations of the code used for my Master's project: searching for parity violations in the large scale structure of the Universe, using Unsupervised Learning.

## Installation

#### Clone

```bash
# clone project
git clone git@github.com:sh2099/parity-violation.git
cd parity-violation
```

#### Install using Conda/Mamba/Micromamba (recommended)

For conda or mamba replace `micromamba` with `conda` or `mamba` below.

```bash
micromamba env create -f environment.yaml  # create mamba environment
micromamba activate parity_env                  # activate environment
pip install -e .                           # install as an editable package
```

#### Install using Pip

Not sorted yet

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
