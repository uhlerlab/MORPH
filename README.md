# MORPH (a modular framework for predicting responses to perturbational changes)

Code for paper: MORPH predicts the single-cell outcome of genetic perturbations across various data modalities

arXiv link: [to-add]

## System requirements
The code has been developed on a high-performance system running Ubuntu 20.04.4 LTS, equipped with an AMD EPYC 7513 32-Core Processor (128 threads total), 1 TB of RAM, and eight NVIDIA RTX A5000 GPUs using CUDA version 11.6. However, the demo application of our pipeline described below only requires a Linux-based system with at least 20 GB of free disk space, one GPU, and an internet connection, and therefore can be run on significantly less powerful hardware.

## Installation
Follow the two steps illustrated below

1. create a conda environment using `environment.yaml` (all dependencies are included; whole process takes about 5 min):
```
conda env create -f environment.yml
```
2. install the current package in editable mode inside the conda environment:
```
pip install -e .
```

## Experiments

After 
1. pre-processing the single-cell dataset (see example in `demo/data_tutorial.ipynb`),
2. specifying the train-test split (see example in `demo/split_tutorial.ipynb`), 
3. generating the gene embedding dictionaries that incorporate prior biological knowledge (see example in `demo/gene_embedding_tutorial.ipynb`),

you can specify the hyperparameters in `run.sh` and run MORPH and automatically evaluate the results upon completion:
```
./run.sh
```

Source code folder: `./morph/`

## Figures/Analyses in the paper

Illustraive figures: made using powerpoint

Pointers for nonillustrative figures:

- `./analysis/single_gene_perturbation`: Fig. 2, Supplemental Fig. 2-5
- `./analysis/cell_line_transfer`: Fig. 3, and codes to perform transfer learning
- `./analysis/double_gene`: Fig. 3, Supplemental Fig. 6
- `./analysis/active_learning`: Fig. 4, Supplemental Fig. 7, and codes to perform AL experiments
- `./analysis/grn`: Fig. 5, Supplemental Fig. 8
- `./analysis/ops`: Fig. 6, Supplemental Fig. 9

