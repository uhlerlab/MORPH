# MORPH (a modular framework for predicting responses to perturbational changes)

Code for paper: MORPH predicts the single-cell outcome of genetic perturbations across various data modalities

arXiv link: [to-add]

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

After pre-processing the single-cell dataset (see example in `demo/data_tutorial.ipynb`) and the gene embedding dictionaries that incorporate prior biological knowledge (see example in `demo/gene_embedding_tutorial.ipynb`), you can specify the path and hyperparameters in `run.sh` and run MORPH and automatically evaluate the results upon completion:
```
./run.sh
```

Source code folder: `./morph/`

## Figures in the paper

Illustraive figures: made using powerpoint

Pointers for nonillustrative figures:

- `./optint/notebook/test_ow.ipynb`: Fig. 3, Supplementary Fig. 2

