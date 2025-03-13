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

After downloading the dataset, the CellOT model can be trained via the scripts/train.py script. For example, we can train CellOT on 4i data to predict perturbation effects of Cisplatin:

Run on a synthetic instance, e.g.:
```
python run.py --nnodes 5 --noise_level 1 --DAG_type path --std --a_size 2 --a_target 3 4 --acquisition greedy
```

Source code folder: `./optint/`

More examples given in: `./optint/notebook/test_multigraphs.ipynb`

## Figures in the paper

Illustraive figures: made using powerpoint

Pointers for nonillustrative figures:

- `./optint/notebook/test_ow.ipynb`: Fig. 3, Supplementary Fig. 2

