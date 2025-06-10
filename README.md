# MORPH (a modular framework for predicting responses to perturbational changes)

Code for paper: MORPH predicts the single-cell outcome of genetic perturbations across various data modalities

biorxiv link: [to-add]

## System requirements
The code has been developed on a high-performance system running Ubuntu 20.04.4 LTS, equipped with an AMD EPYC 7513 32-Core Processor (128 threads total), 1 TB of RAM, and eight NVIDIA RTX A5000 GPUs using CUDA version 11.6. The demo application of our pipeline can be run on significantly less powerful hardware, requiring:
- Linux-based system
- at least 20 GB of free disk space
- *one GPU*
- an internet connection

## Step-by-step guide

### 1. Prerequisites

#### Bash shell
To follow the steps in this guide, you’ll need to use a bash shell. First, open a terminal:
- On Ubuntu, press Ctrl + Alt + T, or
- Search for “terminal” in your system’s applications.

Then, type this to start a bash shell (if it’s not already active):
```
bash
```

<details>
<summary><b>
Click here if you see the output: "command "bash" not found".
</b></summary>
 
 Please install ``bash`` as described in the output of your system e.g. via
 ```
 sudo apt-get update
 sudo apt-get install bash
 ```
</details>

#

#### Anaconda installation

You’ll need Anaconda or Miniconda to run the commands in this guide.
To check if it's installed, open a terminal and type:
```
conda
```

<details> <summary><b>
Don't see anything or get an error like "command not found"?
</b></summary>
That means Anaconda or Miniconda is not installed yet. 

To install Miniconda:
1. Open a new terminal.
2. Copy and paste the following commands **one by one**:
```
bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
```
This will install Miniconda using the default settings.

If you encounter any issues, please refer to the official installation guide which can be found [here](https://docs.conda.io/en/latest/miniconda.html#installing).

> [!WARNING]
> You need to close the terminal and open a **new** one to complete the installation

</details>

Make sure conda is initialized appropriately in your shell by typing 

```
bash
conda init bash
source ~/.bashrc
```

# 

<!-- #### Install pytorch
1. **Create a new conda environment**:
   ```
   conda create -n morph python=3.11
   conda activate morph
   ```
2. Check the cuda version by running `nvcc --version` or `ls /usr/local/ | grep cuda`. 
3. Go to the [pytorch webpage](https://pytorch.org/get-started/previous-versions/) and search for the matching CUDA version. Copy and run the suggested command.
4. To test the installation, run `python` and type:
    ```
    import torch
    print(torch.__version__)
    print(torch.cuda.is_available())  # Should return True if CUDA is properly configured
    ``` -->

### 2. Clone the repository

If the prerequisites are satisfied, please clone this repository by running the following command in a **new** terminal.
```
git clone https://github.com/chujunhe/MORPH.git
```

# 

### 3. Installation

Make sure you are in the root directory (i.e., `MORPH/`) by typing `cd MORPH`. 

Then follow the two steps illustrated below

1. Create a conda environment using `environment.yaml` (all dependencies are included; whole process takes about 5 min):
```
conda env create -f environment.yml -n morph
```
2. install the current package in editable mode inside the conda environment:
```
pip install -e .
```

#

### 4. Running the experiments

We provided three versions of the **MORPH** demo application: 
- [**MORPH for test inputs**](test_demo.md): Run the demo using default parameters and example input dataset and prior knowledge. This is the quickest way to verify that everything is working properly.
- [**MORPH for user-provided inputs**](user_demo.md): Apply the full pipeline to your own input dataset and prior knowledge. This version is intended for custom use cases.
- [**MORPH for transfer-learning**](transfer_demo.md): Adapt a model pre-trained on genome-wide Perturb-seq data to your own dataset (recommended usage, as it provides an optimal starting point).

**Please click on the name of the version you would like to run and follow the instructions.**

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

