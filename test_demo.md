# Demo application using test inputs

## Overview

This version of our demo is designed to help you quickly try out our pipeline using publicly available data and prior knowledge from DepMap.

#### What’s Included
- A **public dataset** from [Norman et al. (2019)](https://www.science.org/doi/10.1126/science.aax4438), which includes both single-gene and double-gene perturbations.
- A **prior knowledge matrix** from [DepMap](https://depmap.org/portal), which provides information about gene relationships.

We’ve preprocessed both the Norman dataset and the DepMap prior. You can download them from [this link](https://drive.google.com/drive/folders/1TQJE281q4xH7HcNHMg1v0urD99EDj5bO?usp=drive_link).

This version of the demo is intended to quickly test the functionality and familiarize the user with the outputs of the application. We recommend running this version first, before following the instructions of the [guide here](user_demo.md) that show how you can use MORPH to perform inference on your single-cell data.

---

## Step-by-step guide

> [!IMPORTANT]
> Before you proceed make sure that you have followed the instructions in the [README file](README.md) of this repository.
> To run the code please ensure that your working directory is ``MORPH``. The working directory can be changed via the ``cd`` command.

### 1. Activate the bash shell and conda environment
Please open a new terminal and activate the Bash shell by running:
```
bash
```

Then, activate the Conda environment you set up earlier by running:
```
conda activate morph
```

*Make sure to run the model within this conda environment, which contains all the necessary packages.*

### 2. Set MORPH as your working directory
Please navigate to the directory where you cloned the GitHub repository:
```
cd path/to/MORPH
```
Replace `path/to/MORPH` with the actual path on your machine.

### 3. Download and configure the input single-cell data
1. Download the processed single-cell dataset from Norman et al. (2019) [here](https://drive.google.com/drive/folders/1TQJE281q4xH7HcNHMg1v0urD99EDj5bO?usp=drive_link). The file is named `Norman2019_normalized_hvg.h5ad`. On linux, you can use [gdown](https://github.com/wkentaro/gdown) to download the file by running `pip install gdown` followed by `gdown --folder https://drive.google.com/drive/folders/1TQJE281q4xH7HcNHMg1v0urD99EDj5bO`
3. Open the file located at `MORPH/data/scdata_file_path.csv` to specify the path to the downloaded data:
    - Find the row where `dataset` column is `norman_k562_hvg `(**do not change this name**, as it must match the entries in the train-test split file).
    - In the corresponding `file_path` column, update the value to be the full path to the downloaded `.h5ad` file on your machine.

### 4. Get the train-test splits
We provided a sample split file in the following directory:
```
MORPH/data/norman_k562_hvg_splits.csv
```
No action is needed.

### 5. Download and configure the prior knowledge
Download the file named `depmap_crispr_gene_effect_processed.pkl` from [here](https://drive.google.com/drive/folders/1TQJE281q4xH7HcNHMg1v0urD99EDj5bO?usp=drive_link).

Then, open the file located at:
```
MORPH/data/perturb_embed_file_path.csv
```
and find the row where `representation_type` is `DepMap_GeneEffect`, and update the `file_path` column to the location of the downloaded DepMap file on your machine.

### 6. Run the model
Navigate to the `MORPH/morph` folder. If you've followed the previous steps, you can do this by typing the following in your terminal:
```
cd morph
```
Once you're in the correct directory, you can run MORPH using the default settings and automatically evaluate the results after completion:
```
./run.sh
```


> [!NOTE]
> When you run the code, you might see a prompt from **W&B**, a tool for tracking and visualizing results.  
You **don’t need a W&B account** if you just want to run the model without visualizations.
> If you see this prompt:
> ```
> wandb: (1) Create a W&B account
> wandb: (2) Use an existing W&B account
> wandb: (3) Don't visualize my results
> wandb: Enter your choice:
> ```
> 👉 Simply type `3` and press Enter to skip logging:
> ```
> wandb: Enter your choice: 3
> ```
> W&B will be disabled for this run.

---

That's it! The demo will run end-to-end and output results using the provided public data and prior.

### (Optional / Advanced settings) Downstream Analysis
You may follow the instructions in the last section of [README file](README.md) to reproduce the downstream analysis.
