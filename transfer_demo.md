# Demo application for transfer learning

## Overview

This version of the demo shows how to apply **MORPH** to user-provided single-cell data **using a pre-trained model** we provide. Unlike the [user input demo](user_demo.md), which trains MORPH from scratch, this variant demonstrates how to **fine-tune** a model that has already been trained on large-scale perturbation data.

Leveraging a pre-trained model can significantly improve performance—especially in real-world settings—by transferring learned representations from related datasets. We recommend this approach for most practical applications, as it makes the most of the available prior knowledge.

> [!NOTE]
> We recommend running this version **after** you have tried the demo with the test input data, as described [here](test_demo.md).

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

### 3. Process and configure the input single-cell data
1. Preprocess your single-cell dataset by following the example in `demo/data_tutorial_transfer_learning.ipynb`.  
   *(Note: this is a different notebook than the one used in the standard demo.)*
2. Open the file located at `MORPH/data/scdata_file_path.csv` and add a new row to specify the path to your processed data:
    - In the `dataset` column, enter: `[your_data_id]` (a unique identifier for your dataset).
    - In the `file_path` column of the same row, enter the full path to your processed `.h5ad` file.

### 4. Create the train-test splits
1. Create a train-test split for your data following the example in `demo/split_tutorial.ipynb` 
2. Name this file as `[your_data_id]_splits.csv` (**Note: the data id must match what you have above**)
3. Save this file to the following directory:
```
MORPH/data
```

### 5. Download and configure the prior knowledge
Download the file named `depmap_crispr_gene_effect_processed.pkl` from [here](https://drive.google.com/drive/folders/1TQJE281q4xH7HcNHMg1v0urD99EDj5bO?usp=drive_link).

Then, open the file located at:
```
MORPH/data/perturb_embed_file_path.csv
```
and find the row where `representation_type` is `DepMap_GeneEffect`, and update the `file_path` column to the location of the downloaded DepMap file on your machine.

> [!NOTE]
> Because the pre-trained model was trained using **DepMap** as the source of prior knowledge, we recommend using **DepMap** embeddings when performing transfer learning. If you prefer to use a different type of prior knowledge (e.g., foundation model embeddings or pathway-based features), you should **retrain the model from scratch** using your chosen prior, and then follow the same steps outlined here.
> Note: In some contexts, alternative priors may outperform DepMap, depending on the biological setting and data domain.

### 6. Download the pre-trained model
1. Go to this link: [Pretrained Model Download](https://drive.google.com/drive/folders/1TQJE281q4xH7HcNHMg1v0urD99EDj5bO?usp=drive_link)
2. Download both of the following files:
   - `model.pt`
   - `config.json`
3. Move the downloaded files into the following folder: `MORPH/transfer_learning`.

### 7. Fine-tune the model on your dataset
Navigate to the `MORPH/morph/analysis/cell_line_transfer` folder. If you've followed the previous steps, you can do this by typing the following in your terminal:
```
cd morph/analysis/cell_line_transfer
```

Next, open the `cell_line_transfer_few_shot_learning.py` file and update the following fields:
- Set `test_dataset` to `[your_data_id]` — this should match the ID you used in `scdata_file_path.csv`.
- Set `leave_out_test_set_id` to the appropriate split ID — this should correspond to the ID specified in your train-test split file under column `test_set_id` (created in step 4).

You can also customize or optimize hyperparameters in this file.

To run MORPH using the default settings and automatically evaluate the results upon completion, execute:
```
python -u cell_line_transfer_few_shot_learning.py
```