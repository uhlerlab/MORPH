# Demo application using user-provided inputs

## Overview

This version of our demo application applies MORPH to input single-cell data provided by the user. Specifically, MORPH generates predictions of genetic perturbation effects on the input data.

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
1. Preprocess your single-cell dataset by following the example provided in `demo/data_tutorial.ipynb`
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

### 5. Prepare and configure the prior knowledge
1.  Generating the gene embedding dictionaries that incorporate prior biological knowledge (see example in `demo/gene_embedding_tutorial.ipynb`)
2. Then, open the file located at:
```
MORPH/data/perturb_embed_file_path.csv
```
and find the row where `representation_type` is `DepMap_GeneEffect`, and update the `file_path` column to the location of the downloaded DepMap file on your machine.

### 6. Run the model
Navigate to the `MORPH/morph` folder. If you've followed the previous steps, you can do this by typing the following in your terminal:
```
cd morph
```

Once you're in the correct directory, change the hyperparameters in `run.sh` to account for your new data and train-test split. You can also optimize the hyperparameters for models here.

you can run MORPH using the default settings and automatically evaluate the results after completion:
```
./run.sh
```


you can specify the hyperparameters in `run.sh` and run MORPH and automatically evaluate the results upon completion:
```
./run.sh
```