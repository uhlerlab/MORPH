# Ablation Study Analysis

This directory contains scripts and instructions for running ablation studies to evaluate different components in MORPH.

## Overview

Three types of ablation studies:

1. **Priors and Control-Cell Inputs**: Tests gene embeddings and control cell information
2. **Model Size**: Tests impact of varying latent dimensions (`latdim_ctrl`, `latdim_ptb`)
3. **Attention Mechanism**: Tests impact of attention by running without it

## Type 1: Priors and Control-Cell Inputs

Four ablation studies:

- **Identity Matrix**: Replace DepMap embeddings with one-hot vectors
- **Random Matrix**: Replace DepMap embeddings with random Gaussian vectors
- **Permuted DepMap**: Shuffle DepMap embeddings while keeping gene names fixed
- **Prior-Only**: Mask all control cells (set expression to zeros)

### Scripts
- [`create_identity_matrix_embedding.py`](./create_identity_matrix_embedding.py): Creates one-hot gene embeddings
- [`create_random_matrix_embedding.py`](./create_random_matrix_embedding.py): Creates random Gaussian gene embeddings
- [`create_permuted_depmap_embedding.py`](./create_permuted_depmap_embedding.py): Creates permuted DepMap embeddings
- [`mask_control_cells.py`](./mask_control_cells.py): Masks control cells in AnnData objects

### Usage

**For embedding ablations** (Identity Matrix, Random Matrix, Permuted DepMap):

1. Run the script to create embeddings:
```bash
# Identity Matrix
python create_identity_matrix_embedding.py \
    --adata_path <path_to_adata.h5ad> \
    --output_path <output_path>/identity_matrix_embedding.pkl

# Random Matrix
python create_random_matrix_embedding.py \
    --adata_path <path_to_adata.h5ad> \
    --output_path <output_path>/random_matrix_embedding.pkl \
    --embedding_dim 512

# Permuted DepMap
python create_permuted_depmap_embedding.py \
    --depmap_path <path_to_depmap.pkl> \
    --output_path <output_path>/permuted_depmap_embedding.pkl
```

2. Add embedding path to `MORPH/data/perturb_embed_file_path.csv` with a unique identifier
3. Set `representation_types=("YourIdentifier")` in `MORPH/morph/run.sh`

**For control cell masking** (Prior-Only):

1. Run the masking script:
```bash
python mask_control_cells.py \
    --adata_path <path_to_adata.h5ad> \
    --output_path <output_path>/scadata_ctrl_masked.h5ad \
    [--control_label non-targeting]
```

2. Add dataset path to `MORPH/data/scdata_file_path.csv` with a unique identifier
3. Set `dataset_name="your_dataset_ctrl_masked"` in `MORPH/morph/run.sh`

## Type 2: Model Size Ablation

Modify `MORPH/morph/run.sh`:
- Set `latdim_ctrl=<dimension>` (e.g., 100)
- Set `latdim_ptb=<dimension>` (e.g., 100)

Then run: `bash MORPH/morph/run.sh`

## Type 3: Attention Mechanism Ablation

Modify `MORPH/morph/run.sh`:
- Set `model="MORPH_no_attention"`

Then run: `bash MORPH/morph/run.sh`

## Note
- Ensure embedding dimensions match model expectations
