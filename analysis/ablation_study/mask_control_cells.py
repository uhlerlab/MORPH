"""
Mask Control Cells for Prior-Only Ablation Study

This script masks all control (non-targeting) cells in the AnnData object
by setting their expression values to zeros.

Usage:
    python mask_control_cells.py \
        --adata_path <path_to_adata.h5ad> \
        --output_path <output_path.h5ad> \
        --control_label <label>
"""

import argparse
import numpy as np
import scanpy as sc
from pathlib import Path


def mask_control_cells(adata_path, output_path, control_label='non-targeting'):
    """
    Mask control cells by setting their expression values to zeros.
    
    Parameters:
    -----------
    adata_path : str
        Path to the input AnnData (.h5ad) file
    output_path : str
        Path where the masked AnnData will be saved (.h5ad format)
    control_label : str, optional
        Value used to identify control cells in adata.obs['gene'] (default: 'non-targeting')
    
    Returns:
    --------
    adata_masked : AnnData
        AnnData object with control cells masked (expression set to zeros)
    """
    # Load AnnData object
    print(f"Loading AnnData from: {adata_path}")
    adata = sc.read(adata_path)
    print(f"Original AnnData shape: {adata.shape}")
    print(f"Number of cells: {adata.n_obs}")
    print(f"Number of genes: {adata.n_vars}")

    # Check that 'gene' column exists in adata.obs
    if "gene" not in adata.obs.columns:
        raise ValueError(
            "'gene' column not found in adata.obs.columns. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    
    # Identify control cells
    control_mask = adata.obs['gene'] == control_label
    n_control_cells = control_mask.sum()
    print(f"Number of control cells with gene == '{control_label}': {n_control_cells}")
    
    if n_control_cells == 0:
        print(f"Warning: No control cells found where adata.obs['gene'] == '{control_label}'!")
        adata_masked = adata.copy()
    else:
        adata_masked = adata.copy()
        adata_masked.X[control_mask, :] = 0
        print(f"Masked {n_control_cells} control cells (set expression to zeros)")
    
    # Save the masked AnnData
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving masked AnnData to: {output_path}")
    adata_masked.write(output_path)
    
    print("Done!")
    return adata_masked


def main():
    parser = argparse.ArgumentParser(
        description="Mask control cells in AnnData for prior-only ablation study"
    )
    parser.add_argument(
        '--adata_path',
        type=str,
        required=True,
        help='Path to the input AnnData (.h5ad) file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the masked AnnData (.h5ad format)'
    )
    parser.add_argument(
        '--control_label',
        type=str,
        default='non-targeting',
        help="Value in adata.obs['gene'] that identifies control cells (default: 'non-targeting')"
    )
    
    args = parser.parse_args()
    
    mask_control_cells(
        adata_path=args.adata_path,
        output_path=args.output_path,
        control_label=args.control_label
    )


if __name__ == "__main__":
    main()
