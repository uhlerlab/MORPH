"""
Create Identity Matrix Gene Embeddings for Ablation Study

Each gene is represented by a one-hot vector where only the corresponding 
dimension is set to 1, and all other dimensions are 0.

Usage:
    python create_identity_matrix_embedding.py \
        --adata_path <path_to_adata.h5ad> \
        --output_path <output_path.pkl> \
        [--random_seed <seed>]
"""

import argparse
import numpy as np
import scanpy as sc
import pickle
from pathlib import Path


def create_identity_matrix_embedding(adata_path, output_path, random_seed=None):
    """
    Create identity matrix (one-hot) gene embeddings from an AnnData object.
    
    Parameters:
    -----------
    adata_path : str
        Path to the AnnData (.h5ad) file containing gene perturbation data
    output_path : str
        Path where the gene embedding dictionary will be saved (.pkl format)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    gene_matrix_dict : dict
        Dictionary mapping gene names to their one-hot embedding vectors
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Load AnnData object
    print(f"Loading AnnData from: {adata_path}")
    adata = sc.read(adata_path)
    gene_names = adata.obs['gene'].unique()
    n_genes = len(gene_names)
    print(f"Number of unique genes: {n_genes}")
    
    # Create identity matrix: shape (n_genes, n_genes)
    gene_matrix = np.eye(n_genes)
    print(f"Created identity matrix of shape: {gene_matrix.shape}")
    gene_matrix_dict = {gene_names[i]: gene_matrix[i, :] for i in range(len(gene_names))}
    print(f"Created dictionary with {len(gene_matrix_dict)} gene embeddings")
    
    # Save the dictionary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving gene embedding dictionary to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(gene_matrix_dict, f)
    
    print("Done!")
    return gene_matrix_dict


def main():
    parser = argparse.ArgumentParser(
        description="Create identity matrix (one-hot) gene embeddings for ablation study"
    )
    parser.add_argument(
        '--adata_path',
        type=str,
        required=True,
        help='Path to the AnnData (.h5ad) file containing gene perturbation data'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the gene embedding dictionary (.pkl format)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional, not used for identity matrix)'
    )
    
    args = parser.parse_args()
    
    create_identity_matrix_embedding(
        adata_path=args.adata_path,
        output_path=args.output_path,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
