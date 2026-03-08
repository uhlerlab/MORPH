"""
Create Random Matrix Gene Embeddings for Ablation Study

Each gene is represented by a random vector sampled from a standard normal distribution.

Usage:
    python create_random_matrix_embedding.py \
        --adata_path <path_to_adata.h5ad> \
        --output_path <output_path.pkl> \
        --embedding_dim <dim> \
        [--random_seed <seed>]
"""

import argparse
import numpy as np
import scanpy as sc
import pickle
from pathlib import Path


def create_random_matrix_embedding(adata_path, output_path, embedding_dim, random_seed=12):
    """
    Create random matrix gene embeddings from an AnnData object.
    
    Parameters:
    -----------
    adata_path : str
        Path to the AnnData (.h5ad) file containing gene perturbation data
    output_path : str
        Path where the gene embedding dictionary will be saved (.pkl format)
    embedding_dim : int
        Dimension of the random embedding vectors
    random_seed : int, optional
        Random seed
    
    Returns:
    --------
    gene_matrix_dict : dict
        Dictionary mapping gene names to their random embedding vectors
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Load AnnData object
    print(f"Loading AnnData from: {adata_path}")
    adata = sc.read(adata_path)
    gene_names = adata.obs['gene'].unique()
    n_genes = len(gene_names)
    print(f"Number of unique genes: {n_genes}")
    
    # Create random matrix: shape (n_genes, embedding_dim)
    gene_matrix = np.random.randn(n_genes, embedding_dim)
    print(f"Created random matrix of shape: {gene_matrix.shape}")
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
        description="Create random matrix gene embeddings for ablation study"
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
        '--embedding_dim',
        type=int,
        required=True,
        help='Dimension of the random embedding vectors (e.g., 512)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=12,
        help='Random seed for reproducibility (default: 12)'
    )
    
    args = parser.parse_args()
    
    create_random_matrix_embedding(
        adata_path=args.adata_path,
        output_path=args.output_path,
        embedding_dim=args.embedding_dim,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
