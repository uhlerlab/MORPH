"""
Create Permuted DepMap Gene Embeddings for Ablation Study

This script creates permuted DepMap embeddings by shuffling the values while
keeping the keys (gene names) fixed.

Usage:
    python create_permuted_depmap_embedding.py \
        --depmap_path <path_to_depmap.pkl> \
        --output_path <output_path.pkl> \
        [--random_seed <seed>]
"""

import argparse
import numpy as np
import pickle
from pathlib import Path


def create_permuted_depmap_embedding(depmap_path, output_path, random_seed=12):
    """
    Create permuted DepMap gene embeddings by shuffling values.
    
    Parameters:
    -----------
    depmap_path : str
        Path to the original DepMap embedding dictionary (.pkl format)
    output_path : str
        Path where the permuted gene embedding dictionary will be saved (.pkl format)
    random_seed : int, optional
        Random seed
    
    Returns:
    --------
    permuted_dict : dict
        Dictionary with same keys as input but shuffled values
    """
    # Set random seed
    np.random.seed(random_seed)
    
    # Load original DepMap dictionary
    print(f"Loading DepMap dictionary from: {depmap_path}")
    with open(depmap_path, 'rb') as f:
        depmap_dict = pickle.load(f)
    print(f"Original dictionary contains {len(depmap_dict)} genes")
    
    # Extract gene names and their corresponding embeddings
    gene_names = list(depmap_dict.keys())
    values = list(depmap_dict.values())
    shuffled_indices = np.random.permutation(len(gene_names))
    permuted_dict = {gene_names[i]: values[shuffled_indices[i]] for i in range(len(gene_names))}
    print(f"Created permuted dictionary with {len(permuted_dict)} gene embeddings")
    
    # Save the permuted dictionary
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving permuted gene embedding dictionary to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(permuted_dict, f)
    
    print("Done!")
    return permuted_dict


def main():
    parser = argparse.ArgumentParser(
        description="Create permuted DepMap gene embeddings for ablation study"
    )
    parser.add_argument(
        '--depmap_path',
        type=str,
        required=True,
        help='Path to the original DepMap embedding dictionary (.pkl format)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Output path for the permuted gene embedding dictionary (.pkl format)'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=12,
        help='Random seed for reproducibility (default: 12)'
    )
    
    args = parser.parse_args()
    
    create_permuted_depmap_embedding(
        depmap_path=args.depmap_path,
        output_path=args.output_path,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
