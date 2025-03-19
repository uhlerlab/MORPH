import pandas as pd
import scanpy as sc
import pickle
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from warnings import filterwarnings
import sys

filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'morph'))
sys.path.append(morph_path)
from utils import SCDATA_sampler
from dataset import SCDataset
from torch.utils.data import DataLoader, Subset


def get_data(opts):
    # Read in the split data and get the leave-out test set
    split_path = f'{opts.path_dir}/data/{opts.dataset_name}_splits.csv'
    split_df = pd.read_csv(split_path)
    print('loading split data from: ', split_path)
    print('leave_out_test_set_id: ', opts.leave_out_test_set_id)
    ptb_leave_out_list = split_df[split_df['test_set_id'] == opts.leave_out_test_set_id]['test_set'].apply(lambda x: x.split(',')).values[0]

    # Read in the single-cell data
    scdata_file = pd.read_csv(f'{opts.path_dir}/data/scdata_file_path.csv')
    adata_path = scdata_file[scdata_file['dataset'] == opts.dataset_name]['file_path'].values[0]
    adata = sc.read_h5ad(adata_path)
    print('loading single-cell data from: ', adata_path)

    # Define training candidates and test set
    print('Test set length:')
    print(len(ptb_leave_out_list))
    opts.ptb_leave_out_list = ptb_leave_out_list

    all_ptb_targets = adata.obs['gene'].unique().tolist()
    all_ptb_targets.remove('non-targeting')
    training_candidates = list(set(all_ptb_targets) - set(ptb_leave_out_list))
    print('All training candidates')
    print(len(training_candidates))

    # read in gene_emb file
    embedding_file_df = pd.read_csv(f'{opts.path_dir}/data/perturb_embed_file_path.csv')
    embedding_file_subset = embedding_file_df[embedding_file_df['representation_type']==opts.representation_type]
    embed_file = embedding_file_subset['file_path'].values[0]
    print('loading gene embeddings from', embed_file)
    with open(embed_file, 'rb') as f:
        gene_emb_raw = pickle.load(f)

    # subset gene_emb to those in training_candidates
    gene_emb_dict = {k: v for k, v in gene_emb_raw.items() if k in training_candidates}
    print(len(gene_emb_dict))
    training_candidates_with_embeddings = list(gene_emb_dict.keys())
    print(len(training_candidates_with_embeddings))
    opts.training_candidates_with_embeddings = training_candidates_with_embeddings

    all_gene_emb_dict = {k: v for k, v in gene_emb_raw.items() if k in all_ptb_targets}
    print(len(all_gene_emb_dict))
    all_ptb_targets_with_embeddings = list(all_gene_emb_dict.keys())
    print(len(all_ptb_targets_with_embeddings))
    opts.all_ptb_targets_with_embeddings = all_ptb_targets_with_embeddings

    # Get the first training set -------------------------------------
    np.random.seed(opts.random_seed)
    np.random.shuffle(training_candidates_with_embeddings)
    train_list_1 = training_candidates_with_embeddings[:opts.initial_num_ptb]

    if opts.cluster_first == 'True':
        print('Apply K-means cluster in the latent space first')
        # load in raw gene embeddings
        gene_emb_array = np.stack(list(gene_emb_dict.values()))
        assert gene_emb_array.shape[0] == len(training_candidates_with_embeddings)
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=opts.initial_num_ptb, init='k-means++', random_state=opts.random_seed)
        kmeans.fit(gene_emb_array)
        # Find the closest sample to each center
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, gene_emb_array)
        # Ensure there are no duplicates by finding unique closest samples
        if len(set(closest)) < opts.initial_num_ptb:
            unique_closest = set()
            for center_idx in range(opts.initial_num_ptb):
                closest_idx = closest[center_idx]
                if closest_idx in unique_closest:
                    # Find the next closest sample that is not already selected
                    center_distances = pairwise_distances(kmeans.cluster_centers_[center_idx].reshape(1, -1), gene_emb_array).flatten()
                    sorted_indices = np.argsort(center_distances)
                    for idx in sorted_indices:
                        if idx not in unique_closest:
                            closest_idx = idx
                            break
                unique_closest.add(closest_idx)
        else:
            unique_closest = set(closest)
        # Get the closest samples
        train_list_1 = [list(gene_emb_dict.keys())[idx] for idx in unique_closest]
        train_list_1_path = opts.warmup_model_path + 'train_list_1_cluster_first.pkl'
        val_list_1_path = opts.warmup_model_path + 'val_list_1_cluster_first.pkl'
    else:
        train_list_1_path = opts.warmup_model_path + 'train_list_1.pkl'
        val_list_1_path = opts.warmup_model_path + 'val_list_1.pkl'

    # random leave out 10% of the training set as validation set
    val_list_1 = np.random.choice(train_list_1, int(len(train_list_1)*opts.val_ratio), replace=False)
    train_list_1 = list(set(train_list_1) - set(val_list_1))
    assert len(train_list_1) == opts.initial_num_ptb - int(opts.initial_num_ptb*opts.val_ratio)
    assert set(train_list_1).isdisjoint(set(val_list_1))
    
    # if train_list_1 exists, load it
    if os.path.exists(train_list_1_path):
        with open(train_list_1_path, 'rb') as f:
            train_list_1 = pickle.load(f)
        print('loading train_list_1 from', train_list_1_path)
    else:
        with open(train_list_1_path, 'wb') as f:
            pickle.dump(train_list_1, f)
        print('saving train_list_1 to', train_list_1_path)

    # if val_list_1 exists, load it
    if os.path.exists(val_list_1_path):
        with open(val_list_1_path, 'rb') as f:
            val_list_1 = pickle.load(f)
        print('loading val_list_1 from', val_list_1_path)
    else:
        with open(val_list_1_path, 'wb') as f:
            pickle.dump(val_list_1, f)
        print('saving val_list_1 to', val_list_1_path)

    ### 2.2 Get the first training dataloader and infer index -------------------------------------
    dataset = SCDataset(base_dir=opts.path_dir,
                        dataset_name=opts.dataset_name, 
                        adata_path=adata_path,
                        leave_out_test_set=opts.ptb_leave_out_list,
                        representation_type=opts.representation_type, 
                        representation_type_2=None,
                        representation_type_3=None,
                        min_counts=opts.batch_size,
                        random_seed=opts.random_seed)
    
    train_idx_1 = np.where(np.isin(dataset.ptb_names, train_list_1))[0].tolist()
    print(len(train_idx_1))
    val_idx_1 = np.where(np.isin(dataset.ptb_names, val_list_1))[0].tolist()
    print(len(val_idx_1))
    infer_idx = np.where(np.isin(dataset.ptb_names, ptb_leave_out_list))[0].tolist()
    print(len(infer_idx))

    dataset_train_1 = Subset(dataset, train_idx_1)
    dataloader_train = DataLoader(
        dataset_train_1,
        batch_sampler=SCDATA_sampler(dataset_train_1, opts.batch_size, dataset.ptb_names[train_idx_1]),
        num_workers = 0
    )
    
    dataset_val_1 = Subset(dataset, val_idx_1)
    dataloader_val = DataLoader(
        dataset_val_1,
        batch_sampler=SCDATA_sampler(dataset_val_1, opts.batch_size, dataset.ptb_names[val_idx_1]),
        num_workers = 0
    )
    
    dataset_infer = Subset(dataset, infer_idx)
    dataloader_infer = DataLoader(
        dataset_infer,
        batch_sampler=SCDATA_sampler(dataset_infer, opts.batch_size, dataset.ptb_names[infer_idx]),
        num_workers = 0
    )

    all_idx = np.arange(len(dataset))
    dataset_all = Subset(dataset, all_idx)
    dataloader_all = DataLoader(
        dataset_all,
        batch_sampler=SCDATA_sampler(dataset_all, opts.batch_size, dataset.ptb_names[all_idx]),
        num_workers = 0
    )

    # save the dataloader_infer
    with open(opts.output_parent_dir+'infer_idx.pkl', 'wb') as f:
        pickle.dump(infer_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('saved inference index (test set) to', opts.output_parent_dir+'infer_idx.pkl')
    
    return dataset, dataloader_train, dataloader_val, dataloader_infer, dataloader_all, gene_emb_raw, train_list_1, val_list_1, opts
    

    

