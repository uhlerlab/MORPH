import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import scanpy as sc
import pandas as pd
import json
import os

# Read in and process the single-cell perturbation dataset
class SCDataset(Dataset):
    def __init__(self,
                dataset='replogle_rpe1',
                leave_out_test_set_id=None,
                leave_out_test_set=None,
                perturb_targets=None, 
                representation_type=None, representation_type_2=None, representation_type_3=None,
                min_counts=32,
                random_seed=None,
                use_hvg='False',
                gene_embs = None):
        super(Dataset, self).__init__()
        
        self.seed = random_seed
        
        # first read in the csv file of dataset
        scdata_file = pd.read_csv('./data/scdata_file_path.csv')
        adata_path = scdata_file[scdata_file['dataset'] == dataset][scdata_file['use_hvg'] == (use_hvg == 'True')]['file_path'].values[0]
        dataset_name = dataset
        if use_hvg == 'True':
            dataset_name = dataset_name + '_hvg'

        # if no gene_embs are provided, read in the ptb embedding dictionaries
        if gene_embs is None:
            # let's read in the ptb embedding dictionaries here and do the filtering
            embedding_file_df = pd.read_csv('./data/perturb_embed_file_path.csv')
            embedding_file_subset = embedding_file_df[embedding_file_df['representation_type']==representation_type]

            # Read in correct embedding file path
            if representation_type == 'Baseline':
                embed_file = None
            elif embedding_file_subset.shape[0] == 1:
                embed_file = embedding_file_subset['file_path'].values[0]
            elif embedding_file_subset.shape[0] > 1:
                embed_file = embedding_file_subset[embedding_file_subset['dataset']==dataset][embedding_file_subset['use_hvg'] == (use_hvg == 'True')]['file_path'].values[0]
            if embed_file is not None:
                with open(embed_file, 'rb') as file:
                    gene_embs = pickle.load(file)
                    print("Loading perturbation target embeddings from", embed_file)
            else:
                gene_embs = None
            
            # Read in the second perturbation target embeddings
            if representation_type_2 is not None:
                embedding_file_subset_2 = embedding_file_df[embedding_file_df['representation_type']==representation_type_2]
                if embedding_file_subset_2.shape[0] == 1:
                    embed_file_2 = embedding_file_subset_2['file_path'].values[0]
                elif embedding_file_subset_2.shape[0] > 1:
                    embed_file_2 = embedding_file_subset_2[embedding_file_subset_2['dataset']==dataset][embedding_file_subset_2['use_hvg'] == (use_hvg == 'True')]['file_path'].values[0]
                if embed_file_2 is not None:
                    with open(embed_file_2, 'rb') as file:
                        gene_embs_2 = pickle.load(file)
                        print("Loading perturbation target embeddings from", embed_file_2)
            else:
                gene_embs_2 = None

            # Read in the third perturbation target embeddings
            if representation_type_3 is not None:
                embedding_file_subset_3 = embedding_file_df[embedding_file_df['representation_type']==representation_type_3]
                if embedding_file_subset_3.shape[0] == 1:
                    embed_file_3 = embedding_file_subset_3['file_path'].values[0]
                elif embedding_file_subset_3.shape[0] > 1:
                    embed_file_3 = embedding_file_subset_3[embedding_file_subset_3['dataset']==dataset][embedding_file_subset_3['use_hvg'] == (use_hvg == 'True')]['file_path'].values[0]
                if embed_file_3 is not None:
                    with open(embed_file_3, 'rb') as file:
                        gene_embs_3 = pickle.load(file)
                        print("Loading perturbation target embeddings from", embed_file_3)
            else:
                gene_embs_3 = None
        
        # Get the list of perturbation targets leave out for testing
        if leave_out_test_set is None:
            split_path = './data/' + dataset_name + '_splits.csv'
            split_df = pd.read_csv(split_path)
            print('loading split data from: ', split_path)
            print('leave_out_test_set_id: ', leave_out_test_set_id)
            if leave_out_test_set_id == 'all_train':
                ptb_leave_out_list = ''
            else:
                ptb_leave_out_list = split_df[split_df['test_set_id'] == leave_out_test_set_id]['test_set'].apply(lambda x: x.split(',')).values[0]
        else:
            ptb_leave_out_list = leave_out_test_set
        
        # Filter out the perturbation targets with counts < min_counts
        adata = sc.read_h5ad(adata_path)
        print('Loaded adata from: ', adata_path)
        gene_counts = adata.obs['gene'].value_counts()
        gene_counts = gene_counts[gene_counts >= min_counts]
        print('Length of raw ptb_leave_out_list: ', len(ptb_leave_out_list))
        ptb_leave_out_list = [ptb for ptb in ptb_leave_out_list if ptb in gene_counts.index]
        print('Length of filtered ptb_leave_out_list (after removing gene with counts < ', min_counts ,'): ', len(ptb_leave_out_list))
        
        # Filter out perturbation targets that do not have embeddings from adata and ptb_leave_out_list
        if gene_embs is not None:
            if gene_embs_2 is not None:
                if gene_embs_3 is not None:
                    ptb_lst_with_embs = list(set(gene_embs.keys()) & set(gene_embs_2.keys()) & set(gene_embs_3.keys()))
                else:
                    ptb_lst_with_embs = list(set(gene_embs.keys()) & set(gene_embs_2.keys()))
            else:
                ptb_lst_with_embs = gene_embs.keys()
            print("Checking for perturbation targets without embeddings in adata...")
            
            # Get the single genes in adata
            single_genes = adata.obs['gene'].unique().tolist()
            single_genes = [gene for gene in single_genes if '+' not in gene]
            single_genes.remove('non-targeting')

            # Get the list of single genes without embeddings
            single_genes_without_embs = [gene for gene in single_genes if gene not in ptb_lst_with_embs]
            if len(single_genes_without_embs) > 0:
                print('Perturbation targets without embeddings: ', single_genes_without_embs, 'There are ', len(single_genes_without_embs), ' perturbations without embeddings')
                print('Original shape of adata: ', adata.shape)

                # Create a function to check if any gene in the perturbation (single or double) is in single_genes_without_embs
                def contains_genes_without_embeds(perturbation):
                    genes = perturbation.split('+')
                    return any(gene in single_genes_without_embs for gene in genes)
                
                # Apply the function to filter adata
                adata = adata[~adata.obs['gene'].apply(contains_genes_without_embeds)]
                print('Removed perturbations without embeddings from adata')
                print('New shape of adata: ', adata.shape)
                
                # Update ptb_leave_out_list by removing perturbations without embeddings
                ptb_leave_out_list = [ptb for ptb in ptb_leave_out_list if not contains_genes_without_embeds(ptb)]
                print('Length of filtered ptb_leave_out_list (after removing perturbations without embeddings): ', len(ptb_leave_out_list))
        
        self.ptb_leave_out_list = ptb_leave_out_list
        
        # Get list of perturbation targets
        if perturb_targets is None:
            ptb_targets = list(set(adata.obs['gene']))
            ptb_targets.remove('non-targeting')
        else:
            ptb_targets = perturb_targets
        self.ptb_targets = ptb_targets
        print("There are", len(ptb_targets), "perturbation targets.")
        
        # Map the perturbation targets to the corresponding representation vectors
        ptb_adata = adata[adata.obs['gene']!='non-targeting'].copy()
        self.ptb_samples = ptb_adata.X
        self.ptb_names = ptb_adata.obs['gene'].values
        print("Mapping perturbation targets to the corresponding representation vectors with type:", representation_type)
        self.ptb_vector_dict, self.ptb_vectors_1, self.ptb_vectors_2 = map_ptb_features(ptb_targets, self.ptb_names,
                                                                                        representation_type=representation_type, 
                                                                                        gene_embs = gene_embs)
        
        if gene_embs_2 is not None:
            print("Mapping perturbation targets to the corresponding representation vectors with type:", representation_type_2)
            self.ptb_vector_dict_2, self.ptb_vectors_1_2, self.ptb_vectors_2_2 = map_ptb_features(ptb_targets, self.ptb_names,
                                                                                                  representation_type=representation_type_2, 
                                                                                                  gene_embs = gene_embs_2)
        else:
            self.ptb_vector_dict_2 = None
            self.ptb_vectors_1_2 = None
            self.ptb_vectors_2_2 = None

        if gene_embs_3 is not None:
            print("Mapping perturbation targets to the corresponding representation vectors with type:", representation_type_3)
            self.ptb_vector_dict_3, self.ptb_vectors_1_3, self.ptb_vectors_2_3 = map_ptb_features(ptb_targets, self.ptb_names,
                                                                                                  representation_type=representation_type_3, 
                                                                                                  gene_embs = gene_embs_3)
        else:
            self.ptb_vector_dict_3 = None
            self.ptb_vectors_1_3 = None
            self.ptb_vectors_2_3 = None

        del ptb_adata
        
        # Control samples
        self.ctrl_samples = adata[adata.obs['gene']=='non-targeting'].X.copy()

        # A random subset of control samples, matching in number to the perturbed samples, is selected for balanced comparison.
        np.random.seed(self.seed)
        self.rand_ctrl_samples = self.ctrl_samples[
            np.random.choice(self.ctrl_samples.shape[0], self.ptb_samples.shape[0], replace=True)
            ]
        del adata
    
    def __getitem__(self, item):
        # A control sample
        x = torch.from_numpy(self.rand_ctrl_samples[item].flatten()).double()
        
        # A perturbed sample
        y = torch.from_numpy(self.ptb_samples[item].flatten()).double()

        # The vector representation of the perturbation target (target 1)
        c_1 = torch.from_numpy(self.ptb_vectors_1[item]).double()
        
        # The vector representation of the perturbation target (target 2)
        c_2 = torch.from_numpy(self.ptb_vectors_2[item]).double()

        # Get perturbation target name
        ptb_name = self.ptb_names[item]

        if self.ptb_vector_dict_2 is not None:
            c_1_2 = torch.from_numpy(self.ptb_vectors_1_2[item]).double()
            c_2_2 = torch.from_numpy(self.ptb_vectors_2_2[item]).double()
            if self.ptb_vector_dict_3 is not None:
                c_1_3 = torch.from_numpy(self.ptb_vectors_1_3[item]).double()
                c_2_3 = torch.from_numpy(self.ptb_vectors_2_3[item]).double()
                return x, y, c_1, c_2, ptb_name, c_1_2, c_2_2, c_1_3, c_2_3
            else:
                return x, y, c_1, c_2, ptb_name, c_1_2, c_2_2
        else:
            return x, y, c_1, c_2, ptb_name
        
    def __len__(self):
        # return number of perturbed samples in the dataset
        return self.ptb_samples.shape[0]


def map_ptb_features(all_ptb_targets, ptb_ids, representation_type, gene_embs):
    
    if representation_type == 'Baseline' and gene_embs is None:
        'This function maps the perturbation targets to binary vectors.'
        # get all single ptb_targets
        split_ptb_targets = [target for item in all_ptb_targets for target in item.split('+')]
        split_ptb_targets = list(set(split_ptb_targets))
        gene_embs = {target: np.eye(len(split_ptb_targets))[i] for i, target in enumerate(split_ptb_targets)}
    
    ptb_features_1 = []
    ptb_features_2 = []
    for id in ptb_ids:
        if "+" in id:
            id_1 = id.split("+")[0]
            id_2 = id.split("+")[1]
            feature_1 = gene_embs.get(id_1)
            feature_2 = gene_embs.get(id_2)
            ptb_features_1.append(feature_1)
            ptb_features_2.append(feature_2)
        else:
            feature = gene_embs.get(id)
            ptb_features_1.append(feature)
            ptb_features_2.append(np.full(feature.shape, np.nan))
    return gene_embs, np.vstack(ptb_features_1), np.vstack(ptb_features_2)