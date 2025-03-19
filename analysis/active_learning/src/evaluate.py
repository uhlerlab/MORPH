import numpy as np
import pandas as pd
import pickle
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from warnings import filterwarnings
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from argparse import ArgumentParser
import seaborn as sns
import gc
import json
import random
import os
from helpers import calculate_cka_for_representations, get_predictions, get_mmd_to_ctrl_all
import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from sklearn.metrics import r2_score

filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'morph'))
sys.path.append(morph_path)

from inference import *
from utils import MMD_loss

### Set up experiment environments ------------------------------------------------
def main():
    parser = ArgumentParser(description="Run evaluations")
    parser.add_argument('--random_seed', type=int, default=12, help="Random seed (change this for different runs)")
    parser.add_argument('--dataset_name', type=str, required=True, default='replogle_k562_essential_hvg', help="Name of the dataset")
    parser.add_argument('--representation_type', type=str, default='DepMap_GeneEffect', help='Type of representation to use')
    parser.add_argument('--leave_out_test_set_id', type=str, required=True, help="ID of the test set to leave out")
    parser.add_argument('--method', type=str, required=True, help="Method to use: 'baseline', 'active_learning_kmeans_center', 'uncertainty', 'learn_loss_mse' ")
    parser.add_argument('--device', type=str, default='cuda:5', help="Device to use (default: cuda:5)")
    parser.add_argument('--use_prior', type=str, default='False', help="Whether to use prior in the model (default: False)")
    parser.add_argument('--prior_only', type=str, default='False', help="Whether to use prior only in the model (default: False)")
    
    parser.add_argument('--kernel_type', type=str, default='laplacian', help="Type of kernel to use (default: rbf, laplacian, euclidean, cosine)")
    parser.add_argument('--kernel_gamma', type=float, default=1.0, help="Gamma value for the kernel (default: 1.0)")

    parser.add_argument('--cluster_first', type=str, default='False', help="Whether to cluster first (default: False)")
    
    parser.add_argument('--n_top_deg', type=int, default=50, help="Number of top differentially expressed genes to test on")

    parser.add_argument('--num_round', type=int, default=5, help="Number of rounds")
    parser.add_argument('--initial_num_ptb', type=int, default=100, help="Number of initial perturbations to sample (for the first round)")
    parser.add_argument('--num_ptb', type=int, default=100, help="Number of total perturbations to sample each time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    parser.add_argument('--initial_epochs', type=int, default=100, help="Number of initial epochs")
    parser.add_argument('--incremental_epochs', type=int, default=100, help="Number of incremental epochs")
    parser.add_argument('--decay_factor', type=float, default=1.0, help="Decay factor for learning rate")
    parser.add_argument('--tolerance_epochs', type=int, default=20, help="Tolerance epochs")

    parser.add_argument('--val_ratio', type=float, default=0.1, help="Validation ratio")
    parser.add_argument('--train_style', type=str, default='learn_from_scratch', help="Training style: 'learn_from_scratch' or 'tune_on_whole'")

    args = parser.parse_args()

    args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.path_dir = os.path.abspath(os.path.join(args.base_dir, '..', '..'))
    
    torch.set_num_threads(6)
    random_seed = args.random_seed
    device = args.device
    dataset_name = args.dataset_name
    representation_type = args.representation_type
    leave_out_test_set_id = args.leave_out_test_set_id
    method = args.method
    use_prior = args.use_prior
    prior_only = args.prior_only
    kernel_type = args.kernel_type
    kernel_gamma = args.kernel_gamma
    cluster_first = args.cluster_first
    n_top_deg = args.n_top_deg
    train_tyle = args.train_style

    print('dataset_name:', dataset_name)
    print('representation_type:', representation_type)
    print('leave_out_test_set_id:', leave_out_test_set_id)
    print('method:', method)
    print('use_prior:', use_prior)
    print('prior_only:', prior_only)
    print('kernel_type:', kernel_type)
    print('cluster_first:', cluster_first)
    print('val_ratio:', args.val_ratio)
    print('train_style:', train_tyle)

    # specify the parameters here -----
    num_round = args.num_round 
    batch_size = args.batch_size
    output_parent_dir = f'{args.base_dir}/results/'+args.dataset_name+'/'+ \
        args.leave_out_test_set_id+'/'+args.representation_type+ '/' + \
            str(args.num_round)+'_round_'+str(args.num_ptb)+'_ptb_' + str(args.initial_num_ptb) + '_initial_ptb/' + "seed_"+str(args.random_seed)+'/' 
    print('output_dir:', output_parent_dir)

    # create save_name
    method_name_whole = method
    if use_prior == 'False':
        pass
    else:
        method_name_whole += '_prior'
        if prior_only == 'True':
            method_name_whole += '_only'
    if cluster_first == 'True':
        method_name_whole += "_cluster_first"
    if method == 'maxdist':
        if kernel_type in ['rbf', 'laplacian']:
            method_name_whole += '_kernel_'+kernel_type + '_gamma_' + str(kernel_gamma)
        elif kernel_type in ['euclidean', 'cosine']:
            method_name_whole += '_'+kernel_type
    print('method_name_whole:', method_name_whole)  
    
    output_dir = os.path.join(output_parent_dir, method_name_whole)
    print('output_dir:', output_dir)

    # read in infer idx pickle file
    with open(output_parent_dir+'infer_idx.pkl', 'rb') as f:
        infer_idx = pickle.load(f)

    # read in the config file
    with open(output_dir+'/config.json') as f:
        opts = json.load(f)

    # read in the single-cell data
    scdata_file = pd.read_csv(f'{args.path_dir}/data/scdata_file_path.csv')
    adata_path = scdata_file[scdata_file['dataset'] == dataset_name]['file_path'].values[0]
    adata = sc.read_h5ad(adata_path)
    print('loading single-cell data from: ', adata_path)
    sc.tl.rank_genes_groups(adata, groupby='gene', reference='non-targeting', n_genes=50, use_raw=False, rankby_abs=True)

    # read in distance to ctrl list    
    mmd_to_ctrl_path = f'{args.path_dir}/data/{dataset_name}_mmd_distance_1500_to_non-targeting.csv'
    mmd_to_ctrl_df = pd.read_csv(mmd_to_ctrl_path)
    mmd_to_ctrl_df = mmd_to_ctrl_df[mmd_to_ctrl_df['ptb_gene'].isin(opts['ptb_leave_out_list'])]
    mmd_to_ctrl_sigma = mmd_to_ctrl_df['mmd_sigma'].values[0]

    # read in gene_emb file
    embedding_file_df = pd.read_csv(f'{args.path_dir}/data/perturb_embed_file_path.csv')
    embedding_file_subset = embedding_file_df[embedding_file_df['representation_type']==args.representation_type]
    embed_file = embedding_file_subset['file_path'].values[0]
    print('loading gene embeddings from', embed_file)
    with open(embed_file, 'rb') as f:
        gene_emb_raw = pickle.load(f)

    # load test set data
    split_path = f'{args.path_dir}/data/{args.dataset_name}_splits.csv'
    split_df = pd.read_csv(split_path)
    print('loading split data from: ', split_path)
    print('leave_out_test_set_id: ', leave_out_test_set_id)
    ptb_leave_out_list = split_df[split_df['test_set_id'] == leave_out_test_set_id]['test_set'].apply(lambda x: x.split(',')).values[0]
    assert set(ptb_leave_out_list) == set(opts['ptb_leave_out_list']), 'Error: ptb_leave_out_list and opts have different values'

    # load inference dataloader 
    infer_dataloader_path = f'{args.base_dir}/results/'+dataset_name+'/'+ leave_out_test_set_id+'/infer_dataloader.pkl'
    with open(infer_dataloader_path, 'rb') as f:
        dataloader = pickle.load(f)
    print('loading inference dataloader from: ', infer_dataloader_path)
    assert set(infer_idx) == set(dataloader.dataset.indices), 'Error: infer_idx and dataloader have different indices'

    ### 2. Calculate losses ------------------------------------------------
    mmd_loss =  MMD_loss(fix_sigma=200, kernel_num=10)
    from sklearn.metrics import mean_squared_error as mse
    results_df = pd.DataFrame()
    for round in range(num_round):

        print('Evaluating round:', round)
        last_model_path_temp = output_dir+'/round_'+str(round)+'_last_model.pt'
        best_train_model_path_temp = output_dir+'/round_'+str(round)+'_best_train_model.pt'
        best_val_model_path_temp = output_dir+'/round_'+str(round)+'_best_val_model.pt'

        last_model = torch.load(last_model_path_temp, map_location=device)
        last_model = last_model.to(device)
        last_model.eval()

        best_train_model = torch.load(best_train_model_path_temp, map_location=device)
        best_train_model = best_train_model.to(device)
        best_train_model.eval()

        if args.val_ratio > 0:
            best_val_model = torch.load(best_val_model_path_temp, map_location=device)
            best_val_model = best_val_model.to(device)
            best_val_model.eval()
        else:
            best_val_model = None
        
        # Calculate cka scores for different models
        gt_mtx_path = f"{args.base_dir}/kernels/" + args.dataset_name + "/ground_truth_delta/"
        alignment_score_last_model, _ = calculate_cka_for_representations(gene_emb_raw,
                                                                          opts['all_ptb_targets_with_embeddings'], 
                                                                          last_model,
                                                                          args.batch_size,
                                                                          gt_mtx_path,
                                                                          device)

        alignment_score_best_train_model, _ = calculate_cka_for_representations(gene_emb_raw,
                                                                                opts['all_ptb_targets_with_embeddings'], 
                                                                                best_train_model,
                                                                                args.batch_size,
                                                                                gt_mtx_path,
                                                                                device)

        if args.val_ratio > 0:
            alignment_score_best_val_model, _ = calculate_cka_for_representations(gene_emb_raw,
                                                                                  opts['all_ptb_targets_with_embeddings'], 
                                                                                  best_val_model,
                                                                                  args.batch_size,
                                                                                  gt_mtx_path,
                                                                                  device)

        # Get predictions of the test set for different models 
        pred_vae_last_model, pred_vae_best_train_model, pred_vae_best_val_model,\
              C_y, mmd_to_ctrl_last_model, mmd_to_ctrl_best_train_model, \
                mmd_to_ctrl_best_val_model =  get_predictions(dataloader, args,
                                     last_model, best_train_model, best_val_model,
                                     ptb_leave_out_list,
                                     batch_size,
                                     mmd_to_ctrl_sigma,
                                     device)
        assert len(set(C_y) - set(ptb_leave_out_list)) == 0, 'Error: C_y has perturbations not in ptb_leave_out_list'

        # Get the mmd distance from predictions to ctrl (evaluated on all dataloader)
        mmd_to_ctrl_all_last_model, mmd_to_ctrl_all_best_train_model, mmd_to_ctrl_all_best_val_model = \
            get_mmd_to_ctrl_all(args, 
                                last_model, best_train_model, best_val_model, 
                                mmd_to_ctrl_sigma, device)
        
        # Calculating losses for the test set -----------------------------------------------------------
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        mmd_loss_vae_last_model = {}
        mmd_loss_vae_best_train_model = {}
        mmd_loss_vae_best_val_model = {}
        rmse_loss_vae_last_model = {}
        rmse_loss_vae_best_train_model = {}
        rmse_loss_vae_best_val_model = {}
        r2_vae_last_model = {}
        r2_vae_best_train_model = {}
        r2_vae_best_val_model = {}
        
        for pert in tqdm(set(C_y), desc='Calculating losses for round '+str(round)):
            pert_name = pert
            deg_cat = f'{pert_name}'
            deg_list = adata.uns['rank_genes_groups']['names'][deg_cat]

            y_true = adata[adata.obs['gene'] == pert_name].X
            y_ctrl = adata[adata.obs['gene'] == 'non-targeting'].X
            y_pred_last_model = pred_vae_last_model[pert_name]
            y_pred_best_train_model = pred_vae_best_train_model[pert_name]
            y_pred_best_val_model = pred_vae_best_val_model[pert_name]

            degs = [np.where(adata.var_names == gene)[0][0] for gene in deg_list[:n_top_deg]]
            
            y_true_deg = y_true[:, degs]
            y_pred_last_model_deg = y_pred_last_model[:, degs]
            y_pred_best_train_model_deg = y_pred_best_train_model[:, degs]
            if args.val_ratio > 0:
                y_pred_best_val_model_deg = y_pred_best_val_model[:, degs]
            
            # Calculate losses for last model
            rmse_loss_vae_last_model[pert_name] = np.sqrt(mse(y_true_deg.mean(0), y_pred_last_model_deg.mean(0)))
            r2_vae_last_model[pert_name] = max(r2_score(y_true_deg.mean(0), y_pred_last_model_deg.mean(0)),0)
            
            # Calculate losses for best train model
            rmse_loss_vae_best_train_model[pert_name] = np.sqrt(mse(y_true_deg.mean(0), y_pred_best_train_model_deg.mean(0)))
            r2_vae_best_train_model[pert_name] = max(r2_score(y_true_deg.mean(0), y_pred_best_train_model_deg.mean(0)),0)
            
            # Calculate losses for best val model
            if args.val_ratio > 0:
                rmse_loss_vae_best_val_model[pert_name] = np.sqrt(mse(y_true_deg.mean(0), y_pred_best_val_model_deg.mean(0)))
                r2_vae_best_val_model[pert_name] = max(r2_score(y_true_deg.mean(0), y_pred_best_val_model_deg.mean(0)),0)
                
            # Empirical MMD is divided by sample size, so we should calculate MMD by batches and then take the average
            min_sample_size = min(y_pred_last_model_deg.shape[0], y_true_deg.shape[0])
            y_true_deg = y_true_deg[np.random.choice(y_true_deg.shape[0], min_sample_size, replace=False),:]
            y_pred_last_model_deg = y_pred_last_model_deg[np.random.choice(y_pred_last_model_deg.shape[0], min_sample_size, replace=False),:]
            y_pred_best_train_model_deg = y_pred_best_train_model_deg[np.random.choice(y_pred_best_train_model_deg.shape[0], min_sample_size, replace=False),:]
            
            y_ctrl = y_ctrl[np.random.choice(y_ctrl.shape[0], min_sample_size, replace=False),:]
            y_pred_last_model = y_pred_last_model[np.random.choice(y_pred_last_model.shape[0], min_sample_size, replace=False),:]
            y_pred_best_train_model = y_pred_best_train_model[np.random.choice(y_pred_best_train_model.shape[0], min_sample_size, replace=False),:]
            if args.val_ratio > 0:
                y_pred_best_val_model_deg = y_pred_best_val_model_deg[np.random.choice(y_pred_best_val_model_deg.shape[0], min_sample_size, replace=False),:]
                y_pred_best_val_model = y_pred_best_val_model[np.random.choice(y_pred_best_val_model.shape[0], min_sample_size, replace=False),:]

            # some quick checks
            assert y_true_deg.shape == y_pred_last_model_deg.shape == y_pred_best_train_model_deg.shape, 'Error: y_true_deg and y_pred_last_model_deg have different shapes'
            assert y_ctrl.shape == y_pred_last_model.shape == y_pred_best_train_model.shape, 'Error: y_ctrl and y_pred_last_model have different shapes'
            if args.val_ratio > 0:
                assert y_true_deg.shape == y_pred_best_val_model_deg.shape, 'Error: y_true_deg and y_pred_best_val_model_deg have different shapes'
                assert y_ctrl.shape == y_pred_best_val_model.shape, 'Error: y_ctrl and y_pred_best_val_model have different shapes'

            # begin calculations
            num_batches = min_sample_size // batch_size
            for i in range(num_batches):
                mmd_loss_temp_last_model = mmd_loss(torch.tensor(y_pred_last_model_deg[i*batch_size:(i+1)*batch_size]), torch.tensor(y_true_deg[i*batch_size:(i+1)*batch_size])).item()
                mmd_loss_temp_best_train_model = mmd_loss(torch.tensor(y_pred_best_train_model_deg[i*batch_size:(i+1)*batch_size]), torch.tensor(y_true_deg[i*batch_size:(i+1)*batch_size])).item()
                
                if pert_name not in mmd_loss_vae_last_model.keys():
                    mmd_loss_vae_last_model[pert_name] = [mmd_loss_temp_last_model]
                else:
                    mmd_loss_vae_last_model[pert_name].append(mmd_loss_temp_last_model)

                if pert_name not in mmd_loss_vae_best_train_model.keys():
                    mmd_loss_vae_best_train_model[pert_name] = [mmd_loss_temp_best_train_model]
                else:
                    mmd_loss_vae_best_train_model[pert_name].append(mmd_loss_temp_best_train_model)

                if args.val_ratio > 0:
                    mmd_loss_temp_best_val_model = mmd_loss(torch.tensor(y_pred_best_val_model_deg[i*batch_size:(i+1)*batch_size]), torch.tensor(y_true_deg[i*batch_size:(i+1)*batch_size])).item()
                    if pert_name not in mmd_loss_vae_best_val_model.keys():
                        mmd_loss_vae_best_val_model[pert_name] = [mmd_loss_temp_best_val_model]
                    else:
                        mmd_loss_vae_best_val_model[pert_name].append(mmd_loss_temp_best_val_model)
            
            # free up memory
            del y_true, y_ctrl, y_pred_last_model, y_pred_best_train_model
            del y_true_deg, y_pred_last_model_deg, y_pred_best_train_model_deg
            if args.val_ratio > 0:
                del y_pred_best_val_model, y_pred_best_val_model_deg
            gc.collect()  # Run garbage collector
                    
        assert set(mmd_loss_vae_last_model.keys()) == set(mmd_loss_vae_best_train_model.keys()) == set(C_y), 'Error: mmd_loss_vae_last_model and mmd_loss_vae_best_train_model have different keys'
        if args.val_ratio > 0:
            assert set(mmd_loss_vae_best_val_model.keys()) == set(C_y), 'Error: mmd_loss_vae_best_val_model has different keys'
            
        # free up memory
        del pred_vae_last_model, pred_vae_best_train_model
        if args.val_ratio > 0:
            del pred_vae_best_val_model
        gc.collect()  # Run garbage collector

        # Print out losses results
        print('Size of test set: ', len(set(C_y)))
        print('-------------------')
        print('which model it uses:', method, 'round:', round)
        print('-------------------')
        print('Using top 50 marker genes')

        # Function to summarize and print MMD loss results
        def summarize_mmd_losses(loss_dict):
            loss_summary = {k: (np.average(v), np.std(v)) for k, v in loss_dict.items()}
            loss_mean = np.mean([i[0] for i in loss_summary.values()])
            worst_10_percent_mean = np.mean([i[0] for i in sorted(loss_summary.values(), key=lambda x: x[0], reverse=True)[:int(len(loss_summary) * 0.1)]])
            return loss_mean, worst_10_percent_mean, loss_summary

        # Summarize MMD losses for each model
        mmd_last_model_mean, mmd_last_model_worst_10, mmd_loss_summary_last_model = summarize_mmd_losses(mmd_loss_vae_last_model)
        mmd_best_train_model_mean, mmd_best_train_model_worst_10, mmd_loss_summary_best_train_model = summarize_mmd_losses(mmd_loss_vae_best_train_model)

        # Print out summarized MMD results
        print(f'VAE, mmd (last model): mean = {mmd_last_model_mean:.5f}, worst 10% = {mmd_last_model_worst_10:.5f}')
        print(f'VAE, mmd (best train model): mean = {mmd_best_train_model_mean:.5f}, worst 10% = {mmd_best_train_model_worst_10:.5f}')
        if args.val_ratio > 0:
            mmd_best_val_model_mean, mmd_best_val_model_worst_10, mmd_loss_summary_best_val_model = summarize_mmd_losses(mmd_loss_vae_best_val_model)
            print(f'VAE, mmd (best val model): mean = {mmd_best_val_model_mean:.5f}, worst 10% = {mmd_best_val_model_worst_10:.5f}')

        # Function to summarize other losses
        def summarize_other_losses(loss_dict, reverse=True):
            loss_mean = np.mean([i for i in loss_dict.values()])
            worst_10_percent_mean = np.mean([i for i in sorted(loss_dict.values(), reverse=reverse)[:int(len(loss_dict) * 0.1)]])
            return loss_mean, worst_10_percent_mean

        # Summarize other losses for each model -----------------------------------------------------
        rmse_last_model_mean, rmse_last_model_worst_10 = summarize_other_losses(rmse_loss_vae_last_model, reverse=True)
        rmse_best_train_model_mean, rmse_best_train_model_worst_10 = summarize_other_losses(rmse_loss_vae_best_train_model, reverse=True)
        
        r2_last_model_mean, r2_last_model_worst_10 = summarize_other_losses(r2_vae_last_model, reverse=False)
        r2_best_train_model_mean, r2_best_train_model_worst_10 = summarize_other_losses(r2_vae_best_train_model, reverse=False)
        
        if args.val_ratio > 0:
            rmse_best_val_model_mean, rmse_best_val_model_worst_10 = summarize_other_losses(rmse_loss_vae_best_val_model, reverse=True)
            r2_best_val_model_mean, r2_best_val_model_worst_10 = summarize_other_losses(r2_vae_best_val_model, reverse=False)
        
        # Get list of perturbations with largest values in mmd_to_ctrl --------------------------------
        mmd_to_ctrl_last_model_summary = {k: (np.average(v), np.std(v)) for k, v in mmd_to_ctrl_last_model.items()}
        mmd_to_ctrl_last_model_summary = pd.DataFrame(mmd_to_ctrl_last_model_summary).T.rename(columns={0: 'mean', 1: 'std'})
        mmd_to_ctrl_all_last_model_summary = {k: (np.average(v), np.std(v)) for k, v in mmd_to_ctrl_all_last_model.items()}
        mmd_to_ctrl_all_last_model_summary = pd.DataFrame(mmd_to_ctrl_all_last_model_summary).T.rename(columns={0: 'mean', 1: 'std'})

        mmd_to_ctrl_best_train_model_summary = {k: (np.average(v), np.std(v)) for k, v in mmd_to_ctrl_best_train_model.items()}
        mmd_to_ctrl_best_train_model_summary = pd.DataFrame(mmd_to_ctrl_best_train_model_summary).T.rename(columns={0: 'mean', 1: 'std'})
        mmd_to_ctrl_all_best_train_model_summary = {k: (np.average(v), np.std(v)) for k, v in mmd_to_ctrl_all_best_train_model.items()}
        mmd_to_ctrl_all_best_train_model_summary = pd.DataFrame(mmd_to_ctrl_all_best_train_model_summary).T.rename(columns={0: 'mean', 1: 'std'})

        if args.val_ratio > 0:
            mmd_to_ctrl_best_val_model_summary = {k: (np.average(v), np.std(v)) for k, v in mmd_to_ctrl_best_val_model.items()}
            mmd_to_ctrl_best_val_model_summary = pd.DataFrame(mmd_to_ctrl_best_val_model_summary).T.rename(columns={0: 'mean', 1: 'std'})
            mmd_to_ctrl_all_best_val_model_summary = {k: (np.average(v), np.std(v)) for k, v in mmd_to_ctrl_all_best_val_model.items()}
            mmd_to_ctrl_all_best_val_model_summary = pd.DataFrame(mmd_to_ctrl_all_best_val_model_summary).T.rename(columns={0: 'mean', 1: 'std'})
        
        # distance to ctrl evaluations
        mmd_to_ctrl_df_summary = mmd_to_ctrl_df[['ptb_gene', 'mean_mmd_distance']].set_index('ptb_gene')
        mmd_to_ctrl_df_summary = mmd_to_ctrl_df_summary.loc[mmd_to_ctrl_last_model_summary.index]
        true_rank_list = mmd_to_ctrl_df_summary.sort_values(by='mean_mmd_distance', ascending=False, inplace=False).index.values.tolist()
        last_model_rank_list = mmd_to_ctrl_last_model_summary.sort_values(by='mean', ascending=False, inplace=False).index.values.tolist()
        precision_10_last_model = len(set(last_model_rank_list[:10]).intersection(set(true_rank_list[:10])))/10

        last_model_rank_list_all = mmd_to_ctrl_all_last_model_summary.sort_values(by='mean', ascending=False, inplace=False).index.values.tolist()

        mmd_to_ctrl_best_train_model_summary = mmd_to_ctrl_best_train_model_summary.loc[mmd_to_ctrl_last_model_summary.index]
        best_train_model_rank_list = mmd_to_ctrl_best_train_model_summary.sort_values(by='mean', ascending=False, inplace=False).index.values.tolist()
        precision_10_best_train_model = len(set(best_train_model_rank_list[:10]).intersection(set(true_rank_list[:10])))/10

        best_train_model_rank_list_all = mmd_to_ctrl_all_best_train_model_summary.sort_values(by='mean', ascending=False, inplace=False).index.values.tolist()

        if args.val_ratio > 0:
            mmd_to_ctrl_best_val_model_summary = mmd_to_ctrl_best_val_model_summary.loc[mmd_to_ctrl_last_model_summary.index]
            best_val_model_rank_list = mmd_to_ctrl_best_val_model_summary.sort_values(by='mean', ascending=False, inplace=False).index.values.tolist()
            precision_10_best_val_model = len(set(best_val_model_rank_list[:10]).intersection(set(true_rank_list[:10])))/10

            best_val_model_rank_list_all = mmd_to_ctrl_all_best_val_model_summary.sort_values(by='mean', ascending=False, inplace=False).index.values.tolist()

        # Save results into dataframe
        new_row_df = pd.DataFrame([{
            'test_method': method,
            'round': round,
            
            # losses for test set/ all ----------------------------------------------------------- 
            'mmd_last_model': mmd_last_model_mean,
            'mmd_worst_10_last_model': mmd_last_model_worst_10,

            'mmd_best_train_model': mmd_best_train_model_mean,
            'mmd_worst_10_best_train_model': mmd_best_train_model_worst_10,

            'mmd_best_val_model': mmd_best_val_model_mean if args.val_ratio > 0 else None,
            'mmd_worst_10_best_val_model': mmd_best_val_model_worst_10 if args.val_ratio > 0 else None,

            'rmse_last_model': rmse_last_model_mean,
            'rmse_worst_10_last_model': rmse_last_model_worst_10,

            'rmse_best_train_model': rmse_best_train_model_mean,
            'rmse_worst_10_best_train_model': rmse_best_train_model_worst_10,

            'rmse_best_val_model': rmse_best_val_model_mean if args.val_ratio > 0 else None,
            'rmse_worst_10_best_val_model': rmse_best_val_model_worst_10 if args.val_ratio > 0 else None,

            'r2_last_model': r2_last_model_mean,
            'r2_worst_10_last_model': r2_last_model_worst_10,

            'r2_best_train_model': r2_best_train_model_mean,
            'r2_worst_10_best_train_model': r2_best_train_model_worst_10,
            
            'r2_best_val_model': r2_best_val_model_mean if args.val_ratio > 0 else None,
            'r2_worst_10_best_val_model': r2_best_val_model_worst_10 if args.val_ratio > 0 else None,
            
            'alignment_score_last_model': alignment_score_last_model,
            'alignment_score_best_train_model': alignment_score_best_train_model,
            'alignment_score_best_val_model': alignment_score_best_val_model if args.val_ratio > 0 else None,

            'precision_10_last_model': precision_10_last_model,
            'precision_10_best_train_model': precision_10_best_train_model,
            'precision_10_best_val_model': precision_10_best_val_model if args.val_ratio > 0 else None,

            # rank of test set -----------------------------------------------------------
            'true_rank_list': true_rank_list,
            'true_mmd_list': mmd_to_ctrl_df_summary.sort_values(by='mean_mmd_distance', ascending=False, inplace=False)['mean_mmd_distance'].values.tolist(),
            
            'last_model_rank_list': last_model_rank_list,
            'last_model_mmd_list': mmd_to_ctrl_last_model_summary.sort_values(by='mean', ascending=False, inplace=False)['mean'].values.tolist(),
            'last_model_rank_list_all': last_model_rank_list_all,
            'last_model_mmd_list_all': mmd_to_ctrl_all_last_model_summary.sort_values(by='mean', ascending=False, inplace=False)['mean'].values.tolist(),

            'best_train_model_rank_list': best_train_model_rank_list,
            'best_train_model_mmd_list': mmd_to_ctrl_best_train_model_summary.sort_values(by='mean', ascending=False, inplace=False)['mean'].values.tolist(),
            'best_train_model_rank_list_all': best_train_model_rank_list_all,
            'best_train_model_mmd_list_all': mmd_to_ctrl_all_best_train_model_summary.sort_values(by='mean', ascending=False, inplace=False)['mean'].values.tolist(),

            'best_val_model_rank_list': best_val_model_rank_list if args.val_ratio > 0 else None,
            'best_val_model_mmd_list': mmd_to_ctrl_best_val_model_summary.sort_values(by='mean', ascending=False, inplace=False)['mean'].values.tolist() if args.val_ratio > 0 else None,
            'best_val_model_rank_list_all': best_val_model_rank_list_all if args.val_ratio > 0 else None,
            'best_val_model_mmd_list_all': mmd_to_ctrl_all_best_val_model_summary.sort_values(by='mean', ascending=False, inplace=False)['mean'].values.tolist() if args.val_ratio > 0 else None,
        }])
        
        results_df = pd.concat([results_df, new_row_df], ignore_index=True)

    ### 3. Save results ------------------------------------------------
    df_save_path = output_dir+'/results_df.csv'
    results_df.to_csv(df_save_path, index=False)
    print('Results saved to:', df_save_path)

if __name__ == "__main__":
    main()
