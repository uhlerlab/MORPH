import numpy as np
from cka import linear_CKA, kernel_CKA
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import auc
import torch
import pickle
import gc
import matplotlib.pyplot as plt
import umap
import sys
from scipy.spatial.distance import euclidean, cosine
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'morph'))
sys.path.append(morph_path)
from utils import MMD_loss

# generate the training schedule
def generate_decreasing_schedule(initial_epochs, incremental_epochs, num_rounds, decay_factor):
    schedule = [0, initial_epochs]
    for round in range(1, num_rounds):
        next_increment = int(incremental_epochs * (decay_factor ** round))
        next_epoch = schedule[-1] + next_increment
        schedule.append(next_epoch)
    return schedule

def calculate_cka_for_representations(gene_emb_raw, 
                                      all_ptb_targets_with_embeddings, 
                                      model,
                                      batch_size,
                                      gt_mtx_path,
                                      device):
    '''
    This function calculates the linear CKA score between two representations.
    gene_emb_raw: dictionary of raw gene embeddings
    all_ptb_targets_with_embeddings: list of perturb targets with embeddings
    model: trained model object
    batch_size: batch size
    gt_mtx_path: path to ground truth embedding
    '''
    
    # Get learned latent embeddings -----------------------------------------------------
    rows_single = [{'gene_name': gene, 'gene_emb': gene_emb_raw[gene]} for gene in all_ptb_targets_with_embeddings]
    single_gene_embeddings = pd.DataFrame(rows_single, columns=['gene_name', 'gene_emb'])
    single_gene_embeddings['type'] = 'single'
    gene_latent_df = single_gene_embeddings
    assert gene_latent_df.shape[0] == len(all_ptb_targets_with_embeddings)

    model.eval()
    fine_tuned_embeddings = []
    for _, row in tqdm(gene_latent_df.iterrows(), total=len(gene_latent_df), desc='Get fine-tuned gene embeddings'):
        type = row['type']
        if type == 'single':
            gene_emb = torch.tensor(row['gene_emb'], dtype=torch.float64).to(device)
            gene_emb = gene_emb.repeat(batch_size, 1) # make batch_size gene_emb 
            with torch.no_grad():
                gene_emb_ft = model.c_encode(gene_emb) # Assuming c_encode expects a batch dimension
            fine_tuned_embeddings.append(gene_emb_ft[0].detach().cpu().numpy())
    gene_latent_df['gene_emb_ft'] = fine_tuned_embeddings

    # Get ground truth embeddings -----------------------------------------------------
    with open(gt_mtx_path + 'pert_list.pkl', 'rb') as f:
            pert_list = pickle.load(f)
    with open(gt_mtx_path + 'feat.pkl', 'rb') as f:
        feat = pickle.load(f)
    feat_df = pd.DataFrame(feat, index=pert_list)
    feat_df = feat_df.loc[gene_latent_df['gene_name'].values.tolist()]

    # Calculate CKA score -----------------------------------------------------
    gene_latent_df_array = np.stack(gene_latent_df['gene_emb_ft'].values)
    # alignment_score = linear_CKA(np.array(feat_df), gene_latent_df_array)
    alignment_score = kernel_CKA(np.array(feat_df), gene_latent_df_array, sigma=5)
    
    # Update gene_latent_df -----------------------------------------------------
    gene_latent_df['pseudo_bulk'] = feat_df.values.tolist()
    
    return alignment_score, gene_latent_df

def plot_umap_by_selections(round, opts, gene_latent_df, train_list_current, train_list_temp):
    # plot UMAP of the latent space color by selected, test_set, and non-selected
    pseudo_bulk_array_all = np.stack(gene_latent_df['pseudo_bulk'].values)
    reducer = umap.UMAP(random_state=opts.random_seed)
    umap_embedding = reducer.fit_transform(pseudo_bulk_array_all)
    
    # Create boolean masks for each group based on gene names
    gene_names = gene_latent_df['gene_name'].values
    train_list_current_set = set(train_list_current) - set(train_list_temp)
    test_set_set = set(opts.ptb_leave_out_list)
    selected_mask = np.array([gene in train_list_current_set for gene in gene_names])
    to_select_mask = np.array([gene in train_list_temp for gene in gene_names])
    test_set_mask = np.array([gene in test_set_set for gene in gene_names])
    non_selected_mask = ~(selected_mask | test_set_mask | to_select_mask)

    # Plot UMAP with different colors for each group
    plt.figure(figsize=(10, 7))
    plt.scatter(umap_embedding[non_selected_mask, 0], umap_embedding[non_selected_mask, 1], c='gray', label='Non-selected Samples', s=50, alpha=0.5)
    plt.scatter(umap_embedding[selected_mask, 0], umap_embedding[selected_mask, 1], c='blue', label='Selected Samples', s=100, edgecolors='black')
    plt.scatter(umap_embedding[test_set_mask, 0], umap_embedding[test_set_mask, 1], c='green', label='Test Set Samples', s=50, alpha=0.7)
    plt.scatter(umap_embedding[to_select_mask, 0], umap_embedding[to_select_mask, 1], c='red', label='To Select Samples', s=100, edgecolors='black')

    # Add labels and title
    plt.title('UMAP of Pseudo-bulk profiles with Sample Groups')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    # Save the plot
    if round == 0:
        plt.savefig(opts.warmup_model_path + '/umap_pseudo_bulk_sample_groups_round'+ str(round) +'.png')
    else:
        plt.savefig(opts.output_dir + '/umap_pseudo_bulk_sample_groups_round'+ str(round) +'.png')
    plt.show()

def get_predictions(dataloader, args,
                    last_model, best_train_model, best_val_model,
                    ptb_leave_out_list,
                    batch_size,
                    mmd_sigma,
                    device):
    
    # define mmd function
    mmd_loss_whole = MMD_loss(fix_sigma=mmd_sigma, kernel_num=10)

    # result dictionaries
    gt_y = []
    pred_y_last_model = []
    pred_y_best_train_model = []
    pred_y_best_val_model = []
    gt_c_1 = []
    gt_c_2 = []
    gt_c_1_2 = []
    gt_c_2_2 = []
    gt_C_y = []
    gt_x = []

    mmd_to_ctrl_last_model = {}
    mmd_to_ctrl_best_train_model = {}
    mmd_to_ctrl_best_val_model = {}

    last_model.eval()
    best_train_model.eval()
    if args.val_ratio > 0:
        best_val_model.eval()

    # Loop through inference dataloader
    for i, X in enumerate(tqdm(dataloader, desc='Get predictions')):
        x = X[0].to(device) #ctrl samples
        y = X[1] #ptb samples
        c_1 = X[2].to(device) #ptb target labels (target 1)
        c_2 = X[3].to(device) #ptb target labels (target 2)
        C_y = X[4] #ptb target names

        with torch.no_grad():
            y_hat_last_model, _, _ , _ = last_model(x, c_1, c_2)
            y_hat_best_train_model, _, _ , _ = best_train_model(x, c_1, c_2)
            if args.val_ratio > 0:
                y_hat_best_val_model, _, _ , _ = best_val_model(x, c_1, c_2)
        
        # calculate MMD distance to control for different models
        mmd_to_ctrl_last_model_temp = mmd_loss_whole(y_hat_last_model, x).item()
        mmd_to_ctrl_best_train_model_temp = mmd_loss_whole(y_hat_best_train_model, x).item()
        if args.val_ratio > 0:
            mmd_to_ctrl_best_val_model_temp = mmd_loss_whole(y_hat_best_val_model, x).item()
        ptb_name = C_y[0]
        
        if ptb_name not in mmd_to_ctrl_last_model.keys():
            mmd_to_ctrl_last_model[ptb_name] = mmd_to_ctrl_last_model_temp
        elif ptb_name in mmd_to_ctrl_last_model.keys():
            mmd_to_ctrl_last_model[ptb_name] = np.hstack((mmd_to_ctrl_last_model[ptb_name], mmd_to_ctrl_last_model_temp))

        if ptb_name not in mmd_to_ctrl_best_train_model.keys():
            mmd_to_ctrl_best_train_model[ptb_name] = mmd_to_ctrl_best_train_model_temp
        elif ptb_name in mmd_to_ctrl_best_train_model.keys():
            mmd_to_ctrl_best_train_model[ptb_name] = np.hstack((mmd_to_ctrl_best_train_model[ptb_name], mmd_to_ctrl_best_train_model_temp))
                                                                
        if args.val_ratio > 0:
            if ptb_name not in mmd_to_ctrl_best_val_model.keys():
                mmd_to_ctrl_best_val_model[ptb_name] = mmd_to_ctrl_best_val_model_temp
            elif ptb_name in mmd_to_ctrl_best_val_model.keys():
                mmd_to_ctrl_best_val_model[ptb_name] = np.hstack((mmd_to_ctrl_best_val_model[ptb_name], mmd_to_ctrl_best_val_model_temp))
        
        # stack other results
        gt_x.append(x.cpu().numpy())
        gt_y.append(y.numpy())
        gt_c_1.append(c_1.detach().cpu().numpy())
        gt_c_2.append(c_2.detach().cpu().numpy())
        gt_c_1_2.append(None)
        gt_c_2_2.append(None)
        pred_y_last_model.append(y_hat_last_model.cpu().numpy())
        pred_y_best_train_model.append(y_hat_best_train_model.cpu().numpy())
        if args.val_ratio > 0:
            pred_y_best_val_model.append(y_hat_best_val_model.cpu().numpy())
        else:
            pred_y_best_val_model.append(None)
        gt_C_y.append(np.array(C_y))

    # assert set(ptb_leave_out_list) == set(np.vstack(gt_C_y).flatten()), 'Error: ptb_leave_out_list does not match the ptb targets in the dataset'
    result_dic = {
        'gt_x': np.vstack(gt_x),
        'gt_y': np.vstack(gt_y),
        'pred_y_last_model': np.vstack(pred_y_last_model),
        'pred_y_best_train_model': np.vstack(pred_y_best_train_model),
        'pred_y_best_val_model': np.vstack(pred_y_best_val_model),
        'gt_c_1': np.vstack(gt_c_1),
        'gt_c_2': np.vstack(gt_c_2),
        'gt_c_1_2': np.vstack(gt_c_1_2),
        'gt_c_2_2': np.vstack(gt_c_2_2),
        'gt_C_y': np.vstack(gt_C_y)
    }

    C_y = result_dic['gt_C_y'].flatten().tolist()
    print("There are", len(set(C_y)), "unique perturbations in the test set.")
    if ptb_leave_out_list is not None:
        assert (len(set(C_y) - set(ptb_leave_out_list)) == 0), 'Error: ptb_leave_out_list does not match the ptb targets in the dataset'
        assert set(mmd_to_ctrl_last_model.keys()) == set(mmd_to_ctrl_best_train_model.keys()) == set(C_y), 'Error: mmd_to_ctrl_last_model and mmd_to_ctrl_best_train_model have different keys'
        assert set(mmd_to_ctrl_best_val_model.keys()) == set(C_y), 'Error: mmd_to_ctrl_best_val_model has different keys'
    
    pred_vae_last_model = {} 
    pred_vae_best_train_model = {}
    pred_vae_best_val_model = {}
    
    # Get predictions from VAE
    for i in range(len(C_y)//batch_size):
        c = C_y[i*batch_size]
        c_test = np.unique(result_dic['gt_C_y'][i]).item()
        assert c == c_test, 'The perturbation is not the same!'
        # Predictions from VAE
        if c not in pred_vae_last_model.keys():
            pred_vae_last_model[c] = result_dic['pred_y_last_model'][i*batch_size:(i+1)*batch_size]
        else:
            pred_vae_last_model[c] = np.vstack((pred_vae_last_model[c], result_dic['pred_y_last_model'][i*batch_size:(i+1)*batch_size]))
        if c not in pred_vae_best_train_model.keys():
            pred_vae_best_train_model[c] = result_dic['pred_y_best_train_model'][i*batch_size:(i+1)*batch_size]
        else:
            pred_vae_best_train_model[c] = np.vstack((pred_vae_best_train_model[c], result_dic['pred_y_best_train_model'][i*batch_size:(i+1)*batch_size]))
        if args.val_ratio > 0:
            if c not in pred_vae_best_val_model.keys():
                pred_vae_best_val_model[c] = result_dic['pred_y_best_val_model'][i*batch_size:(i+1)*batch_size]
            else:
                pred_vae_best_val_model[c] = np.vstack((pred_vae_best_val_model[c], result_dic['pred_y_best_val_model'][i*batch_size:(i+1)*batch_size]))

    del result_dic, pred_y_last_model, pred_y_best_train_model, pred_y_best_val_model
    gc.collect()
    torch.cuda.empty_cache()
    
    return pred_vae_last_model, pred_vae_best_train_model, pred_vae_best_val_model, C_y, mmd_to_ctrl_last_model, mmd_to_ctrl_best_train_model, mmd_to_ctrl_best_val_model

def get_mmd_to_ctrl_all(args, last_model, best_train_model, best_val_model, mmd_sigma, device):
    
    # define mmd function
    mmd_loss_whole = MMD_loss(fix_sigma=mmd_sigma, kernel_num=10)

    # result dictionaries
    mmd_to_ctrl_all_last_model = {}
    mmd_to_ctrl_all_best_train_model = {}
    mmd_to_ctrl_all_best_val_model = {}

    last_model.eval()
    best_train_model.eval()
    if args.val_ratio > 0:
        best_val_model.eval()

    # load in all dataloader
    all_dataloader_path = f'{args.base_dir}/results/'+args.dataset_name+'/all_dataloader.pkl'
    with open(all_dataloader_path, 'rb') as f:
        dataloader = pickle.load(f)

    # loop through all dataloader
    for i, X in enumerate(tqdm(dataloader, desc='Get predictions')):
        x = X[0].to(device)
        y = X[1]
        c_1 = X[2].to(device)
        c_2 = X[3].to(device)
        C_y = X[4]

        with torch.no_grad():
            y_hat_last_model, _, _ , _ = last_model(x, c_1, c_2)
            y_hat_best_train_model, _, _ , _ = best_train_model(x, c_1, c_2)
            if args.val_ratio > 0:
                y_hat_best_val_model, _, _ , _ = best_val_model(x, c_1, c_2)
        
        # calculate MMD distance to control for different models
        mmd_to_ctrl_last_model_temp = mmd_loss_whole(y_hat_last_model, x).item()
        mmd_to_ctrl_best_train_model_temp = mmd_loss_whole(y_hat_best_train_model, x).item()
        if args.val_ratio > 0:
            mmd_to_ctrl_best_val_model_temp = mmd_loss_whole(y_hat_best_val_model, x).item()
        ptb_name = C_y[0]
        
        if ptb_name not in mmd_to_ctrl_all_last_model.keys():
            mmd_to_ctrl_all_last_model[ptb_name] = mmd_to_ctrl_last_model_temp
        elif ptb_name in mmd_to_ctrl_all_last_model.keys():
            mmd_to_ctrl_all_last_model[ptb_name] = np.hstack((mmd_to_ctrl_all_last_model[ptb_name], mmd_to_ctrl_last_model_temp))

        if ptb_name not in mmd_to_ctrl_all_best_train_model.keys():
            mmd_to_ctrl_all_best_train_model[ptb_name] = mmd_to_ctrl_best_train_model_temp
        elif ptb_name in mmd_to_ctrl_all_best_train_model.keys():
            mmd_to_ctrl_all_best_train_model[ptb_name] = np.hstack((mmd_to_ctrl_all_best_train_model[ptb_name], mmd_to_ctrl_best_train_model_temp))
                                                                
        if args.val_ratio > 0:
            if ptb_name not in mmd_to_ctrl_all_best_val_model.keys():
                mmd_to_ctrl_all_best_val_model[ptb_name] = mmd_to_ctrl_best_val_model_temp
            elif ptb_name in mmd_to_ctrl_all_best_val_model.keys():
                mmd_to_ctrl_all_best_val_model[ptb_name] = np.hstack((mmd_to_ctrl_all_best_val_model[ptb_name], mmd_to_ctrl_best_val_model_temp))

    # quick checks
    assert set(mmd_to_ctrl_all_last_model.keys()) == set(dataloader.dataset.dataset.ptb_names), 'Error: did not get all perturbations'
    assert set(mmd_to_ctrl_all_last_model.keys()) == set(mmd_to_ctrl_all_best_train_model.keys()), 'Error: mmd_to_ctrl_all_last_model and mmd_to_ctrl_all_best_train_model have different keys'
    if args.val_ratio > 0:
        assert set(mmd_to_ctrl_all_best_val_model.keys()) == set(mmd_to_ctrl_all_best_train_model.keys()), 'Error: mmd_to_ctrl_all_best_val_model has different keys'

    del dataloader
    
    return mmd_to_ctrl_all_last_model, mmd_to_ctrl_all_best_train_model, mmd_to_ctrl_all_best_val_model

