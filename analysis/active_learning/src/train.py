import torch
from warnings import filterwarnings
import sys
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle
from helpers import calculate_cka_for_representations
from acquisition import query_sample_to_label
import json
import wandb
from sklearn.metrics import r2_score, mean_squared_error as mse
import os
from loss import loss_function

filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'morph'))
sys.path.append(morph_path)

from model import MORPH

def train(
        opts,
        dataset,
        dataloader_train,
        dataloader_val,
        dataloader_infer,
        gene_emb_raw, gene_latent_df, gene_distance_df, 
        train_list_temp, 
        train_list_current,
        alpha_schedule,
        beta_schedule,
        log
):

    if log:
        wandb.init(project='active_learning'+'_'+ \
                   opts.dataset_name+'_'+ \
                    opts.leave_out_test_set_id + '_' + \
                    opts.train_style + '_' + \
                    str(opts.num_round)+'_round_'+str(opts.num_ptb)+'_ptb_' + str(opts.initial_num_ptb) + '_initial_ptb',
                    name=opts.output_dir.split('/')[-1] + '_seed_' + str(opts.random_seed))
    
    device = opts.device

    # Get the initial gene embeddings for all targets
    rows_single = [{'gene_name': gene, 'gene_emb': gene_emb_raw[gene]} for gene in opts.all_ptb_targets_with_embeddings]
    single_gene_embeddings = pd.DataFrame(rows_single, columns=['gene_name', 'gene_emb'])
    single_gene_embeddings['type'] = 'single'
    gene_target_emb_raw_df = single_gene_embeddings
    assert gene_target_emb_raw_df.shape[0] == len(opts.all_ptb_targets_with_embeddings)

    # 4.0 Initial labelled list -----------------------------------------------------
    train_list_record = train_list_current.copy()
    alignment_score_record = []
    rounds = [i for i in range(1, opts.num_round)]

    # 4.1 Load trained warm-up model ---------------------------------------
    if opts.train_style == 'tune_on_whole':
        model_1_path = opts.output_dir + '/round_0_best_train_model.pt'
        if os.path.exists(model_1_path):
            print('Model 1 already trained, loaded from:', model_1_path)
            mvae = torch.load(model_1_path, map_location=device)
        else:
            raise ValueError('Model 1 not trained yet')
    elif opts.train_style == 'learn_from_scratch':
        mvae = MORPH(
            dim = opts.dim,
            c_dim = opts.cdim,
            opts = opts,
            device = device
            )
        mvae.double()
        mvae.to(device)
        print('learning from scratch on the whole dataset')
    
    # optimizer
    optimizer = torch.optim.Adam(params=mvae.parameters(), lr=opts.lr)
    print('Learning rate: ', opts.lr)
    mvae.train()
    
    # 4.2 Start training -----------------------------------------------------
    print('Start training...')
    for round in rounds:
        print('Round:', round)
        start_epoch = opts.training_schedule[round]
        end_epoch = opts.training_schedule[round+1]
        print('Start epoch:', start_epoch)
        print('End epoch:', end_epoch)

        # reset for each round
        min_train_loss = np.inf
        min_val_loss = np.inf
        patience = 0

        if log:
            wandb.log({'round': round})

        # Training loop for each round
        for n in range(start_epoch, end_epoch):
            lossAv = 0
            ct = 0
            mmdAv = 0
            reconAv = 0
            klAv = 0

            for i, X in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
                x = X[0].to(device) #ctrl samples
                y = X[1].to(device) #ptb samples
                c_1 = X[2].to(device) #ptb target labels (target 1)
                c_2 = X[3].to(device) #ptb target labels (target 2)

                optimizer.zero_grad()
                y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2)
                mmd_loss, recon_loss, kl_loss = loss_function(y_hat=y_hat, y=y,
                                                              x_recon=x_recon, x=x, 
                                                              mu=z_mu, logvar=z_logvar, 
                                                              MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num,
                                                              gamma1=1, gamma2=0)
                loss = alpha_schedule[n] * mmd_loss + recon_loss + beta_schedule[n]*kl_loss
                loss.backward()

                if opts.grad_clip:
                    for param in mvae.parameters():
                        if param.grad is not None:
                            param.grad.data = param.grad.data.clamp(min=-0.5, max=0.5)
                optimizer.step()

                ct += 1
                lossAv += loss.detach().cpu().numpy()
                mmdAv += mmd_loss.detach().cpu().numpy()
                reconAv += recon_loss.detach().cpu().numpy()
                if z_logvar is not None:
                    klAv += kl_loss.detach().cpu().numpy()
                if log:
                    wandb.log({'loss': loss})
                    wandb.log({'mmd_loss': mmd_loss})
                    wandb.log({'recon_loss': recon_loss})
                    wandb.log({'kl_loss': kl_loss})

            if log:
                wandb.log({'epoch avg loss': lossAv/ct})
                wandb.log({'epoch avg mmd loss': mmdAv/ct})
                wandb.log({'epoch avg recon loss': reconAv/ct})
                wandb.log({'epoch avg kl loss': klAv/ct})
                wandb.log({'Epoch:': n})

            if (mmdAv + reconAv + klAv)/ct < min_train_loss:
                min_train_loss = (mmdAv + reconAv + klAv)/ct 
                best_model = deepcopy(mvae)
                best_model_path_temp = opts.output_dir+'/round_'+str(round)+'_best_train_model.pt'
                torch.save(best_model, best_model_path_temp)
                print('saved best train model for round:', round, 'to', best_model_path_temp)
                if log:
                    wandb.log({'min_train_loss': min_train_loss})
                    wandb.log({'min_train_epoch': n})

            # Validation -----------------------------------------------------
            if opts.val_ratio > 0:
                mvae.eval()
                val_mmdAv = 0
                val_reconAv = 0
                val_klAv = 0
                val_ct = 0

                for (i, X) in tqdm(enumerate(dataloader_val), total=len(dataloader_val)):
                    x = X[0].to(device)
                    y = X[1].to(device)
                    c_1 = X[2].to(device)
                    c_2 = X[3].to(device)
                    ptb_target = X[4]

                    with torch.no_grad(): 
                        y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2)

                    mmd_loss, recon_loss, kl_loss = loss_function(y_hat=y_hat, y=y,
                                                                x_recon=x_recon, x=x, 
                                                                mu=z_mu, logvar=z_logvar, 
                                                                MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num,
                                                                gamma1=1, gamma2=0)
                    val_ct += 1
                    val_mmdAv += mmd_loss.detach().cpu().numpy()
                    val_reconAv += recon_loss.detach().cpu().numpy()
                    if z_logvar is not None:
                        val_klAv += kl_loss.detach().cpu().numpy()

                    if log:
                        wandb.log({'val_mmd': mmd_loss})
                        wandb.log({'val_recon': recon_loss})
                        wandb.log({'val_kl': kl_loss})

                if log:
                    wandb.log({'epoch avg val_mmd': val_mmdAv / val_ct})
                    wandb.log({'epoch avg val_recon': val_reconAv / val_ct})
                    wandb.log({'epoch avg val_kl': val_klAv / val_ct})

                if (val_mmdAv + val_reconAv + val_klAv)/val_ct < min_val_loss:
                    min_val_loss = (val_mmdAv + val_reconAv + val_klAv)/val_ct
                    best_model_val = deepcopy(mvae)
                    best_val_model_path_temp = opts.output_dir+'/round_'+str(round)+'_best_val_model.pt'
                    torch.save(best_model_val, best_val_model_path_temp)
                    print('saved best val model for round:', round, 'to', best_val_model_path_temp)
                    patience = 0
                    if log:
                        wandb.log({'min_val_loss': min_val_loss})
                        wandb.log({'min_val_epoch': n})
                else:
                    patience += 1

                if log:
                    wandb.log({'patience': patience})

                if patience >= opts.tolerance_epochs:
                    print('Early stopping at epoch:', n)
                    if log:
                        wandb.log({'early_stopping_epoch': n})
                    break

            # Get learning embeddings for certain acquisition methods ---------------------
            if opts.method in ['kmeans_center_weight_uncertainty', 
                               'kmeans_center_weight_uncertainty_min_max', 
                               'kmeans_center_weight_uncertainty_softmax',
                               'kmeans_center_weight_uncertainty_scaled',
                               'uncertainty', 'uncertainty_random']:
                mvae.eval()
                fine_tuned_embeddings = []
                for _, row in tqdm(gene_target_emb_raw_df.iterrows(), total=len(gene_target_emb_raw_df), desc='Get fine-tuned gene embeddings'):
                    type = row['type']
                    if type == 'single':
                        gene_emb = torch.tensor(row['gene_emb'], dtype=torch.float64).to(device)
                        gene_emb = gene_emb.repeat(opts.batch_size, 1) # make batch_size gene_emb 
                        with torch.no_grad():
                            gene_emb_ft = mvae.c_encode(gene_emb) # Assuming c_encode expects a batch dimension
                        fine_tuned_embeddings.append(gene_emb_ft[0].detach().cpu().numpy())
                gene_distance_df['gene_emb_ft'] = fine_tuned_embeddings
                if n > start_epoch:
                    # get distance between gene_emb_old and gene_emb_ft for each gene
                    gene_distance_df['distance_temp'] = np.linalg.norm(np.vstack(gene_distance_df['gene_emb_ft'].values) - np.vstack(gene_distance_df['gene_emb_old'].values), axis=1)
                    assert gene_distance_df['distance_temp'].sum() > 0, 'distance_temp should be greater than 0'
                    gene_distance_df['distance'] = gene_distance_df['distance'] + gene_distance_df['distance_temp']
                    gene_distance_df['epoch'] = gene_distance_df['epoch'] + 1
                    if log:
                        # log the distance distribution
                        min_value = gene_distance_df['distance_temp'].min()
                        max_value = gene_distance_df['distance_temp'].max()
                        wandb.log({"distance_temp_min": min_value, "distance_temp_max": max_value})
                else:
                    gene_distance_df['distance'] = 0
                    gene_distance_df['distance_temp'] = 0
                    gene_distance_df['epoch'] = 0
                # set new to old
                gene_distance_df['gene_emb_old'] = gene_distance_df['gene_emb_ft']
                
        # save the last model
        model_save_temp = deepcopy(mvae)
        model_path_temp = opts.output_dir+'/round_'+str(round)+'_last_model.pt'
        torch.save(model_save_temp, model_path_temp)
        print('saved model for round:', round, 'to', model_path_temp)
        
        # Let's also record the matrix alignment scores with groundtruth kernel
        gt_mtx_path = f"{opts.base_dir}/kernels/{opts.dataset_name}/ground_truth_delta/"
        alignment_score, gene_latent_df = calculate_cka_for_representations(gene_emb_raw, opts.all_ptb_targets_with_embeddings, 
                                                                         best_model, opts.batch_size, gt_mtx_path, device)
        alignment_score_record.append(alignment_score)
        
        # if last round, then no need to query more samples, skip to the end
        if round == rounds[-1]:
            break
        
        # query samples to label
        del dataloader_train, dataloader_val
        train_list_temp, train_list_current, \
            dataloader_train, dataloader_val = query_sample_to_label(opts, dataset,
                                                                     train_list_current, gene_latent_df, gene_distance_df, 
                                                                     round = round+1, log=log)
        train_list_record.append(train_list_temp)

        if opts.train_style == 'tune_on_whole':
            mvae.train()
        elif opts.train_style == 'learn_from_scratch':
            mvae = MORPH(
                dim = opts.dim,
                c_dim = opts.cdim,
                opts = opts,
                device = device
                )
            mvae.double()
            mvae.to(device)
            optimizer = torch.optim.Adam(params=mvae.parameters(), lr=opts.lr)
            mvae.train()
            print('learning from scratch on the whole dataset')
    
    ### 5. Save the training record -----------------------------------------------------
    print(len(train_list_current))
    assert len(train_list_current) == opts.total_samples
    
    # confirm that none of train_list_current overlap with ptb_leave_out_list
    if opts.method != 'test_set':
        assert len(set(train_list_current) & set(opts.ptb_leave_out_list)) == 0
    else:
        assert set(train_list_current) - set(opts.ptb_leave_out_list) == set()

    opts.train_list = train_list_record
    with open(opts.output_dir+'/config.json', 'w') as f:
        json.dump(opts.__dict__, f, indent=4)
    if len(alignment_score_record) > 0:
        with open(opts.output_dir+'/alignment_score_last_model.json', 'w') as f:
            json.dump(alignment_score_record, f)




        


