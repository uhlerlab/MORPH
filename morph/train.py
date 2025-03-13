import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import wandb
from copy import deepcopy
import numpy as np
import os
import sys
from model import *
from utils import MMD_loss
import time
from sklearn.metrics import r2_score, mean_squared_error as mse
import torch.optim.lr_scheduler as lr_scheduler
import json

# loss function definition
def loss_function(y_hat, y, y_hat_marker, y_marker, x_recon, x, mu, logvar, MMD_sigma, kernel_num, 
                  gamma1=0, gamma2=0, reconstruction_loss='mse'):

    # Compute prediction loss
    mmd_function_pred = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    if y_hat is None:
        pred_loss = 0
    else:
        pred_mmd = mmd_function_pred(y_hat, y)
        # compute MSE loss between mean of y_hat and mean of y
        y_hat_mean = y_hat.mean(0)
        y_mean = y.mean(0)
        pred_mse = nn.MSELoss()(y_hat_mean, y_mean)
        if reconstruction_loss == 'mmd_pred_mmd_mse' or reconstruction_loss == 'mmd_mse_both':
            pred_loss = (1-gamma2)*pred_mmd + gamma2*pred_mse
        else:
            pred_loss = pred_mmd
    
    # Compute marker prediction loss
    matching_function_interv_marker = MMD_loss(fix_sigma=200, kernel_num=10) # MMD Distance since we don't have paired data
    if y_hat_marker is None:
        MMD_marker = 0
    else:
        MMD_marker = matching_function_interv_marker(y_hat_marker, y_marker)
    
    # Compute reconstruction loss
    mmd_function_recon = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    if reconstruction_loss == 'mse':
        recon_mse = nn.MSELoss()(x_recon, x)
        recon_mmd = 0
        assert gamma1 == 1, "gamma1 should be 1 for reconstruction loss = 'mse'"
    elif reconstruction_loss == 'mmd':
        recon_mse = 0
        recon_mmd = mmd_function_recon(x_recon, x)
        assert gamma1 == 0, "gamma1 should be 0 for reconstruction loss = 'mmd'"
    elif reconstruction_loss == 'mmd_mse_both':
        recon_mse = nn.MSELoss()(x_recon, x)
        if gamma1 == 1:
            recon_mmd = 0 # avoid computing MMD to save time
        else:
            recon_mmd = mmd_function_recon(x_recon, x)
    recon_loss = (1-gamma1)*recon_mmd + gamma1*recon_mse
    
    # Compute KL divergence
    if logvar is None:
        KLD = 0
    else:
        KLD = -0.5*torch.sum(logvar -(mu.pow(2)+logvar.exp())+1)/x.shape[0]
    
    return pred_loss, recon_loss, KLD, MMD_marker, pred_mmd, pred_mse, recon_mmd, recon_mse


# fit VAE with MMD loss to data
def train_validate(
    dataloader,
    dataloader_infer,
    dataloader_val,
    opts,
    device,
    savedir,
    model,
    log
    ):

    if log:
        if opts.modality == 'rna':
            dataset_name = opts.dataset+'_hvg' if opts.use_hvg == 'True' else opts.dataset
            project_name = f'morph_{dataset_name}_{opts.leave_out_test_set_id}'
        elif opts.modality == 'ops':
            project_name = f'morph_{opts.leave_out_test_set_id}'
        wandb.init(project=project_name, name=savedir.split('/')[-1])  #name should be the run time after fixing the os.makedirs bug
    
    if model == 'MORPH':
        mvae = MORPH(
            dim = opts.dim,
            c_dim = opts.cdim,
            opts = opts,
            device = device
        )
    
    # move model to device
    mvae.double()
    mvae.to(device)
    optimizer = torch.optim.Adam(params=mvae.parameters(), lr=opts.lr)
    mvae.train()
    print("Training for maximum {} epochs...".format(str(opts.epochs)))
    
    ## Loss parameters
    beta_schedule = torch.zeros(opts.epochs) # weight on the KLD
    if opts.modality == 'rna':
        if opts.tolerance_epochs <= 10:
            beta_schedule[:2] = 0
            beta_schedule[2:] = torch.linspace(0,opts.mxBeta,opts.epochs-2) 
        else:
            beta_schedule[:10] = 0
            beta_schedule[10:] = torch.linspace(0,opts.mxBeta,opts.epochs-10)
    else:
        beta_schedule[:2] = 0
        beta_schedule[2:] = torch.linspace(0,opts.mxBeta,opts.epochs-2)
    
    if 'gwps' in opts.dataset:
        alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
        alpha_schedule[:] = opts.mxAlpha
        alpha_schedule[:1] = 0
        alpha_schedule[1:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-1) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha
    else:
        alpha_schedule = torch.zeros(opts.epochs) # weight on the MMD
        alpha_schedule[:] = opts.mxAlpha
        alpha_schedule[:5] = 0
        alpha_schedule[5:int(opts.epochs/2)] = torch.linspace(0,opts.mxAlpha,int(opts.epochs/2)-5) 
        alpha_schedule[int(opts.epochs/2):] = opts.mxAlpha

    # Save as JSON
    config_filename = os.path.join(savedir, 'alpha_schedule.json')
    with open(config_filename, "w") as json_file:
        json.dump({"alpha_schedule": alpha_schedule.tolist()}, json_file, indent=4)
    print(f"Alpha schedule saved to {config_filename}") 
    
    min_train_loss = np.inf
    min_train_loss_2 = np.inf
    best_model = deepcopy(mvae)
    best_model_2 = deepcopy(mvae)
    min_val_loss = np.inf
    min_val_loss_2 = np.inf
    best_model_val = deepcopy(mvae)
    best_model_val_2 = deepcopy(mvae)

    # Read in the DEG file
    if opts.modality == 'rna':
        deg_file = pd.read_csv('/home/che/perturb-project/git/gene_ptb_prediction/deg_file_path.csv')
        deg_path = deg_file[deg_file['dataset'] == opts.dataset][deg_file['use_hvg'] == (opts.use_hvg == 'True')]['file_path'].values[0]
        with open(deg_path, 'rb') as f:
            de_50_genes_dict = pickle.load(f)

    # Define tolenrance and patience for early stopping
    tolerance = opts.tolerance_epochs
    patience = 0

    # Start timing
    start_time = time.time()
    for epoch in range(0, opts.epochs):
        lossAv = 0
        ct = 0
        mmdAv = 0
        reconAv = 0
        klAv = 0
        pred_mmdAv = 0
        pred_mseAv = 0
        recon_mmdAv = 0
        recon_mseAv = 0

        # train
        for (i, X) in tqdm(enumerate(dataloader)):
            x = X[0] #control samples
            y = X[1] #perturbation samples
            c_1 = X[2] #perturbation labels (target 1)
            c_2 = X[3] #perturbation labels (target 2)
            if 'moe' in model:
                c_1_2 = X[5] #perturbation labels (target 1)
                c_2_2 = X[6] #perturbation labels (target 2)
                if '3expert' in model:
                    c_1_3 = X[7]
                    c_2_3 = X[8]
            
            if mvae.cuda:
                x = x.to(device)
                y = y.to(device)
                c_1 = c_1.to(device)
                c_2 = c_2.to(device)
                if 'moe' in model:
                    c_1_2 = c_1_2.to(device)
                    c_2_2 = c_2_2.to(device)
                    if '3expert' in model:
                        c_1_3 = c_1_3.to(device)
                        c_2_3 = c_2_3.to(device)
                
            optimizer.zero_grad()

            if 'moe' in model:
                if '3expert' in model:
                    y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2, c_1_2, c_2_2, c_1_3, c_2_3, num_interv=opts.num_interv)
                else:
                    y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2, c_1_2, c_2_2, num_interv=opts.num_interv)
            else:
                y_hat, x_recon, z_mu, z_logvar = mvae(x,c_1, c_2, num_interv=opts.num_interv)

            mmd_loss, recon_loss, kl_loss, _, \
                pred_mmd, pred_mse, recon_mmd, recon_mse = loss_function(y_hat, y, None, None, x_recon, x, 
                                                                         z_mu, z_logvar, opts.MMD_sigma, opts.kernel_num, 
                                                                         opts.Gamma1, opts.Gamma2, opts.reconstruction_loss)
            loss = alpha_schedule[epoch] * mmd_loss + recon_loss + beta_schedule[epoch]*kl_loss

            if(recon_loss.isnan()):
                print('recon_loss: ',recon_loss)
                print('y_hat: ',y_hat)
                print('x_recon: ',x_recon)
                print('x: ',x)
                sys.exit()

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
            pred_mmdAv += pred_mmd.detach().cpu().numpy() if pred_mmd != 0 else 0
            pred_mseAv += pred_mse.detach().cpu().numpy() if pred_mse != 0 else 0
            recon_mmdAv += recon_mmd.detach().cpu().numpy() if recon_mmd != 0 else 0
            recon_mseAv += recon_mse.detach().cpu().numpy() if recon_mse != 0 else 0
            if z_logvar is not None:
                klAv += kl_loss.detach().cpu().numpy()
            else:
                klAv = 0

            if log:
                wandb.log({'loss':loss})
                wandb.log({'mmd_loss':mmd_loss})
                wandb.log({'recon_loss':recon_loss})
                wandb.log({'kl_loss':kl_loss})
                wandb.log({'pred_mmd':pred_mmd})
                wandb.log({'pred_mse':pred_mse})
                wandb.log({'recon_mmd':recon_mmd})
                wandb.log({'recon_mse':recon_mse})

        print('Epoch '+str(epoch)+': Loss='+str(lossAv/ct)+', '+'MMD='+str(mmdAv/ct)+', '+'MSE='+str(reconAv/ct)+', '+'KL='+str(klAv/ct))
        
        if log:
            wandb.log({'epoch avg loss': lossAv/ct})
            wandb.log({'epoch avg mmd_loss': mmdAv/ct})
            wandb.log({'epoch avg recon_loss': reconAv/ct})
            wandb.log({'epoch avg kl_loss': klAv/ct})
            wandb.log({'epoch avg pred_mmd': pred_mmdAv/ct})
            wandb.log({'epoch avg pred_mse': pred_mseAv/ct})
            wandb.log({'epoch avg recon_mmd': recon_mmdAv/ct})
            wandb.log({'epoch avg recon_mse': recon_mseAv/ct})
            wandb.log({'alpha': alpha_schedule[epoch]})
            wandb.log({'beta': beta_schedule[epoch]})
            wandb.log({'Gamma1': opts.Gamma1})
            wandb.log({'Gamma2': opts.Gamma2})
            wandb.log({'Epoch:': epoch})
        
        if (mmdAv + reconAv + klAv)/ct < min_train_loss:
            min_train_loss = (mmdAv + reconAv + klAv)/ct 
            best_model = deepcopy(mvae)
            torch.save(best_model, os.path.join(savedir, 'best_model.pt'))
            if log:
                wandb.log({'min_train_loss': min_train_loss})
                wandb.log({'min_train_epoch': epoch})

        if (mmdAv + reconAv)/ct < min_train_loss_2:
            min_train_loss_2 = (mmdAv + reconAv)/ct 
            best_model_2 = deepcopy(mvae)
            torch.save(best_model_2, os.path.join(savedir, 'best_model_2.pt'))
            if log:
                wandb.log({'min_train_loss_2': min_train_loss_2})
                wandb.log({'min_train_epoch_2': epoch})
        
        # Validation loop (validation - for early stopping and save best model)
        mvae.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            val_lossAv = 0
            val_mmdAv = 0
            val_reconAv = 0
            val_klAv = 0
            val_ct = 0
            val_pred_mmdAv = 0
            val_pred_mseAv = 0
            val_recon_mmdAv = 0
            val_recon_mseAv = 0
            val_mmdMarkerAv = 0
            val_r2MarkerAv = 0
            val_rmseMarkerAv = 0
            
            for (i, X) in enumerate(dataloader_val):
                x = X[0]  # control samples
                y = X[1]  # perturbation samples
                c_1 = X[2]  # perturbation labels (target 1)
                c_2 = X[3]  # perturbation labels (target 2)
                ptb_target = X[4] #name of perturbation target

                if 'moe' in model:
                    c_1_2 = X[5]
                    c_2_2 = X[6]
                    if '3expert' in model:
                        c_1_3 = X[7]
                        c_2_3 = X[8]

                if mvae.cuda:
                    x = x.to(device)
                    y = y.to(device)
                    c_1 = c_1.to(device)
                    c_2 = c_2.to(device)
                    if 'moe' in model:
                        c_1_2 = c_1_2.to(device)
                        c_2_2 = c_2_2.to(device)
                        if '3expert' in model:
                            c_1_3 = c_1_3.to(device)
                            c_2_3 = c_2_3.to(device)

                if 'moe' in model:
                    if '3expert' in model:
                        y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2, c_1_2, c_2_2, c_1_3, c_2_3, num_interv=opts.num_interv)
                    else:
                        y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2, c_1_2, c_2_2, num_interv=opts.num_interv)
                else:
                    y_hat, x_recon, z_mu, z_logvar = mvae(x, c_1, c_2, num_interv=opts.num_interv)
                
                mmd_loss, recon_loss, kl_loss, _, \
                    pred_mmd, pred_mse, recon_mmd, recon_mse = loss_function(y_hat, y, None, None, x_recon, x, z_mu, z_logvar,
                                                                             opts.MMD_sigma, opts.kernel_num, 
                                                                             opts.Gamma1, opts.Gamma2, opts.reconstruction_loss)
                if z_logvar is not None:
                    val_loss = mmd_loss + recon_loss + kl_loss
                else:
                    val_loss = mmd_loss + recon_loss
                
                if opts.modality == 'rna':
                    ptb_target = list(set(ptb_target))[0]
                    degs = de_50_genes_dict[ptb_target]
                    y_true_deg = y[:, degs]
                    y_pred_deg = y_hat[:, degs]
                    mmd_loss_marker =  MMD_loss(fix_sigma=200, kernel_num=10)
                    val_mmd_marker = mmd_loss_marker(y_pred_deg, y_true_deg).item()
                    y_true_deg = y_true_deg.detach().cpu().numpy()
                    y_pred_deg = y_pred_deg.detach().cpu().numpy()
                    val_rmse_marker = np.sqrt(mse(y_true_deg.mean(0), y_pred_deg.mean(0)))
                    val_r2_marker = max(r2_score(y_true_deg.mean(0), y_pred_deg.mean(0)),0)

                val_ct += 1
                val_lossAv += val_loss.detach().cpu().numpy()
                val_mmdAv += mmd_loss.detach().cpu().numpy()
                val_reconAv += recon_loss.detach().cpu().numpy()
                val_pred_mmdAv += pred_mmd.detach().cpu().numpy() if pred_mmd != 0 else 0
                val_pred_mseAv += pred_mse.detach().cpu().numpy() if pred_mse != 0 else 0
                val_recon_mmdAv += recon_mmd.detach().cpu().numpy() if recon_mmd != 0 else 0
                val_recon_mseAv += recon_mse.detach().cpu().numpy() if recon_mse != 0 else 0
                if opts.modality == 'rna':
                    val_mmdMarkerAv += val_mmd_marker
                    val_r2MarkerAv += val_r2_marker
                    val_rmseMarkerAv += val_rmse_marker
                if z_logvar is not None:
                    val_klAv += kl_loss.detach().cpu().numpy()
                else:
                    val_klAv = 0

            # Log validation results
            print('Validation - Epoch ' + str(epoch) + ': Loss=' + str(val_lossAv / val_ct) + ', MMD=' + str(val_mmdAv / val_ct) + ', MSE=' + str(val_reconAv / val_ct) + ', KL=' + str(val_klAv / val_ct))

            if log:
                wandb.log({'epoch avg val_loss': val_lossAv / val_ct})
                wandb.log({'epoch avg val_mmd_loss': val_mmdAv / val_ct})
                wandb.log({'epoch avg val_recon_loss': val_reconAv / val_ct})
                wandb.log({'epoch avg val_kl_loss': val_klAv / val_ct})
                wandb.log({'epoch avg val_pred_mmd': val_pred_mmdAv / val_ct})
                wandb.log({'epoch avg val_pred_mse': val_pred_mseAv / val_ct})
                wandb.log({'epoch avg val_recon_mmd': val_recon_mmdAv / val_ct})
                wandb.log({'epoch avg val_recon_mse': val_recon_mseAv / val_ct})
                if opts.modality == 'rna':
                    wandb.log({'epoch avg val_mmd_marker': val_mmdMarkerAv / val_ct})
                    wandb.log({'epoch avg val_r2_marker': val_r2MarkerAv / val_ct})
                    wandb.log({'epoch avg val_rmse_marker': val_rmseMarkerAv / val_ct})
            
            # reconstruction + MMD + KL
            if (val_mmdAv + val_reconAv + val_klAv)/val_ct < min_val_loss:
                min_val_loss = (val_mmdAv + val_reconAv + val_klAv)/val_ct 
                best_model_val = deepcopy(mvae)
                torch.save(best_model_val, os.path.join(savedir, 'best_model_val.pt'))
                if opts.modality == 'rna':
                    patience = 0  # Reset patience since we have a new minimum validation loss
                if log:
                    wandb.log({'min_val_loss': min_val_loss})
                    wandb.log({'min_val_epoch': epoch})
            else:
                if opts.modality == 'rna':
                    patience += 1
            
            # reconstruction + MMD
            if (val_mmdAv + val_reconAv)/val_ct < min_val_loss_2:
                min_val_loss_2 = (val_mmdAv + val_reconAv)/val_ct 
                best_model_val_2 = deepcopy(mvae)
                torch.save(best_model_val_2, os.path.join(savedir, 'best_model_val_2.pt'))
                if opts.modality == 'ops':
                    patience = 0
                if log:
                    wandb.log({'min_val_loss_2': min_val_loss_2})
                    wandb.log({'min_val_epoch_2': epoch})
            else:
                if opts.modality == 'ops':
                    patience += 1
            
            if log:
                wandb.log({'patience': patience})

            # Early stopping check
            if patience >= tolerance:
                print(f"Early stopping on epoch {epoch}. No improvement in validation loss for {tolerance} rounds.")
                if log:
                    wandb.log({'early_stopping_epoch': epoch})
                break
        
        mvae.train()  # Set the model back to training mode
    
    # Calculate the total duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training completed in {total_time} seconds.")
    if log:
        wandb.log({'total_time': total_time})
    last_model = deepcopy(mvae)
    torch.save(last_model, os.path.join(savedir, 'last_model.pt'))
    