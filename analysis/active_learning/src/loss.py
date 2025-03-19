import os
import sys
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'morph'))
sys.path.append(morph_path)

from utils import MMD_loss

# loss function definition
def loss_function(y_hat, y, x_recon, x, mu, logvar, 
                  MMD_sigma, kernel_num, 
                  gamma1=1, gamma2=0):

    # Compute MMD loss between predicted perturbed samples and true perturbed samples
    mmd_function_pred = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    if y_hat is None:
        pred_loss = 0
    else:
        pred_loss = mmd_function_pred(y_hat, y)
    
    # Compute reconstruction loss between reconstructed control samples and true control samples
    mmd_function_recon = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    if gamma1 > 0:
        recon_mmd = mmd_function_recon(x_recon, x)
    else:
        recon_mmd = 0
    if gamma2 > 0:
        recon_mse = nn.MSELoss()(x_recon, x)
    else:
        recon_mse = 0
    recon_loss = gamma1*recon_mmd + gamma2*recon_mse
    
    # Compute KL divergence
    if logvar is None:
        KLD = 0
    else:
        KLD = -0.5*torch.sum(logvar -(mu.pow(2)+logvar.exp())+1)/x.shape[0]
    
    return pred_loss, recon_loss, KLD