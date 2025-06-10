import numpy as np
import pandas as pd
import pickle
import torch
import sys
import matplotlib.pyplot as plt
from matplotlib import cm,colors
from warnings import filterwarnings
from tqdm import tqdm
import seaborn as sns
import json
import wandb
import os

import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pytorch_lightning as pl

filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', 'morph'))
morph_main_path = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(morph_path)

from inference import *
from utils import SCDATA_sampler, MMD_loss, split_scdata
from dataset import SCDataset
from torch.utils.data import DataLoader

# set random seed
torch.set_num_threads(8)
rand_seed = 9
pl.seed_everything(rand_seed)
log = True

# Update here --------------------------------------------------------
test_dataset = 'norman_k562_hvg' # put your dataset id here
leave_out_test_set_id = 'random_fold_1' # put your test set id here
trained_model = 'replogle_gwps_trained_model_small' # use the small one by default, change to the large one by setting this to 'replogle_gwps_trained_model_large'
# ---------------------------------------------------------------

# hyperparameters (optional to change)
train_epoch = 50
mxBeta=2
mxAlpha=2
print('train_epoch:', train_epoch)
print('test_dataset:', test_dataset)
print('leave_out_test_set_id:', leave_out_test_set_id)

device = 'cuda:5'
print('device:', device)
savedir = f'{morph_main_path}/transfer_learning/{trained_model}'
print('savedir:', savedir)
model_name = 'model.pt'
print('model_name:', model_name)

# 1. Load in single-cell data ------------------------------------------------
scdata_file = pd.read_csv(f'{morph_main_path}/data/scdata_file_path.csv')
adata_path = scdata_file[scdata_file['dataset'] == test_dataset]['file_path'].values[0]
adata_test = sc.read_h5ad(adata_path)
print('loaded adata_test:', adata_path)
print('adata_test.shape:', adata_test.shape)

# Get top 50 DE genes
sc.tl.rank_genes_groups(adata_test, groupby='gene', reference='non-targeting', n_genes=50, use_raw=False, rankby_abs=True)

# 2. Load trained model -------------------------------------------------------
model = torch.load(f'{savedir}/{model_name}', map_location=device) if savedir is not None else None
print('loaded model from:', f'{savedir}/{model_name}')
with open(f'{savedir}/config.json', 'r') as f:
    config = json.load(f)

print('batch size:', config['batch_size'])
print('model_type:', config['model'])
print('label_1:', config['label'])
print('label_2:', config['label_2'])
print('label_3:', config['label_3'])
print('lr:', config['lr'])
print('mmd_sigma:', config['MMD_sigma'])
print('latdim_ctrl:',config['latdim_ctrl'])
print('latdim_ptb:',config['latdim_ptb'])
if 'hidden_decoder_dim' in config:
    print('hidden_decoder_dim:',config['hidden_decoder_dim'])
if 'kaiming_init' in config:
    print('kaiming_init:',config['kaiming_init'])
print('training epochs:', config['epochs'])
print('trained on data:', config['dataset_name'])

# 3. Create dataloader test cell line -----------------------------------------
split_path = f'{morph_main_path}/data/{test_dataset}_splits.csv'
split_df = pd.read_csv(split_path)
test_set_list = split_df[split_df['test_set_id'] == leave_out_test_set_id]['test_set'].apply(lambda x: x.split(',')).values[0]
print('len of test_set_list', len(test_set_list))

all_perturbations = adata_test.obs['gene'].unique().tolist()
all_perturbations.remove('non-targeting')
train_set_list = [x for x in all_perturbations if x not in test_set_list]
print('len of train_set_list', len(train_set_list))
print('len of all_perturbations', len(all_perturbations))

assert len(train_set_list) > 0, "This script is for few-shot learning, so the training set should not be empty."

output_dir = f"{savedir}/{test_dataset}/{leave_out_test_set_id}/len_train_set_{len(train_set_list)}/epochs_{train_epoch}_rand_seed_{rand_seed}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset = SCDataset(base_dir = os.path.abspath(os.path.join(current_dir, '..', '..')),
                    dataset_name = test_dataset, 
                    adata_path = adata_path,
                    leave_out_test_set = test_set_list,
                    representation_type=config['label'], 
                    representation_type_2 = config['label_2'],
                    representation_type_3=config['label_3'],
                    min_counts=config['batch_size'],
                    random_seed=config['seed'])

train_idx = np.where(np.isin(dataset.ptb_names, train_set_list))[0].tolist()
print('len(train_idx)', len(train_idx))
infer_idx = np.where(np.isin(dataset.ptb_names, test_set_list))[0].tolist()
print('len(infer_idx)', len(infer_idx))

# Inference dataloader
from torch.utils.data import DataLoader, Subset
dataset_train = Subset(dataset, train_idx)
ptb_name = dataset.ptb_names[train_idx]
dataloader_train = DataLoader(
    dataset_train,
    batch_sampler=SCDATA_sampler(dataset_train, config['batch_size'], ptb_name),
    num_workers = 0
)

dataset_infer = Subset(dataset, infer_idx)
ptb_name = dataset.ptb_names[infer_idx]
dataloader_infer = DataLoader(
    dataset_infer,
    batch_sampler=SCDATA_sampler(dataset_infer, config['batch_size'], ptb_name),
    num_workers = 0
)

# 4. Fine-tune model on test cell line ----------------------------------------
import torch
import torch.nn as nn
print('model_name:', model_name)
model_ft = torch.load(f'{savedir}/{model_name}', map_location=device) if savedir is not None else None

# loss function definition
def loss_function(y_hat, y, y_hat_marker, y_marker, x_recon, x, mu, logvar, MMD_sigma, kernel_num, matched_IO=False, 
                  reconstruction_loss='mse'):

    # Define loss functions
    if not matched_IO:
        matching_function_interv = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num) # MMD Distance since we don't have paired data
        matching_function_interv_marker = MMD_loss(fix_sigma=200, kernel_num=10) # MMD Distance since we don't have paired data
    else:
        matching_function_interv = nn.MSELoss() # MSE if there is matched interv/observ samples

    # Compute losses
    if y_hat is None:
        MMD = 0
    else:
        MMD = matching_function_interv(y_hat, y)

    if y_hat_marker is None:
        MMD_marker = 0
    else:
        MMD_marker = matching_function_interv_marker(y_hat_marker, y_marker)
    
    if reconstruction_loss == 'mse':
        matching_function_recon = nn.MSELoss() # MSE for reconstruction, reduction = mean by default
    elif reconstruction_loss == 'mmd':
        matching_function_recon = MMD_loss(fix_sigma=MMD_sigma, kernel_num=kernel_num)
    recon_loss = matching_function_recon(x_recon, x)

    if logvar is None:
        KLD = 0
    else:
        KLD = -0.5*torch.sum(logvar -(mu.pow(2)+logvar.exp())+1)/x.shape[0]
    return MMD, recon_loss, KLD, MMD_marker

## Loss parameters
beta_schedule = torch.zeros(train_epoch) # weight on the KLD
beta_schedule[:10] = 0
beta_schedule[10:] = torch.linspace(0,mxBeta,train_epoch-10) 
alpha_schedule = torch.zeros(train_epoch) # weight on the MMD
alpha_schedule[:] = mxAlpha
alpha_schedule[:5] = 0
alpha_schedule[5:int(train_epoch/2)] = torch.linspace(0,mxAlpha,int(train_epoch/2)-5) 
alpha_schedule[int(train_epoch/2):] = mxAlpha

model_ft.double()
model_ft.to(device)

# Only include parameters that require gradients in the optimizer
optimizer = torch.optim.Adam(model_ft.parameters(), lr=config['lr'])

if log:
    project_name = f'transfer_few_shot_{test_dataset}'
    wandb.init(project=project_name, name=savedir.split('/')[-1])  #name should be the run time after fixing the os.makedirs bug

for n in range(0, train_epoch):
    lossAv = 0
    ct = 0
    mmdAv = 0
    mmdMarkerAv = 0
    reconAv = 0
    klAv = 0

    for i, X in tqdm(enumerate(dataloader_train), total=len(dataloader_train)):
        x = X[0]  # ctrl samples
        y = X[1]  # ptb samples
        c_1 = X[2]  # ptb target labels (target 1)
        c_2 = X[3]  # ptb target labels (target 2)
        C_y = X[4]  # ptb target names

        x = x.to(device)
        y = y.to(device)
        c_1 = c_1.to(device)
        c_2 = c_2.to(device)

        optimizer.zero_grad()
        
        y_hat, x_recon, z_mu, z_logvar = model_ft(x, c_1, c_2)
        
        mmd_loss, recon_loss, kl_loss, _ = loss_function(y_hat, y, 
                                                         None, None, 
                                                         x_recon, x, 
                                                         z_mu, z_logvar, 
                                                         config['MMD_sigma'], config['kernel_num'], config['matched_IO'],
                                                         reconstruction_loss='mmd')
        loss = alpha_schedule[n]*mmd_loss + recon_loss + beta_schedule[n]*kl_loss
        
        loss.backward()
        if config['grad_clip']:
            for param in model_ft.parameters():
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
            wandb.log({'loss':loss})
            wandb.log({'mmd_loss':mmd_loss})
            wandb.log({'recon_loss':recon_loss})
            wandb.log({'kl_loss':kl_loss})
        
    print('Epoch ' + str(n) + ': Loss=' + str(lossAv / ct) + ', ' + 'MMD=' + str(mmdAv / ct) + ', ' + 'MSE=' + str(reconAv / ct) + ', ' + 'KL=' + str(klAv / ct))
    
    if log:
        wandb.log({'epoch avg loss': lossAv/ct})
        wandb.log({'epoch avg mmd_loss': mmdAv/ct})
        wandb.log({'epoch avg recon_loss': reconAv/ct})
        wandb.log({'epoch avg kl_loss': klAv/ct})
        wandb.log({'beta': beta_schedule[n]})
        wandb.log({'Epoch:': n})

print('Finished training')
print('----------------------------------------------------------------------------------------------')

# Save the model
torch.save(model_ft, f'{output_dir}/model_ft.pt')

# 5. Infer on test cell line --------------------------------------------------
model_ft = model_ft.to(device)
model_ft.eval()

# Result dictionaries
gt_y = []
pred_y_1 = []
gt_c_1 = []
gt_c_2 = []
gt_c_1_2 = []
gt_c_2_2 = []
gt_C_y = []
gt_x = []
gt_x_recon = []

for i, X in tqdm(enumerate(dataloader_infer), total=len(dataloader_infer)):
    x = X[0] #ctrl samples
    y = X[1] #ptb samples
    c_1 = X[2] #ptb target labels (target 1)
    c_2 = X[3] #ptb target labels (target 2)
    C_y = X[4] #ptb target names

    x = x.to(device)
    c_1 = c_1.to(device)
    c_2 = c_2.to(device)

    with torch.no_grad():
        y_hat, x_recon, _ , _ = model_ft(x, c_1, c_2)
    
    gt_x.append(x.cpu().numpy())
    gt_x_recon.append(x_recon.cpu().numpy())
    gt_y.append(y.numpy())
    gt_c_1.append(c_1.detach().cpu().numpy())
    gt_c_2.append(c_2.detach().cpu().numpy())
    gt_c_1_2.append(None)
    gt_c_2_2.append(None)
    pred_y_1.append(y_hat.detach().cpu().numpy() if model_ft is not None else None)
    gt_C_y.append(np.array(C_y))

result_dic = {
    'gt_x': np.vstack(gt_x),
    'gt_x_recon': np.vstack(gt_x_recon),
    'gt_y': np.vstack(gt_y),
    'pred_y_1': np.vstack(pred_y_1),
    'gt_c_1': np.vstack(gt_c_1),
    'gt_c_2': np.vstack(gt_c_2),
    'gt_c_1_2': np.vstack(gt_c_1_2),
    'gt_c_2_2': np.vstack(gt_c_2_2),
    'gt_C_y': np.vstack(gt_C_y)
}

C_y = result_dic['gt_C_y'].flatten().tolist()
print('test set on all overlapping perturbations')
print("There are", len(set(C_y)), "unique perturbations in the test set.")
print("These testing perturbations are:", set(C_y))

pred_morph = {} 
batch_size = config['batch_size']
# Get predictions
for i in range(len(C_y)//batch_size):
    c = C_y[i*batch_size]
    c_test = np.unique(result_dic['gt_C_y'][i]).item()
    assert c == c_test, 'The perturbation is not the same!'
    # Predictions from MORPH
    if c not in pred_morph.keys():
        pred_morph[c] = result_dic['pred_y_1'][i*batch_size:(i+1)*batch_size]
    else:
        pred_morph[c] = np.vstack((pred_morph[c], result_dic['pred_y_1'][i*batch_size:(i+1)*batch_size]))

# Save predictions
with open(f'{output_dir}/pred.pkl', 'wb') as f:
    pickle.dump(pred_morph, f)

# Define MMD loss function
mmd_loss_de_fn =  MMD_loss(fix_sigma=200, kernel_num=10)
mmd_loss_whole_fn = MMD_loss(fix_sigma=1500, kernel_num=10)
print('Using MMD with fixed sigma = 200 for DE genes and 1500 for whole genome')

from sklearn.metrics import mean_squared_error as mse
### Calculate R^2 and RMSE using the top 50 marker genes
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)

# Evaluations on DE genes for each perturbation ------------
mmd_loss_de = {}
rmse_loss_de = {}
mse_loss_de = {}
r2_de = {}
l2_de = {}
pearsonr_de = {}
# change over control
rmse_change_de = {}
pearsonr_change_de = {}
fraction_de = {}

# Evaluations on whole genomes for each perturbation ------------
mmd_loss_whole = {}
rmse_loss_whole = {}
mse_loss_whole = {}
r2_whole = {}
pearsonr_whole = {}
# change over control
rmse_change_whole = {}
pearsonr_change_whole = {}
fraction_whole = {}

for pert in tqdm(set(C_y)):
    pert_name = pert

    deg_cat = f'{pert_name}'
    deg_list = adata_test.uns['rank_genes_groups']['names'][deg_cat]
    
    y_true = adata_test[adata_test.obs['gene'] == pert_name].X
    y_pred = pred_morph[pert_name]
    y_ctrl = adata_test[adata_test.obs['gene'] == 'non-targeting'].X

    n_top_deg = 50
    degs = [np.where(adata_test.var_names == gene)[0][0] for gene in deg_list[:n_top_deg]]
    # print(degs)
    
    y_true_deg = y_true[:, degs]
    y_pred_deg = y_pred[:, degs]
    y_ctrl_deg = y_ctrl[:, degs]
    
    # Evaluations on DE genes for each perturbation ------------
    rmse_loss_de[pert_name] = np.sqrt(mse(y_true_deg.mean(0), y_pred_deg.mean(0)))
    mse_loss_de[pert_name] = mse(y_true_deg.mean(0), y_pred_deg.mean(0))
    r2_de[pert_name] = max(r2_score(y_true_deg.mean(0), y_pred_deg.mean(0)),0)
    l2_de[pert_name] = np.linalg.norm(y_true_deg.mean(0) - y_pred_deg.mean(0))
    pearsonr_de[pert_name] = pearsonr(y_true_deg.mean(0), y_pred_deg.mean(0))[0]
    # change over control
    rmse_change_de[pert_name] = np.sqrt(mse(y_true_deg.mean(0) - y_ctrl_deg.mean(0), y_pred_deg.mean(0) - y_ctrl_deg.mean(0)))
    pearsonr_change_de[pert_name] = pearsonr(y_true_deg.mean(0) - y_ctrl_deg.mean(0), y_pred_deg.mean(0) - y_ctrl_deg.mean(0))[0]
    fraction_de[pert_name] = np.sum(np.sign(y_true_deg.mean(0) - y_ctrl_deg.mean(0)) == np.sign(y_pred_deg.mean(0) - y_ctrl_deg.mean(0)))/n_top_deg

    # Evaluations on whole genomes for each perturbation ------------
    rmse_loss_whole[pert_name] = np.sqrt(mse(y_true.mean(0), y_pred.mean(0)))
    mse_loss_whole[pert_name] = mse(y_true.mean(0), y_pred.mean(0))
    r2_whole[pert_name] = max(r2_score(y_true.mean(0), y_pred.mean(0)),0)
    pearsonr_whole[pert_name] = pearsonr(y_true.mean(0), y_pred.mean(0))[0]
    # change over control
    rmse_change_whole[pert_name] = np.sqrt(mse(y_true.mean(0) - y_ctrl.mean(0), y_pred.mean(0) - y_ctrl.mean(0)))
    pearsonr_change_whole[pert_name] = pearsonr(y_true.mean(0) - y_ctrl.mean(0), y_pred.mean(0) - y_ctrl.mean(0))[0]
    fraction_whole[pert_name] = np.sum(np.sign(y_true.mean(0) - y_ctrl.mean(0)) == np.sign(y_pred.mean(0) - y_ctrl.mean(0)))/y_true.shape[1]

    y_pred_mmd_deg = y_pred_deg
    y_true_mmd_deg = y_true_deg
    y_pred_mmd_whole = y_pred
    y_true_mmd_whole = y_true

    # Empirical MMD is divided by sample size, so we calculate MMD by batches and then take the average
    min_sample_size = min(y_pred_mmd_deg.shape[0], y_true_mmd_deg.shape[0])
    y_pred_mmd_deg = y_pred_mmd_deg[np.random.choice(y_pred_mmd_deg.shape[0], min_sample_size, replace=False),:]
    y_true_mmd_deg = y_true_mmd_deg[np.random.choice(y_true_mmd_deg.shape[0], min_sample_size, replace=False),:]
    y_pred_mmd_whole = y_pred_mmd_whole[np.random.choice(y_pred_mmd_whole.shape[0], min_sample_size, replace=False),:]
    y_true_mmd_whole = y_true_mmd_whole[np.random.choice(y_true_mmd_whole.shape[0], min_sample_size, replace=False),:]
    
    min_counts = 32
    num_batches = min_sample_size // min_counts
    for i in range(num_batches):
        mmd_loss_de_temp = mmd_loss_de_fn(torch.tensor(y_pred_mmd_deg[i*min_counts:(i+1)*min_counts]), torch.tensor(y_true_mmd_deg[i*min_counts:(i+1)*min_counts])).item()
        mmd_loss_whole_temp = mmd_loss_whole_fn(torch.tensor(y_pred_mmd_whole[i*min_counts:(i+1)*min_counts]), torch.tensor(y_true_mmd_whole[i*min_counts:(i+1)*min_counts])).item()

        if pert_name not in mmd_loss_de.keys():
            mmd_loss_de[pert_name] = [mmd_loss_de_temp]
            mmd_loss_whole[pert_name] = [mmd_loss_whole_temp]
        else:
            mmd_loss_de[pert_name].append(mmd_loss_de_temp)
            mmd_loss_whole[pert_name].append(mmd_loss_whole_temp)

print('Test set on all overlapping perturbations')
print(set(C_y))
print('Size of test set: ', len(set(C_y)))
print('Save directory:', savedir)
print('-------------------')
print('which model it uses:', model_name)
print('-------------------')

print('Using top 50 marker genes')

print('MORPH, mmd')
mmd_loss_de_summary = {}
for k in mmd_loss_de.keys():
    mmd_loss_de_summary[k] = (np.average(mmd_loss_de[k]), np.std(mmd_loss_de[k]))
mmd_loss_whole_summary = {}
for k in mmd_loss_whole.keys():
    mmd_loss_whole_summary[k] = (np.average(mmd_loss_whole[k]), np.std(mmd_loss_whole[k]))

# format to 3 decimal places
mmd_de = np.mean([i[0] for i in mmd_loss_de_summary.values()])
print('mean', '%.5f'%(mmd_de))
print('ste', np.std([i[0] for i in mmd_loss_de_summary.values()])/np.sqrt(len(mmd_loss_de_summary.keys())))

mmd_whole = np.mean([i[0] for i in mmd_loss_whole_summary.values()])
print('mean', '%.5f'%(mmd_whole))
print('ste', np.std([i[0] for i in mmd_loss_whole_summary.values()])/np.sqrt(len(mmd_loss_whole_summary.keys())))

print('morph, rmse')
rmse_de = np.mean([i for i in rmse_loss_de.values()])
print('mean', '%.5f'%(rmse_de))
print('ste', np.std([i for i in rmse_loss_de.values()])/np.sqrt(len(rmse_loss_de.keys())))

print('morph, rmse (whole genome)')
rmse_whole = np.mean([i for i in rmse_loss_whole.values()])
print('mean', '%.5f'%(rmse_whole))
print('ste', np.std([i for i in rmse_loss_whole.values()])/np.sqrt(len(rmse_loss_whole.keys())))

print('morph, mse')
mse_de = np.mean([i for i in mse_loss_de.values()])
print('mean', '%.5f'%(mse_de))
print('ste', np.std([i for i in mse_loss_de.values()])/np.sqrt(len(mse_loss_de.keys())))

print('morph, mse (whole genome)')
mse_whole = np.mean([i for i in mse_loss_whole.values()])
print('mean', '%.5f'%(mse_whole))
print('ste', np.std([i for i in mse_loss_whole.values()])/np.sqrt(len(mse_loss_whole.keys())))

print('morph, r2')
r2_de_mean = np.mean([i for i in r2_de.values()])
print('mean', '%.5f'%(r2_de_mean))
print('ste', np.std([i for i in r2_de.values()])/np.sqrt(len(r2_de.keys())))

print('morph, r2 (whole genome)')
r2_whole_mean = np.mean([i for i in r2_whole.values()])
print('mean', '%.5f'%(r2_whole_mean))
print('ste', np.std([i for i in r2_whole.values()])/np.sqrt(len(r2_whole.keys())))

print('morph, l2')
l2_de_mean = np.mean([i for i in l2_de.values()])
print('mean', '%.5f'%(l2_de_mean))

print('morph, pearsonr')
pearsonr_de_mean = np.nanmean([i for i in pearsonr_de.values()]) #some ptb has nan pearsonr as their true degs might be all 0s
print('mean', '%.5f'%(pearsonr_de_mean))

print('morph, pearsonr (whole genome)')
pearsonr_whole_mean = np.mean([i for i in pearsonr_whole.values()])
print('mean', '%.5f'%(pearsonr_whole_mean))

# change over control checks
print('morph, rmse change')
rmse_change_de_mean = np.mean([i for i in rmse_change_de.values()])
print('mean', '%.5f'%(rmse_change_de_mean))

print('morph, pearsonr change')
pearsonr_change_de_mean = np.mean([i for i in pearsonr_change_de.values()])
print('mean', '%.5f'%(pearsonr_change_de_mean))

print('morph, fraction of DE genes with same direction')
fraction_de_mean = np.mean([i for i in fraction_de.values()])
print('mean', '%.5f'%(fraction_de_mean))

print('morph, rmse change (whole genome)')
rmse_change_whole_mean = np.mean([i for i in rmse_change_whole.values()])
print('mean', '%.5f'%(rmse_change_whole_mean))

print('morph, pearsonr change (whole genome)')
pearsonr_change_whole_mean = np.mean([i for i in pearsonr_change_whole.values()])
print('mean', '%.5f'%(pearsonr_change_whole_mean))

print('morph, fraction of DE genes with same direction (whole genome)')
fraction_whole_mean = np.mean([i for i in fraction_whole.values()])
print('mean', '%.5f'%(fraction_whole_mean))

print('-------------------')

### Save results -----------------------------------
df = pd.DataFrame()
df['pert'] = mmd_loss_de_summary.keys()
df['mmd_de'] = [i[0] for i in mmd_loss_de_summary.values()]
df['rmse_de'] = [i for i in rmse_loss_de.values()]
df['mse_de'] = [i for i in mse_loss_de.values()]
df['r2_de'] = [i for i in r2_de.values()]
df['pearsonr_de'] = [i for i in pearsonr_de.values()]
df['rmse_change_de'] = [i for i in rmse_change_de.values()]
df['pearsonr_change_de'] = [i for i in pearsonr_change_de.values()]
df['fraction_de'] = [i for i in fraction_de.values()] 

csv_path = f'{output_dir}/results.csv'
df.to_csv(csv_path, index=False)

### Save summary results -----------------------------------
# save results into csv file
results_df = pd.DataFrame({
    'mmd_de': mmd_de,
    'mmd_whole': mmd_whole,
    'r2_de': r2_de_mean ,
    'r2_whole': r2_whole_mean ,
    'l2_de': l2_de_mean ,
    'mse_de': mse_de ,
    'mse_whole': mse_whole ,
    'rmse_de': rmse_de ,
    'rmse_whole': rmse_whole ,
    'pearsonr_de': pearsonr_de_mean ,
    'pearsonr_whole': pearsonr_whole_mean ,
    'rmse_change': rmse_change_de_mean ,
    'rmse_change_whole': rmse_change_whole_mean ,
    'pearsonr_change': pearsonr_change_de_mean ,
    'pearsonr_change_whole': pearsonr_change_whole_mean ,
    'fraction': fraction_de_mean ,
    'fraction_whole': fraction_whole_mean ,
    'note': ['Using top 50 marker genes']
})

# save results into csv file
csv_path = f'{output_dir}/summary_results.csv'
results_df.to_csv(csv_path, index=False)
