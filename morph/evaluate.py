import numpy as np
import pandas as pd
import random
import pickle
import torch
from warnings import filterwarnings
from argparse import ArgumentParser
from tqdm import tqdm
import json
import os
import glob

filterwarnings('ignore')

import scanpy as sc
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from inference import *
from utils import MMD_loss

### Set up experiment environments ------------------------------------------------
def main():
    parser = ArgumentParser(description="Run evaluations")
    parser.add_argument('--modality', type=str, default='rna', help="Modality to use (rna or ops)")
    parser.add_argument('--dataset_name', type=str, default='replogle_rpe1_hvg', help="Name of the dataset to evaluate on")
    parser.add_argument('--leave_out_test_set_id', type=str, default='random_fold_1', help="Test set id")
    parser.add_argument('--label', type=str, default='DepMap_GeneEffect', help="Type of representation to use")
    parser.add_argument('--null_label', type=str, default="zeros", help="Null label (zeros, gaussian, or gaussian_normalized)")
    parser.add_argument('--model_type', type=str, default='MORPH', help="Type of model to use")
    parser.add_argument('--model_name', type=str, default='best_model.pt', help="Name of the model")
    parser.add_argument('--random_seed', type=int, default=12, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda:4', help="Device to run on")
    parser.add_argument('--run_name', type=str, default='', help="Name of the run (if not specified, will use the most recent run)")
    args = parser.parse_args()
    
    args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    modality = args.modality
    print('Modality:', modality)
    dataset_name = args.dataset_name
    print('Evaluating on', dataset_name)
    leave_out_test_set_id = args.leave_out_test_set_id
    print('Test set:', leave_out_test_set_id)
    representation_type = args.label
    print('Representation type:', representation_type)
    null_label = args.null_label
    print('Null label:', null_label)
    model_name = args.model_name
    print('Model name:', model_name)
    random_seed = args.random_seed
    print('Random seed:', random_seed)
    device = args.device
    print('Device:', device)
    run_name = args.run_name
    print('Run name:', run_name)

    # Load single-cell data -----------------------------------
    scdata_file = pd.read_csv(f'{args.base_dir}/data/scdata_file_path.csv')
    adata_path = scdata_file[scdata_file['dataset'] == args.dataset_name]['file_path'].values[0]
    adata = sc.read(adata_path)
    print('Loaded adata from ', adata_path)

    # Get DEGs ------------------------------------------------
    if modality == 'rna':
        print('Getting DEGs')
        sc.tl.rank_genes_groups(adata, groupby='gene', reference='non-targeting', n_genes=50, use_raw=False, rankby_abs=True)
     
    # Load model ----------------------------------------------
    savedir = os.path.join(f"{args.base_dir}/result", modality, dataset_name, leave_out_test_set_id)

    if run_name == '':
        print('No run name specified. Using the most recent run.')
        # Construct the pattern to match the saved directories
        pattern = os.path.join(savedir, f'{representation_type}_{args.model_type}_run*')
        run_dirs = glob.glob(pattern)
        
        # Sort directories by creation time (most recent last)
        run_dirs = sorted(run_dirs, key=os.path.getctime)
        # Get the most recent run directory
        if run_dirs:
            most_recent_run_dir = run_dirs[-1]
            print(f"Most recent run directory: {most_recent_run_dir}")
        else:
            print("No run directories found.")
    else:
        assert representation_type in run_name, 'The representation type is not in the run name!'
        assert args.model_type in run_name, 'The model type is not in the run name!'
        most_recent_run_dir = os.path.join(savedir, run_name)
        print(f"Using run directory: {most_recent_run_dir}")
    
    model = torch.load(f'{most_recent_run_dir}/{model_name}', map_location=device) if most_recent_run_dir is not None else None
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params:,}.")

    # load configs (read json file)
    with open(f'{most_recent_run_dir}/config.json', 'r') as f:
        config = json.load(f)
    assert modality == config['modality']
    batch_size = config['batch_size']
    min_counts = batch_size
    assert args.model_type == config['model']
    label_1 = config['label']
    assert label_1 == representation_type
    label_2 = config['label_2']
    label_3 = config['label_3']
    mmd_sigma = config['MMD_sigma']
    assert random_seed == config['seed']
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    print('Random seed:', random_seed)
    
    # Load data ----------------------------------------------
    result_dic = evaluate_single_model(model, 
                                       most_recent_run_dir, 
                                       device, 
                                       use_index=True, 
                                       infer_data=None)
    
    C_y = result_dic['gt_C_y'].flatten().tolist()
    print('test set:', leave_out_test_set_id)
    print("There are", len(set(C_y)), "unique perturbations in the test set.")
    print("These testing perturbations are:", set(C_y))
    assert all(x is not None for x in C_y), 'Some cells are not in the embedding dictionary!'

    pred_morph = {} 
    batch_size = config['batch_size']
    # Get predictions from MORPH
    for i in range(len(C_y)//batch_size):
        c = C_y[i*batch_size]
        c_test = np.unique(result_dic['gt_C_y'][i]).item()
        assert c == c_test, 'The perturbation is not the same!'
        # Predictions from model
        if c not in pred_morph.keys():
            pred_morph[c] = result_dic['pred_y_1'][i*batch_size:(i+1)*batch_size]
        else:
            pred_morph[c] = np.vstack((pred_morph[c], result_dic['pred_y_1'][i*batch_size:(i+1)*batch_size]))
    
    output_path = f"{most_recent_run_dir}/{model_name.replace('.pt', '')}_pred_test.pkl"
    print('Saving MORPH predictions into pkl file to', output_path, '...')
    with open(output_path, 'wb') as f:
        pickle.dump(pred_morph, f)

    ### Calculate losses -----------------------------------
    if modality == 'rna':
        mmd_loss =  MMD_loss(fix_sigma=200, kernel_num=10)
        print('Using MMD with fixed sigma = 200')
    elif modality == 'ops':
        mmd_loss =  MMD_loss(fix_sigma=mmd_sigma, kernel_num=10)
        print(f'Using MMD with fixed sigma = {mmd_sigma}')

    from sklearn.metrics import mean_squared_error as mse
    ### Calculate R^2 and RMSE using the top 50 marker genes
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    mmd_loss_morph = {}
    rmse_loss_morph = {}
    mse_loss_morph = {}
    r2_morph = {}
    l2_morph = {}
    pearsonr_morph = {}

    # change over control
    rmse_change_morph = {}
    pearsonr_change_morph = {}
    fraction_morph = {}

    rmse_loss_morph_whole = {}
    mse_loss_morph_whole = {}
    r2_morph_whole = {}
    pearsonr_morph_whole = {}

    # change over control
    rmse_change_morph_whole = {}
    pearsonr_change_morph_whole = {}
    fraction_morph_whole = {}

    for pert in tqdm(set(C_y)):
        pert_name = pert
        # print('pert_name:', pert_name)

        y_true = adata[adata.obs['gene'] == pert_name].X
        y_pred = pred_morph[pert_name]
        y_ctrl = adata[adata.obs['gene'] == 'non-targeting'].X

        if modality == 'rna':
            # For RNA-seq data, evaluate on the top 50 marker genes
            deg_cat = f'{pert_name}'
            deg_list = adata.uns['rank_genes_groups']['names'][deg_cat]

            n_top_deg = 50
            degs = [np.where(adata.var_names == gene)[0][0] for gene in deg_list[:n_top_deg]]
        
            y_true_deg = y_true[:, degs]
            y_pred_deg = y_pred[:, degs]
            y_ctrl_deg = y_ctrl[:, degs]

            rmse_loss_morph[pert_name] = np.sqrt(mse(y_true_deg.mean(0), y_pred_deg.mean(0)))
            mse_loss_morph[pert_name] = mse(y_true_deg.mean(0), y_pred_deg.mean(0))
            r2_morph[pert_name] = max(r2_score(y_true_deg.mean(0), y_pred_deg.mean(0)),0)
            l2_morph[pert_name] = np.linalg.norm(y_true_deg.mean(0) - y_pred_deg.mean(0))
            pearsonr_morph[pert_name] = pearsonr(y_true_deg.mean(0), y_pred_deg.mean(0))[0]
            # change over control
            rmse_change_morph[pert_name] = np.sqrt(mse(y_true_deg.mean(0) - y_ctrl_deg.mean(0), y_pred_deg.mean(0) - y_ctrl_deg.mean(0)))
            pearsonr_change_morph[pert_name] = pearsonr(y_true_deg.mean(0) - y_ctrl_deg.mean(0), y_pred_deg.mean(0) - y_ctrl_deg.mean(0))[0]
            # percentage of de genes with same direction (change from control)
            fraction_morph[pert_name] = np.sum(np.sign(y_true_deg.mean(0) - y_ctrl_deg.mean(0)) == np.sign(y_pred_deg.mean(0) - y_ctrl_deg.mean(0)))/n_top_deg

            rmse_loss_morph_whole[pert_name] = np.sqrt(mse(y_true.mean(0), y_pred.mean(0)))
            mse_loss_morph_whole[pert_name] = mse(y_true.mean(0), y_pred.mean(0))
            r2_morph_whole[pert_name] = max(r2_score(y_true.mean(0), y_pred.mean(0)),0)
            pearsonr_morph_whole[pert_name] = pearsonr(y_true.mean(0), y_pred.mean(0))[0]
            # change over control
            rmse_change_morph_whole[pert_name] = np.sqrt(mse(y_true.mean(0) - y_ctrl.mean(0), y_pred.mean(0) - y_ctrl.mean(0)))
            pearsonr_change_morph_whole[pert_name] = pearsonr(y_true.mean(0) - y_ctrl.mean(0), y_pred.mean(0) - y_ctrl.mean(0))[0]
            fraction_morph_whole[pert_name] = np.sum(np.sign(y_true.mean(0) - y_ctrl.mean(0)) == np.sign(y_pred.mean(0) - y_ctrl.mean(0)))/y_true.shape[1]

            y_pred_mmd = y_pred_deg
            y_true_mmd = y_true_deg

        elif modality == 'ops':
            y_true_mmd = y_true
            y_pred_mmd = y_pred

        # Empirical MMD is divided by sample size, so we should calculate MMD by batches and then take the average
        min_sample_size = min(y_pred_mmd.shape[0], y_true_mmd.shape[0])
        # print('min_sample_size:', min_sample_size)
        y_pred_mmd = y_pred_mmd[np.random.choice(y_pred_mmd.shape[0], min_sample_size, replace=False),:]
        y_true_mmd = y_true_mmd[np.random.choice(y_true_mmd.shape[0], min_sample_size, replace=False),:]
        
        num_batches = min_sample_size // min_counts
        for i in range(num_batches):
            mmd_loss_temp = mmd_loss(torch.tensor(y_pred_mmd[i*min_counts:(i+1)*min_counts]), torch.tensor(y_true_mmd[i*min_counts:(i+1)*min_counts])).item()

            if pert_name not in mmd_loss_morph.keys():
                mmd_loss_morph[pert_name] = [mmd_loss_temp]
            else:
                mmd_loss_morph[pert_name].append(mmd_loss_temp)

    leave_out_list = config['ptb_leave_out_list']
    assert(len(mmd_loss_morph) == len(leave_out_list)), 'The number of perturbations is not the same!'
    assert(set(mmd_loss_morph.keys()) == set(leave_out_list)), 'The perturbations are not the same!'

    print('Test set:', leave_out_test_set_id)
    print(set(C_y))
    print('Size of test set: ', len(set(C_y)))
    print('Save directory:', savedir)
    print('-------------------')
    print('which model it uses:', model_name)
    print('-------------------')

    if modality == 'rna':
        print('Using top 50 marker genes')

    print('MORPH, mmd')
    mmd_loss_summary = {}
    for k in mmd_loss_morph.keys():
        mmd_loss_summary[k] = (np.average(mmd_loss_morph[k]), np.std(mmd_loss_morph[k]))

    # format to 3 decimal places
    mmd_morph = np.mean([i[0] for i in mmd_loss_summary.values()])
    print('mean', '%.5f'%(mmd_morph))
    print('ste', np.std([i[0] for i in mmd_loss_summary.values()])/np.sqrt(len(mmd_loss_summary.keys())))

    if modality == 'rna':
        print('morph, rmse')
        rmse_morph = np.mean([i for i in rmse_loss_morph.values()])
        print('mean', '%.5f'%(rmse_morph))
        print('ste', np.std([i for i in rmse_loss_morph.values()])/np.sqrt(len(rmse_loss_morph.keys())))

        print('morph, rmse (whole genome)')
        rmse_morph_whole = np.mean([i for i in rmse_loss_morph_whole.values()])
        print('mean', '%.5f'%(rmse_morph_whole))
        print('ste', np.std([i for i in rmse_loss_morph_whole.values()])/np.sqrt(len(rmse_loss_morph_whole.keys())))

        print('morph, mse')
        mse_morph = np.mean([i for i in mse_loss_morph.values()])
        print('mean', '%.5f'%(mse_morph))
        print('ste', np.std([i for i in mse_loss_morph.values()])/np.sqrt(len(mse_loss_morph.keys())))

        print('morph, mse (whole genome)')
        mse_morph_whole = np.mean([i for i in mse_loss_morph_whole.values()])
        print('mean', '%.5f'%(mse_morph_whole))
        print('ste', np.std([i for i in mse_loss_morph_whole.values()])/np.sqrt(len(mse_loss_morph_whole.keys())))

        print('morph, r2')
        morph_r2 = np.mean([i for i in r2_morph.values()])
        print('mean', '%.5f'%(morph_r2))
        print('ste', np.std([i for i in r2_morph.values()])/np.sqrt(len(r2_morph.keys())))

        print('morph, r2 (whole genome)')
        morph_r2_whole = np.mean([i for i in r2_morph_whole.values()])
        print('mean', '%.5f'%(morph_r2_whole))
        print('ste', np.std([i for i in r2_morph_whole.values()])/np.sqrt(len(r2_morph_whole.keys())))

        print('morph, l2')
        morph_l2 = np.mean([i for i in l2_morph.values()])
        print('mean', '%.5f'%(morph_l2))

        print('morph, pearsonr')
        pearsonr_morph_mean = np.nanmean([i for i in pearsonr_morph.values()]) #some ptb has nan pearsonr as their true degs might be all 0s
        print('mean', '%.5f'%(pearsonr_morph_mean))

        print('morph, pearsonr (whole genome)')
        pearsonr_morph_whole_mean = np.mean([i for i in pearsonr_morph_whole.values()])
        print('mean', '%.5f'%(pearsonr_morph_whole_mean))

        # change over control checks
        print('morph, rmse change')
        rmse_change_morph_mean = np.mean([i for i in rmse_change_morph.values()])
        print('mean', '%.5f'%(rmse_change_morph_mean))

        print('morph, pearsonr change')
        pearsonr_change_morph_mean = np.mean([i for i in pearsonr_change_morph.values()])
        print('mean', '%.5f'%(pearsonr_change_morph_mean))

        print('morph, fraction of DE genes with same direction')
        fraction_morph_mean = np.mean([i for i in fraction_morph.values()])
        print('mean', '%.5f'%(fraction_morph_mean))

        print('morph, rmse change (whole genome)')
        rmse_change_morph_whole_mean = np.mean([i for i in rmse_change_morph_whole.values()])
        print('mean', '%.5f'%(rmse_change_morph_whole_mean))

        print('morph, pearsonr change (whole genome)')
        pearsonr_change_morph_whole_mean = np.mean([i for i in pearsonr_change_morph_whole.values()])
        print('mean', '%.5f'%(pearsonr_change_morph_whole_mean))

        print('morph, fraction of DE genes with same direction (whole genome)')
        fraction_morph_whole_mean = np.mean([i for i in fraction_morph_whole.values()])
        print('mean', '%.5f'%(fraction_morph_whole_mean))

    ### Save results -----------------------------------
    df = pd.DataFrame()
    df['pert'] = mmd_loss_summary.keys()
    df['mmd'] = [i[0] for i in mmd_loss_summary.values()]
    if modality == 'rna':
        df['rmse'] = [i for i in rmse_loss_morph.values()]
        df['mse'] = [i for i in mse_loss_morph.values()]
        df['r2'] = [i for i in r2_morph.values()]
        df['pearsonr'] = [i for i in pearsonr_morph.values()]
        df['rmse_change'] = [i for i in rmse_change_morph.values()]
        df['pearsonr_change'] = [i for i in pearsonr_change_morph.values()]
        df['fraction'] = [i for i in fraction_morph.values()] 

    # save this df into csv file
    csv_output_path = most_recent_run_dir+ '/' + model_name.replace('.pt', '') + '_evaluations.csv'
    print('Saving evaluation results into csv file to', csv_output_path, '...')
    df.to_csv(csv_output_path, index=False)

    ### Save summary results -----------------------------------
    new_row_df = pd.DataFrame([{
        'random_seed': random_seed,
        'data': dataset_name,
        'test_set_id': args.leave_out_test_set_id,
        'model_type': args.model_type,
        'model_name': model_name,
        'label_1': label_1,
        'label_2': label_2,
        'label_3': label_3,
        'mmd_sigma': mmd_sigma if modality == 'ops' else None,
        'mmd': mmd_morph,
        'r2': morph_r2 if modality == 'rna' else None,
        'l2': morph_l2 if modality == 'rna' else None,
        'mse': mse_morph if modality == 'rna' else None,
        'rmse': rmse_morph if modality == 'rna' else None,
        'pearsonr': pearsonr_morph_mean if modality == 'rna' else None,
        'pearsonr_whole': pearsonr_morph_whole_mean if modality == 'rna' else None,
        'rmse_change': rmse_change_morph_mean if modality == 'rna' else None,
        'rmse_change_whole': rmse_change_morph_whole_mean if modality == 'rna' else None,
        'pearsonr_change': pearsonr_change_morph_mean if modality == 'rna' else None,
        'pearsonr_change_whole': pearsonr_change_morph_whole_mean if modality == 'rna' else None,
        'fraction': fraction_morph_mean if modality == 'rna' else None,
        'fraction_whole': fraction_morph_whole_mean if modality == 'rna' else None,
        'model_path': most_recent_run_dir,
        'note': 'top 50 marker genes' if modality == 'rna' else 'ops data'
    }])
    
    # save this df into csv file
    csv_output_path = most_recent_run_dir+'/' + model_name.replace('.pt', '')+'_summary_evaluations.csv'
    print('Saving summary evaluation results into csv file to', csv_output_path, '...')
    new_row_df.to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    main()