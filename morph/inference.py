import torch
import numpy as np
import pickle
from collections import Counter
import json
from tqdm import tqdm
import random
import os
import pandas as pd

from torch.utils.data import DataLoader, Subset
from dataset import SCDataset
from utils import SCDATA_sampler

def evaluate_single_model(modality, model, savedir, device, use_index=False, infer_data = None):

    # load configs
    with open(f'{savedir}/config.json') as f:
        config = json.load(f)
    ptb_leave_out_list = config['ptb_leave_out_list']

    if 'label_3' not in config:
        config['label_3'] = None
    
    # Set random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])

    # Load split indices
    if use_index:
        with open(f'{savedir}/split_idx.pkl', 'rb') as f:
            split_idx = pickle.load(f)
            train_idx = split_idx['train_idx']
            infer_idx = split_idx['infer_idx']
        
        dataset = config['dataset']
        
        if config['label'] == 'Baseline':
            print('Loading one-hot vectors for ptb targets generated during training...')
            with open(f'{savedir}/ptb_vector_dict.pkl', 'rb') as f:
                gene_embs = pickle.load(f)
            print('Loaded from:', f'{savedir}/ptb_vector_dict.pkl')
        else:
            gene_embs = None
        
        dataset = SCDataset(modality = modality, dataset = config['dataset'], 
                            leave_out_test_set_id = config['leave_out_test_set_id'],
                            perturb_targets=None, 
                            representation_type=config['label'], 
                            representation_type_2 = config['label_2'],
                            representation_type_3=config['label_3'],
                            use_hvg=config['use_hvg'],
                            random_seed=config['seed'],
                            min_counts=config['batch_size'],
                            gene_embs=gene_embs)
        
        # Check that the saved ptb_leave_out_list is the same as the one in the dataset
        assert(set(ptb_leave_out_list) == set(dataset.ptb_leave_out_list)), 'Error: ptb_leave_out_list does not match the ptb targets in the dataset'
        print('dataset created')

        # Training dataloader
        dataset1 = Subset(dataset, train_idx) 
        train_ptb_name = dataset.ptb_names[train_idx] 
        dataloader_train = DataLoader(
            dataset1,
            batch_sampler=SCDATA_sampler(dataset1, config['batch_size'], train_ptb_name),
            num_workers=0
        )
        print('train dataloader created')

        # Inference dataloader
        dataset_infer = Subset(dataset, infer_idx)
        if infer_idx is None:
            raise ValueError('Inference set is None, stopping inference')
        else:
            ptb_name = dataset.ptb_names[infer_idx]
        assert(set(ptb_name) == set(ptb_leave_out_list)), "Perturbation targets in inference set are those specified in ptb_leave_out_list"
        assert(ptb_name.isin(train_ptb_name).any() == False), "Perturbation targets in inference set should NOT be in training set"
        dataloader = DataLoader(
            dataset_infer,
            batch_sampler=SCDATA_sampler(dataset_infer, config['batch_size'], ptb_name),
            num_workers=0
        )
        print('infer dataloader created')
    else:
        if infer_data is not None:
            dataloader = infer_data
        else:
            # Load data
            with open(f'{savedir}/infer_data.pkl', 'rb') as f:
                dataloader = pickle.load(f)

        dataloader_train = None

    # Loading seen ptb targets
    if dataloader_train is not None:
        all_ptb_names = dataloader_train.dataset.dataset.ptb_names
        train_idx = dataloader_train.dataset.indices
        train_ptb_name = all_ptb_names[train_idx]
        train_ptb_name = np.unique(train_ptb_name)

        # None of ptb_leave_out_list should be in train_ptb_name
        assert(set(ptb_leave_out_list).isdisjoint(set(train_ptb_name))), 'Error: ptb_leave_out_list should not be in train_ptb_name'

    model = model.to(device)
    model.eval()

    # Result dictionaries
    gt_y = []
    pred_y_1 = []
    gt_c_1 = []
    gt_c_2 = []
    gt_c_1_2 = []
    gt_c_2_2 = []
    gt_C_y = []
    gt_x = []

    for i, X in tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating model'):
        x = X[0] #ctrl samples
        y = X[1] #ptb samples
        c_1 = X[2] #ptb target labels (target 1)
        c_2 = X[3] #ptb target labels (target 2)
        C_y = X[4] #ptb target names

        x = x.to(device)
        c_1 = c_1.to(device)
        c_2 = c_2.to(device)

        if 'moe' in config['model']:
            c_1_2 = X[5].to(device)
            c_2_2 = X[6].to(device)
            if '3expert' in config['model']:
                c_1_3 = X[7].to(device)
                c_2_3 = X[8].to(device)

        with torch.no_grad():
            if 'moe' in config['model']:
                if '3expert' in config['model']:
                    y_hat, _, _ , _ = model(x, c_1, c_2, c_1_2, c_2_2, c_1_3, c_2_3, num_interv=2)
                else:
                    y_hat, _, _ , _ = model(x, c_1, c_2, c_1_2, c_2_2, num_interv=2)
            else:
                y_hat, _, _ , _ = model(x, c_1, c_2, num_interv=2)
        
        gt_x.append(x.cpu().numpy())
        gt_y.append(y.numpy())
        gt_c_1.append(c_1.detach().cpu().numpy())
        gt_c_2.append(c_2.detach().cpu().numpy())
        if 'moe' in config['model']:
            gt_c_1_2.append(c_1_2.detach().cpu().numpy())
            gt_c_2_2.append(c_2_2.detach().cpu().numpy())
        else:
            gt_c_1_2.append(None)
            gt_c_2_2.append(None)
        pred_y_1.append(y_hat.detach().cpu().numpy() if model is not None else None)
        gt_C_y.append(np.array(C_y))

    result_dic = {
        'gt_x': np.vstack(gt_x),
        'gt_y': np.vstack(gt_y),
        'pred_y_1': np.vstack(pred_y_1),
        'gt_c_1': np.vstack(gt_c_1),
        'gt_c_2': np.vstack(gt_c_2),
        'gt_c_1_2': np.vstack(gt_c_1_2),
        'gt_c_2_2': np.vstack(gt_c_2_2),
        'gt_C_y': np.vstack(gt_C_y)
    }

    return result_dic