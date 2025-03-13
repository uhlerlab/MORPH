import os
import argparse
from argparse import Namespace
import time
import json
import pickle

import torch
import numpy as np
import random

from train import train_validate
from utils import get_data


def main(args):
	print(f'using device: {args.device}')
	torch.set_num_threads(8)
	
	opts = Namespace(
		num_interv = 2,
		mode = 'train',
		lr = 1e-4,
		kaiming_init = True,
		grad_clip = True,
		kernel_num = 10,
		matched_IO = False
	)
	
	opts.seed = args.random_seed
	torch.manual_seed(opts.seed)
	np.random.seed(opts.seed)
	random.seed(opts.seed)
	opts.validation_set_ratio = args.validation_set_ratio
	opts.validation_ood_ratio = args.validation_ood_ratio
	opts.latdim_ctrl = args.latdim_ctrl
	opts.latdim_ptb = args.latdim_ptb
	print('Latent dimension of control cells: ', opts.latdim_ctrl, ' Latent dimension of perturbation: ', opts.latdim_ptb)
	opts.geneset_num = args.geneset_num
	opts.geneset_dim = args.geneset_dim
	print('Gene set number: ', opts.geneset_num, ' Gene set dimension: ', opts.geneset_dim)
	opts.batch_size = args.batch_size
	opts.tolerance_epochs = args.tolerance_epochs
	opts.MMD_sigma = args.MMD_sigma
	opts.reconstruction_loss = args.reconstruction_loss
	opts.mxAlpha = args.mxAlpha
	opts.mxBeta = args.mxBeta
	opts.Gamma1 = args.Gamma1
	opts.Gamma2 = args.Gamma2
	assert 0 <= opts.Gamma1 <= 1, 'Gamma1 should be between 0 and 1'
	assert 0 <= opts.Gamma2 <= 1, 'Gamma2 should be between 0 and 1'
	
	# First check if we want to call mixture of expert model
	if 'moe' in args.model and '3expert' in args.model:
		print('Running mixture of expert model with 3 experts', 'expert_1: ', args.label, ' expert_2: ', args.label_2, ' expert_3: ', args.label_3)
	elif 'moe' in args.model:
		args.label_3 = None
		print('Running mixture of expert model with 2 experts, expert_1: ', args.label, ' expert_2: ', args.label_2)
	else:
		args.label_2 = None
		args.label_3 = None

	# Load data
    # Accepted representation types: 'Geneformer', 'Baseline', 'Rank', 'GO', 'GenePT'
	opts.leave_out_test_set_id = args.leave_out_test_set_id
	opts.modality = args.modality
	opts.dataset = args.dataset
	opts.use_hvg = args.use_hvg
	opts.label = args.label
	opts.label_2 = args.label_2
	opts.label_3 = args.label_3
	opts.null_label = args.null_label
	dataloader, train_idx, dataloader_val, val_idx, dataloader_infer, infer_idx, \
		dim, cdim, cdim_2, cdim_3, ptb_targets, ptb_vector_dict, \
			 ptb_leave_out_list = get_data(opts=opts)
	
	opts.dim = dim #input dimension of the gene expression decoder
	opts.cdim = cdim #input dimension of the perturbation decoder
	opts.cdim_2 = cdim_2 #input dimension of the perturbation decoder for the second expert
	opts.cdim_3 = cdim_3 #input dimension of the perturbation decoder for the third expert
	opts.model = args.model #which model to run

	opts.epochs = args.epochs
	opts.ptb_leave_out_list = ptb_leave_out_list
	print('Running model: ', opts.model)
	
	# list of configs: batch_size, mode, lr, epochs, ...
	with open(f'{args.savedir}/config.json', 'w') as f:
		json.dump(opts.__dict__, f, indent=4)

	with open(f'{args.savedir}/ptb_targets.pkl', 'wb') as f:
		pickle.dump(ptb_targets, f, protocol=pickle.HIGHEST_PROTOCOL)

	if args.label == 'Baseline':
		with open(f'{args.savedir}/ptb_vector_dict.pkl', 'wb') as f:
			pickle.dump(ptb_vector_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	print('Saving split indices...') 
	split_idx = {'train_idx': train_idx, 'val_idx': val_idx, 'infer_idx': infer_idx}
	with open(f'{args.savedir}/split_idx.pkl', 'wb') as f:
		pickle.dump(split_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	print('Finished saving data')
	
	# Train model
	train_validate(dataloader, dataloader_infer, dataloader_val, opts, args.device, args.savedir, args.model, log=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='parse args')
	parser.add_argument('-s', '--savedir', type=str, default='./result/', help='directory to save the results')
	parser.add_argument('--random_seed', type=int, default=12, help='random seed')
	parser.add_argument('--modality', type=str, default='rna', help='modality to use: rna, ops')
	parser.add_argument('--dataset', type=str, default='replogle_k562_essential', help='dataset to use: norman_k562, replogle_rpe1, replogle_k562_essential, replogle_k562_gwps')
	parser.add_argument('--use_hvg', type=str, default='True', help='whether to use highly variable genes for replogle_rpe1 and replogle_k562')
	parser.add_argument('--leave_out_test_set_id', type=str, default='random_fold_5', help='test id to identify list of perturbations to leave out for testing')
	parser.add_argument('--device', type=str, default="cuda:6", help='device to run the training')
	parser.add_argument('--model', type=str, default='MORPH', help='model to run the training')
	parser.add_argument('--label', type=str, default='DepMap_GeneEffect', help='label mode to run the training (Geneformer, Baseline (ont-hot vector), or Rank (use rank vectors as input), or Control_Gene_Expression, or GenePT, or GO or Geneformer_n_GenePT or DepMap_GeneEffect)')
	parser.add_argument('--label_2', type=str, default='GenePT_v1', help='Only applicable when you want to run mixture of expert model')
	parser.add_argument('--label_3', type=str, default='STRING', help='Only applicable when you want to run mixture of expert model')
	parser.add_argument('--null_label', type=str, default='zeros', help='null representation to use for the control (options: zeros, gaussian, gaussian_normalized)')
	parser.add_argument('--epochs', type=int, default=100, help='number of epochs to run the training')
	parser.add_argument('--tolerance_epochs', type=int, default=10, help='number of epochs to tolerate before early stopping')
	parser.add_argument('--batch_size', type=int, default=32, help='batch size')
	parser.add_argument('--MMD_sigma', type=int, default=1500, help='sigma for MMD loss')
	parser.add_argument('--reconstruction_loss', type=str, default='mse', help='reconstruction loss to use (mse, mmd)')
	parser.add_argument('--mxAlpha', type=float, default=10, help='alpha for MMD loss')
	parser.add_argument('--mxBeta', type=float, default=2, help='beta for KL loss')
	parser.add_argument('--Gamma1', type=float, default=0, help='gamma for reconstruction loss')
	parser.add_argument('--Gamma2', type=float, default=0, help='gamma for prediction loss')
	parser.add_argument('--validation_set_ratio', type=float, default=0.1, help='ratio of validation set (*total_training_candidates)')
	parser.add_argument('--validation_ood_ratio', type=float, default=0.15, help='ratio of ood validation set (*validation_set), default: 0.15')
	parser.add_argument('--latdim_ctrl', type=int, default=50, help='latent dimension of control cell encoder')
	parser.add_argument('--latdim_ptb', type=int, default=50, help='latent dimension of ptb representations')
	parser.add_argument('--geneset_num', type=int, default=50, help='number of genesets to use in the model')
	parser.add_argument('--geneset_dim', type=int, default=50, help='dimension of geneset embeddings')
	args = parser.parse_args()
	
	if args.modality == 'rna':
		if args.use_hvg == 'True':
			args.savedir = f"{args.savedir}rna/{args.dataset}_hvg/latdim_ctrl_{args.latdim_ctrl}_latdim_ptb_{args.latdim_ptb}_geneset_num_{args.geneset_num}/{str.lower(args.leave_out_test_set_id)}/recon_loss_{args.reconstruction_loss}/null_label_{args.null_label}/epochs_{str(args.epochs)}/tolerance_epochs_{args.tolerance_epochs}/mxAlpha_{args.mxAlpha}/val_{args.validation_set_ratio}_ood_{args.validation_ood_ratio}/random_seed_{str(args.random_seed)}/{args.label}_{args.model}_run{int(time.time())}"
		else:
			args.savedir = f"{args.savedir}rna/{args.dataset}/latdim_ctrl_{args.latdim_ctrl}_latdim_ptb_{args.latdim_ptb}_geneset_num_{args.geneset_num}/{str.lower(args.leave_out_test_set_id)}/recon_loss_{args.reconstruction_loss}/null_label_{args.null_label}/epochs_{str(args.epochs)}/tolerance_epochs_{args.tolerance_epochs}/mxAlpha_{args.mxAlpha}/val_{args.validation_set_ratio}_ood_{args.validation_ood_ratio}/random_seed_{str(args.random_seed)}/{args.label}_{args.model}'_run{int(time.time())}"
	elif args.modality == 'ops':
		args.savedir = f"{args.savedir}ops/latdim_ctrl_{args.latdim_ctrl}_latdim_ptb_{args.latdim_ptb}_geneset_num_{args.geneset_num}/{str.lower(args.leave_out_test_set_id)}/{args.dataset}/recon_loss_{args.reconstruction_loss}/null_label_{args.null_label}/epochs_{str(args.epochs)}/tolerance_epochs_{args.tolerance_epochs}/mxAlpha_{args.mxAlpha}/val_{args.validation_set_ratio}_ood_{args.validation_ood_ratio}/random_seed_{str(args.random_seed)}/{args.label}_{args.model}_run{int(time.time())}"
	else:
		raise ValueError('Invalid modality')
	
	print('Will save results to: ', args.savedir)
	if not os.path.exists(args.savedir):
		os.makedirs(args.savedir)
	
	main(args)