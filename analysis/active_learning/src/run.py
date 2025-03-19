from argparse import ArgumentParser, Namespace
import torch
from helpers import generate_decreasing_schedule
import os
import json
from data import get_data
from train import train
from train_warmup import train_warmup_model
from acquisition import query_sample_to_label
import pickle
import random
import numpy as np

def main(args):
    print(f'using device: {args.device}')
    torch.set_num_threads(6)

    opts = Namespace(**vars(args))
    random_seed = opts.random_seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    print('dataset_name:', opts.dataset_name)
    print('representation_type:', opts.representation_type)
    print('null_label:', opts.null_label)
    print('leave_out_test_set_id:', opts.leave_out_test_set_id)
    print('method:', opts.method)
    print('use_prior:', opts.use_prior)
    print('prior_only:', opts.prior_only)
    print('kernel_type:', opts.kernel_type)
    print('kernel_gamma:', opts.kernel_gamma)
    print('val_ratio:', opts.val_ratio)
    print('train_style:', opts.train_style)

    # specify the parameters here -----
    output_parent_dir = f'{opts.base_dir}/results/'+opts.dataset_name+'/'+ \
        opts.leave_out_test_set_id+'/'+opts.representation_type+ '/' + \
            str(opts.num_round)+'_round_'+str(opts.num_ptb)+'_ptb_' + str(opts.initial_num_ptb) + '_initial_ptb/' + "seed_"+str(opts.random_seed)+'/' 
    print('output_dir:', output_parent_dir)
    opts.output_parent_dir = output_parent_dir

    warmup_model_path = f'{opts.base_dir}/results/'+opts.dataset_name+'/'+ \
        opts.leave_out_test_set_id+'/'+opts.representation_type+'/'+ \
            str(opts.initial_num_ptb) + '_initial_ptb/'
    if opts.cluster_first == 'True':
        raise NotImplementedError('Cluster first not implemented for warm-up model')
    else:
        warmup_model_path += 'random/'
    warmup_model_path += 'seed_'+str(opts.random_seed)+'/'
    opts.warmup_model_path = warmup_model_path
    print('warmup_model_path:', warmup_model_path)

    # create output_dir
    method_name_whole = opts.method
    if opts.use_prior == 'False':
        pass  # No change needed to make to method name
    else:
        method_name_whole += '_prior'
        if opts.prior_only == 'True':
            method_name_whole += '_only'
    if opts.cluster_first == 'True':
        method_name_whole += '_cluster_first'

    output_dir = os.path.join(output_parent_dir, method_name_whole)
    print('output_dir:', output_dir)
    opts.output_dir = output_dir

    # if the output directory does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(warmup_model_path):
        os.makedirs(warmup_model_path)

    # Generate the training schedule
    if opts.train_style == 'learn_from_scratch':
        assert opts.decay_factor == 1.0, 'Decay factor must be 1.0 for learn_from_scratch'
        assert opts.initial_epochs == opts.incremental_epochs, 'Initial epochs must be equal to incremental epochs for learn_from_scratch'
    training_schedule = generate_decreasing_schedule(opts.initial_epochs, 
                                                     opts.incremental_epochs, 
                                                     opts.num_round, 
                                                     opts.decay_factor)

    # Calculate total epochs
    total_epochs = training_schedule[-1]
    print('Training schedule:', training_schedule)
    print('Total epochs:', total_epochs)

    total_samples = opts.initial_num_ptb + (opts.num_round-1) * opts.num_ptb
    print('Total samples:', total_samples)

    # read in json file config
    with open(f'{opts.base_dir}/config.json') as f:
        config = json.load(f)
    config = Namespace(**config)

    opts.mxAlpha = config.mxAlpha
    opts.mxBeta = config.mxBeta
    opts.total_samples = total_samples
    opts.output_dir = output_dir
    opts.total_epochs = total_epochs
    opts.training_schedule = training_schedule
    opts.lr = config.lr
    opts.grad_clip = config.grad_clip
    opts.MMD_sigma = config.MMD_sigma
    opts.kernel_num = config.kernel_num
    opts.matched_IO = config.matched_IO
    opts.latdim_ctrl = config.latdim_ctrl
    opts.latdim_ptb = config.latdim_ptb
    opts.geneset_num = config.geneset_num
    opts.geneset_dim = config.geneset_dim
    opts.hidden_dim = config.hidden_dim

    # Get warm-up data -----------------------------------------------------
    dataset, dataloader_train, dataloader_val, dataloader_infer, dataloader_all, gene_emb_raw, train_list_1, val_list_1, opts = get_data(opts)
    opts.dim = dataset.ctrl_samples.shape[1]
    if 'ETS2' in gene_emb_raw:
        assert gene_emb_raw['ETS2'].shape[0] == dataset.ptb_vectors_1.shape[1]
    opts.cdim = dataset.ptb_vectors_1.shape[1]
    print('dim:', opts.dim)
    print('cdim:', opts.cdim)

    ## save dataloader_all if dataloader_all does not exist for the current dataset
    all_parent_dir = f'{opts.base_dir}/results/'+\
        opts.dataset_name+'/all_dataloader.pkl'
    if not os.path.exists(all_parent_dir):
        with open(all_parent_dir, 'wb') as f:
            pickle.dump(dataloader_all, f)
        print('Saved dataloader_all to:', all_parent_dir)
    else:
        print('dataloader_all already exists:', all_parent_dir)
    del dataloader_all
    
    ## save dataloader_infer if dataloader_infer does not exist for the current dataset and test_set_id
    test_parent_dir = f'{opts.base_dir}/results/'+\
        opts.dataset_name+'/'+ \
        opts.leave_out_test_set_id + '/infer_dataloader.pkl'
    if not os.path.exists(test_parent_dir):
        with open(test_parent_dir, 'wb') as f:
            pickle.dump(dataloader_infer, f)
        print('Saved dataloader_infer to:', test_parent_dir)
    else:
        print('dataloader_infer already exists:', test_parent_dir)
    
    # Start training -----------------------------------------------------
    opts.model = 'MORPH'
    
    ## loss parameters
    mxAlpha = opts.mxAlpha
    mxBeta = opts.mxBeta
    if opts.train_style == 'tune_one_whole':
        beta_schedule = torch.zeros(opts.total_epochs) # weight on the KLD
        beta_schedule[:10] = 0
        beta_schedule[10:] = torch.linspace(0, mxBeta,opts.total_epochs-10) 
        alpha_schedule = torch.zeros(opts.total_epochs) # weight on the MMD
        alpha_schedule[:] = mxAlpha
        alpha_schedule[:5] = 0
        alpha_schedule[5:int(opts.total_epochs/2)] = torch.linspace(0,mxAlpha,int(opts.total_epochs/2)-5) 
        alpha_schedule[int(opts.total_epochs/2):] = mxAlpha
    elif opts.train_style == 'learn_from_scratch':
        # create the schedule for each round
        beta_schedule_temp = torch.zeros(opts.initial_epochs)
        beta_schedule_temp[:10] = 0
        beta_schedule_temp[10:] = torch.linspace(0, mxBeta, opts.initial_epochs-10)
        alpha_schedule_temp = torch.zeros(opts.initial_epochs)
        alpha_schedule_temp[:] = mxAlpha
        alpha_schedule_temp[:5] = 0
        alpha_schedule_temp[5:int(opts.initial_epochs/2)] = torch.linspace(0, mxAlpha, int(opts.initial_epochs/2)-5)
        alpha_schedule_temp[int(opts.initial_epochs/2):] = mxAlpha

        # duplicate the schedule for each round
        beta_schedule = torch.cat([beta_schedule_temp] * opts.num_round)
        alpha_schedule = torch.cat([alpha_schedule_temp] * opts.num_round)

    opts.alpha_schedule = alpha_schedule.tolist()
    opts.beta_schedule = beta_schedule.tolist()

    ## train warm-up model if needed
    model_1_path = opts.warmup_model_path + 'round_0_last_model.pt'
    if os.path.exists(model_1_path):
        print('Model 1 already trained')
    else:
        if os.path.exists(opts.warmup_model_path + 'round_0_best_train_model.pt'):
            raise ValueError('Warm-up model exists but last model does not exist')
        train_warmup_model(opts=opts, 
                           dataloader_train=dataloader_train, 
                           dataloader_val=dataloader_val, 
                           dataloader_infer=dataloader_infer, 
                           gene_emb_raw=gene_emb_raw, 
                           train_list_1=train_list_1, 
                           val_list_1=val_list_1,
                           alpha_schedule=alpha_schedule, 
                           beta_schedule=beta_schedule,
                           log=True
        )
    
    ## save a copy of warm-up models to the method folder
    os.system('cp '+opts.warmup_model_path+'round_0_best_train_model.pt '+opts.output_dir+'/round_0_best_train_model.pt')
    os.system('cp '+opts.warmup_model_path+'round_0_best_val_model.pt '+opts.output_dir+'/round_0_best_val_model.pt')
    os.system('cp '+opts.warmup_model_path+'round_0_last_model.pt '+opts.output_dir+'/round_0_last_model.pt')
    
    ## query samples to label for second round
    with open(opts.warmup_model_path + 'train_list_1.pkl', 'rb') as f:
        train_list_1 = pickle.load(f)
    with open(opts.warmup_model_path + 'val_list_1.pkl', 'rb') as f:
        val_list_1 = pickle.load(f)
    train_list_current = train_list_1
    train_list_current.extend(val_list_1)
    assert len(train_list_current) == opts.initial_num_ptb
    with open(opts.warmup_model_path + 'gene_latent_df.pkl', 'rb') as f:
        gene_latent_df = pickle.load(f)
    with open(opts.warmup_model_path + 'gene_distance_df.pkl', 'rb') as f:
        gene_distance_df = pickle.load(f)
    
    del dataloader_train, dataloader_val
    train_list_temp, train_list_current, \
        dataloader_train, dataloader_val = query_sample_to_label(opts, dataset,
                                                                 train_list_current, gene_latent_df, gene_distance_df, 
                                                                 round=1, log=False) # Must set log to False here
    
    ## train the following rounds
    train(opts, dataset, 
          dataloader_train, dataloader_val, dataloader_infer, 
          gene_emb_raw, gene_latent_df, gene_distance_df,
          train_list_temp, train_list_current,
          alpha_schedule, beta_schedule,
          log = True)



if __name__ == "__main__":
    parser = ArgumentParser(description="Run experiments")
    parser.add_argument('--random_seed', type=int, default=12, help="Random seed (change this for different runs)")
    parser.add_argument('--dataset_name', type=str, required=True, default='replogle_k562_essential_hvg', help="Name of the dataset")
    parser.add_argument('--representation_type', type=str, default='DepMap_GeneEffect', help='Type of representation to use')
    parser.add_argument('--Gamma1', type=float, default=1, help='gamma for reconstruction loss (MMD)')
    parser.add_argument('--Gamma2', type=float, default=0, help='gamma for reconstruction loss (MSE)')
    parser.add_argument('--null_label', type=str, default='zeros', help='Null label')
    parser.add_argument('--leave_out_test_set_id', type=str, required=True, help="ID of the test set to leave out")
    parser.add_argument('--method', type=str, required=True, help="Method to use: 'baseline', 'active_learning_kmeans_center', 'learn_loss_mse', or 'uncertainty' ")
    
    parser.add_argument('--use_prior', type=str, default='False', help="Whether to use prior (default: False), Prior fusion: mean")
    parser.add_argument('--prior_only', type=str, default='False', help="Whether to use prior only (default: False)")
    
    parser.add_argument('--kernel_type', type=str, default='', help="Type of kernel to use (default: rbf)")
    parser.add_argument('--kernel_gamma', type=float, default=1.0, help="Gamma value for the kernel (default: 1.0)")

    parser.add_argument('--cluster_first', type=str, default='False', help="Whether to cluster first (default: False)")
    parser.add_argument('--device', type=str, default='cuda:5', help="Device to use (default: cuda:5)")

    parser.add_argument('--num_round', type=int, default=5, help="Number of rounds")
    parser.add_argument('--initial_num_ptb', type=int, default=100, help="Number of initial perturbations to sample (for the first round)")
    parser.add_argument('--num_ptb', type=int, default=100, help="Number of total perturbations to sample each time")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")

    parser.add_argument('--initial_epochs', type=int, default=100, help="Number of initial epochs")
    parser.add_argument('--incremental_epochs', type=int, default=100, help="Number of incremental epochs")
    parser.add_argument('--decay_factor', type=float, default=1.0, help="Decay factor")
    parser.add_argument('--tolerance_epochs', type=int, default=20, help="Tolerance epochs")

    parser.add_argument('--val_ratio', type=float, default=0.1, help="Validation ratio")

    parser.add_argument('--train_style', type=str, default='learn_from_scratch', help="Training style: 'learn_from_scratch'")

    args = parser.parse_args()

    args.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.path_dir = os.path.abspath(os.path.join(args.base_dir, '..', '..'))

    main(args)
    
