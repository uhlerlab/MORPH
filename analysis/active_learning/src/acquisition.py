import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from helpers import plot_umap_by_selections
from lossnet import LossNet
import torch.nn.functional as F
import torch.nn as nn
import torch
import wandb
import sys
from tqdm import tqdm
import os
from loss import loss_function

current_dir = os.path.dirname(os.path.abspath(__file__))
morph_path = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'morph'))
sys.path.append(morph_path)

from utils import SCDATA_sampler
from torch.utils.data import DataLoader, Subset

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max to prevent overflow
    return exp_x / exp_x.sum()

def query_sample_to_label(opts, dataset,
                          train_list_current, gene_latent_df, gene_distance_df, 
                          round, log=False):
    # call acquisition function to query new samples to label
    if opts.method == 'baseline':
        training_candidates_with_embeddings = list(set(opts.training_candidates_with_embeddings) - set(train_list_current))
        np.random.shuffle(training_candidates_with_embeddings)
        train_list_temp = training_candidates_with_embeddings[:opts.num_ptb]
    else:
        # add a column in gene_latent_df called 'train_indicator', set to 1 if gene_name in train_list_current, 0 otherwise
        gene_latent_df['train_indicator'] = gene_latent_df['gene_name'].apply(lambda x: 1 if x in train_list_current else 0)
        assert gene_latent_df['train_indicator'].sum() == len(train_list_current)
        if opts.method == 'active_learning_kmeans_center':
            train_list_temp = kmeans_center_select(train_list_current, opts, gene_latent_df, round, weight_df=None)
        elif opts.method in ['kmeans_center_weight_uncertainty', 
                                'kmeans_center_weight_uncertainty_min_max', 'kmeans_center_weight_uncertainty_softmax',
                                'kmeans_center_weight_uncertainty_scaled',
                                'uncertainty', 'uncertainty_random']:
            gene_distance_df['weight'] = gene_distance_df['distance'] / gene_distance_df['epoch']
            if log:
                min_value = gene_distance_df['weight'].min()
                max_value = gene_distance_df['weight'].max()
                wandb.log({"uncertainty_weight_min": min_value, "uncertainty_weight_max": max_value})
            # check not all weights are 0
            assert gene_distance_df['weight'].sum() > 0
            # reset distance and epoch
            gene_distance_df['distance'] = 0
            gene_distance_df['epoch'] = 0
            if opts.method in ['kmeans_center_weight_uncertainty', 'kmeans_center_weight_uncertainty_min_max', 'kmeans_center_weight_uncertainty_softmax' ,'kmeans_center_weight_uncertainty_scaled']:
                train_list_temp = kmeans_center_select(train_list_current, opts, gene_latent_df, round, weight_df=gene_distance_df[['gene_name', 'weight']])
            elif opts.method == 'uncertainty':
                train_list_temp = uncertainty_select(train_list_current, opts, gene_distance_df, random=False)
            elif opts.method == 'uncertainty_random':
                train_list_temp = uncertainty_select(train_list_current, opts, gene_distance_df, random=True)
        elif opts.method in ['learn_loss_mse', 'learn_loss_ce']:
            train_list_temp = learn_loss_select(dataset, train_list_current, opts, gene_latent_df, round, log=log)

    # check no overlapping with train_list_current
    print('new samples to label', train_list_temp)
    print('length of new training set', len(train_list_temp))
    assert len(set(train_list_temp) & set(train_list_current)) == 0, 'overlap between train_list_temp and train_list_current'
    train_list_current.extend(train_list_temp)

    # random sample 10% of the training set for validation
    val_list = np.random.choice(train_list_current, int(len(train_list_current)*opts.val_ratio), replace=False)
    train_list = list(set(train_list_current) - set(val_list))
    assert set(train_list) & set(val_list) == set()

    train_idx_temp = np.where(np.isin(dataset.ptb_names, train_list))[0].tolist()
    val_idx_temp = np.where(np.isin(dataset.ptb_names, val_list))[0].tolist()
    assert set(train_idx_temp) & set(val_idx_temp) == set()
    print('length of updated training set', len(train_list_current))
    print('length of train_idx_temp',len(train_idx_temp))
    print('length of val_idx_temp',len(val_idx_temp))
    
    dataset_train_temp = Subset(dataset, train_idx_temp)
    ptb_name = dataset.ptb_names[train_idx_temp]
    dataloader_train = DataLoader(
        dataset_train_temp,
        batch_sampler=SCDATA_sampler(dataset_train_temp, opts.batch_size, ptb_name),
        num_workers = 0
    )
    print('new training dataloader created')

    dataset_val_temp = Subset(dataset, val_idx_temp)
    ptb_name = dataset.ptb_names[val_idx_temp]
    dataloader_val = DataLoader(
        dataset_val_temp,
        batch_sampler=SCDATA_sampler(dataset_val_temp, opts.batch_size, ptb_name),
        num_workers = 0
    )
    print('new validation dataloader created')
    
    # plot umap of selected genes
    plot_umap_by_selections(round, opts, gene_latent_df, train_list_current, train_list_temp)
    return train_list_temp, train_list_current, dataloader_train, dataloader_val

def kmeans_center_select(train_list_current, 
                         opts, 
                         gene_latent_df, 
                         round,
                         weight_df):
    
    # This method nominates samples by returning the closest sample to each center of the unlabeled data clusters
    exclude_indices = set(train_list_current + opts.ptb_leave_out_list)
    if opts.use_groundtruth_matrix == 'False':
        filtered_gene_latent_df = gene_latent_df[~gene_latent_df['gene_name'].isin(exclude_indices)]
        if opts.use_prior == 'False':
            gene_emb_ft_array = np.stack(filtered_gene_latent_df['gene_emb_ft'].values)
        else:
            if opts.prior_only == 'True':
                gene_emb_array_raw = np.stack(filtered_gene_latent_df['gene_emb'].values)
                # pca gene_emb_array lower dimensions
                pca = PCA(n_components=50, random_state=opts.random_seed)
                gene_emb_array = pca.fit_transform(gene_emb_array_raw)
                gene_emb_ft_array = gene_emb_array
            else:
                raise NotImplementedError('Fuse latent with prior knowledge using mean for active learning')
                # print('Fuse latent with prior knowledge using mean for active learning')
                # gene_emb_ft_array_raw = np.stack(filtered_gene_latent_df['gene_emb_ft'].values)
                # gene_emb_array_raw = np.stack(filtered_gene_latent_df['gene_emb'].values)
                # # pca gene_emb_array to the same dimensions
                # pca = PCA(n_components=gene_emb_ft_array_raw.shape[1], random_state=opts.random_seed)
                # gene_emb_array = pca.fit_transform(gene_emb_array_raw)
                # # min-max normalize gene_emb_array
                # gene_emb_array = MinMaxScaler().fit_transform(gene_emb_array)
                # gene_emb_ft_array_raw = MinMaxScaler().fit_transform(gene_emb_ft_array_raw)
                # # concatenate gene_emb_ft_array and gene_emb_array
                # gene_emb_ft_array = np.concatenate((gene_emb_ft_array_raw, gene_emb_array), axis=1)
        
        # Perform KMeans++ clustering
        kmeans = KMeans(n_clusters=opts.num_ptb, init='k-means++', random_state=opts.random_seed)
        if weight_df is None:
            kmeans.fit(gene_emb_ft_array)
        else:
            weight_df.set_index('gene_name', inplace=True)
            weight_df = weight_df.loc[filtered_gene_latent_df['gene_name']]
            weight = weight_df['weight'].values
            if opts.method == 'kmeans_center_weight_uncertainty_min_max':
                # Normalize the weights to [0, 1]
                weight = MinMaxScaler().fit_transform(weight.reshape(-1, 1)).flatten()
            elif opts.method == 'kmeans_center_weight_uncertainty_softmax':
                weight = softmax(weight)
            elif opts.method == 'kmeans_center_weight_uncertainty_scaled':
                weight = softmax(weight * 5)
            kmeans.fit(gene_emb_ft_array, sample_weight=weight)
        
        # Find the closest sample to each center
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, gene_emb_ft_array)
        
        # Ensure there are no duplicates by finding unique closest samples
        if len(set(closest)) < opts.num_ptb:
            unique_closest = set()
            for center_idx in range(opts.num_ptb):
                closest_idx = closest[center_idx]
                if closest_idx in unique_closest:
                    # Find the next closest sample that is not already selected
                    center_distances = pairwise_distances(kmeans.cluster_centers_[center_idx].reshape(1, -1), gene_emb_ft_array).flatten()
                    sorted_indices = np.argsort(center_distances)
                    for idx in sorted_indices:
                        if idx not in unique_closest:
                            closest_idx = idx
                            break
                unique_closest.add(closest_idx)
        else:
            unique_closest = set(closest)
        
        # Get the closest samples
        closest_samples = filtered_gene_latent_df.iloc[list(unique_closest)]
        train_list_temp = closest_samples['gene_name'].tolist()

        # let's plot the UMAP of the latent space, color by k-means cluster and mark the selected ones
        reducer = umap.UMAP(random_state=opts.random_seed)
        umap_embedding = reducer.fit_transform(gene_emb_ft_array)

        # Plot UMAP with KMeans clusters
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], c=kmeans.labels_, cmap='viridis', s=50)
        plt.colorbar(scatter, label='Cluster Label')

        # Mark the unique closest samples
        unique_closest_indices = list(unique_closest)
        plt.scatter(umap_embedding[unique_closest_indices, 0], umap_embedding[unique_closest_indices, 1], c='red', edgecolors='black', s=100, label='Selected Samples')

        # Add labels and title
        plt.title('UMAP of Gene Embedding with KMeans Clustering and Selected Samples')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.savefig(opts.output_dir + '/umap_latent_round_' + str(round) + '.png')
        plt.show()
        
        assert len(set(train_list_temp)) == opts.num_ptb
        assert set(train_list_temp) - set(opts.all_ptb_targets_with_embeddings) == set(), 'train_list_temp contains genes not in all_ptb_targets_with_embeddings'
        assert len(set(train_list_temp) & set(train_list_current)) == 0, 'overlap between train_list_temp and train_list_current'
        assert len(set(train_list_temp) & set(opts.ptb_leave_out_list)) == 0, 'overlap between train_list_temp and ptb_leave_out_list'

        return train_list_temp
    
    else:
        raise NotImplementedError('Groundtruth matrix is not supported for active learning')
    
def uncertainty_select(train_list_current, 
                opts, 
                uncertainty_df,
                random,
                scale_factor=10):
    
    # This method nominates samples by returning the samples with the highest uncertainty
    # random - if True, randomly select samples (weighted by uncertainty), if False, select samples with highest uncertainty
    # scale - if True, scale the uncertainty values to [0, 1], if False, use raw uncertainty values

    # This method nominates samples by returning the closest sample to each center of the unlabeled data clusters
    exclude_indices = set(train_list_current + opts.ptb_leave_out_list)
    uncertainty_df = uncertainty_df[~uncertainty_df['gene_name'].isin(exclude_indices)]

    if random:
        # scale weights to probabilities
        uncertainty_df['prob'] = softmax(uncertainty_df['weight'].values * scale_factor)
        # sample based on uncertainty
        sampled_indices = np.random.choice(uncertainty_df.index, size=opts.num_ptb, replace=False, p=uncertainty_df['prob'].values)
        sampled_df = uncertainty_df.loc[sampled_indices]
        train_list_temp = sampled_df['gene_name'].tolist()
    else:
        # sort by uncertainty
        uncertainty_df = uncertainty_df.sort_values(by='weight', ascending=False)
        uncertainty_df = uncertainty_df.head(opts.num_ptb)
        # label samples with highest uncertainty
        train_list_temp = uncertainty_df['gene_name'].tolist()
        
    assert len(set(train_list_temp)) == opts.num_ptb
    assert set(train_list_temp) - set(opts.all_ptb_targets_with_embeddings) == set(), 'train_list_temp contains genes not in all_ptb_targets_with_embeddings'
    assert len(set(train_list_temp) & set(train_list_current)) == 0, 'overlap between train_list_temp and train_list_current'
    assert len(set(train_list_temp) & set(opts.ptb_leave_out_list)) == 0, 'overlap between train_list_temp and ptb_leave_out_list'

    return train_list_temp

def learn_loss_select(dataset, train_list_current, opts, gene_latent_df, round, log=False):

    if opts.method == 'learn_loss_mse':
        pred_loss_fn = nn.MSELoss()
    elif opts.method == 'learn_loss_ce':
        pred_loss_fn = nn.CrossEntropyLoss()

    # load in the last model from the training
    if opts.train_style == 'learn_from_scratch':
        model_path = opts.output_dir+'/round_' + str(round-1) + '_best_val_model.pt'
    elif opts.train_style == 'tune_on_whole':
        model_path = opts.output_dir+'/round_' + str(round-1) + '_last_model.pt'
    main_model = torch.load(model_path, map_location=opts.device)
    print('main model loaded from', model_path)
    main_model.eval()

    if opts.use_prior == 'True' and opts.prior_only == 'True':
        print('Will use prior embeddings only for active learning')
        loss_model_input_dim = opts.cdim
    elif opts.use_prior == 'False' and opts.prior_only == 'False':
        loss_model_input_dim = opts.latdim_ptb
    else:
        raise NotImplementedError('Parameter combination not supported for active learning')
    
    # update the embedding to be those from the correct model
    fine_tuned_embeddings = []
    for _, row in tqdm(gene_latent_df.iterrows(), total=len(gene_latent_df), desc='Get fine-tuned gene embeddings'):
        type = row['type']
        if type == 'single':
            gene_emb = torch.tensor(row['gene_emb'], dtype=torch.float64).to(opts.device)
            gene_emb = gene_emb.repeat(opts.batch_size, 1) # make batch_size gene_emb 
            with torch.no_grad():
                gene_emb_ft = main_model.c_encode(gene_emb) # Assuming c_encode expects a batch dimension
            fine_tuned_embeddings.append(gene_emb_ft[0].detach().cpu().numpy())
    gene_latent_df['gene_emb_ft'] = fine_tuned_embeddings
    
    # random sample 15% of the training set for validation
    val_list = np.random.choice(train_list_current, int(len(train_list_current)*0.15), replace=False)
    train_list = list(set(train_list_current) - set(val_list))

    # Create a dataloader based on the current training set
    train_idx = np.where(np.isin(dataset.ptb_names, train_list))[0].tolist()
    dataset_train_temp = Subset(dataset, train_idx)
    dataloader_train = DataLoader(
        dataset_train_temp,
        batch_sampler=SCDATA_sampler(dataset_train_temp, opts.batch_size, dataset.ptb_names[train_idx]),
        num_workers = 0
    )

    val_idx = np.where(np.isin(dataset.ptb_names, val_list))[0].tolist()
    dataset_val_temp = Subset(dataset, val_idx)
    dataloader_val = DataLoader(
        dataset_val_temp,
        batch_sampler=SCDATA_sampler(dataset_val_temp, opts.batch_size, dataset.ptb_names[val_idx]),
        num_workers = 0
    )

    # Create the loss prediction model
    if opts.method == 'learn_loss_mse':
        loss_model = LossNet(dim=loss_model_input_dim, output_dim=1).to(opts.device)
        loss_optimizer = torch.optim.Adam(loss_model.parameters(), lr=opts.lr*20)
    elif opts.method == 'learn_loss_ce':
        num_bins = 15
        loss_model = LossNet(dim=loss_model_input_dim, output_dim=num_bins).to(opts.device)
        loss_optimizer = torch.optim.Adam(loss_model.parameters(), lr=opts.lr*10)

    loss_model.double()
    loss_model.train()
    # Now, loop through the dataloader and train the loss prediction model
    min_loss = np.inf
    max_loss = -np.inf
    min_val_loss = np.inf
    min_train_loss = np.inf
    for epoch in range(51):

        # Summary statistics
        train_lossAv = 0
        train_ct = 0
        val_lossAv = 0
        val_ct = 0

        # Train the loss prediction model
        for (i, X) in enumerate(dataloader_train):
            x = X[0].to(opts.device)
            y = X[1].to(opts.device)
            c_1 = X[2].to(opts.device)
            c_2 = X[3].to(opts.device)
            ptb_name = X[4][0]

            with torch.no_grad():
                y_hat, x_recon, z_mu, z_logvar, z_ptb = main_model(x, c_1, c_2, return_latents=True)
            z_ptb = z_ptb.detach() # Detach to prevent gradients from flowing back to the main model
            assert np.array_equal(gene_latent_df[gene_latent_df['gene_name'] == ptb_name]['gene_emb_ft'].values[0], z_ptb[0].cpu().numpy())

            main_loss, _, _ = loss_function(y_hat=y_hat, y=y,
                                            x_recon=x_recon, x=x,
                                            mu=z_mu, logvar=z_logvar,
                                            MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num,
                                            gamma1=1, gamma2=0)
            
            main_loss = main_loss.item()
            if epoch == 0:
                # Use first epoch to set min and max loss
                if main_loss < min_loss:
                    min_loss = main_loss
                if main_loss > max_loss:
                    max_loss = main_loss
                if log:
                    wandb.log({"learn_loss_min": min_loss, "learn_loss_max": max_loss})
            else:
                if opts.method == 'learn_loss_mse':
                    main_loss = torch.tensor(main_loss).expand(z_ptb.shape[0]).to(opts.device).double()  # Expand to match batch size
                elif opts.method == 'learn_loss_ce':
                    # Bin main_loss into bins from min_loss to max_loss
                    bins = torch.linspace(min_loss, max_loss, num_bins + 1).to(opts.device)  # Bin edges
                    bin_idx = torch.bucketize(main_loss, bins, right=False) - 1  # Adjust for 0-based index
                    bin_idx = torch.clamp(bin_idx, min=0, max=num_bins - 1)  # Ensure indices are valid
                    bin_idx = bin_idx.expand(z_ptb.shape[0])  # Expand to match batch size
                    
                # Now train the loss prediction model to predict loss
                loss_model.train()
                loss_optimizer.zero_grad()

                if opts.use_prior == 'True' and opts.prior_only == 'True':
                    c_1 = c_1.detach()
                    loss_hat = loss_model(c_1)
                elif opts.use_prior == 'False' and opts.prior_only == 'False':
                    loss_hat = loss_model(z_ptb)

                if opts.method == 'learn_loss_mse':
                    loss = pred_loss_fn(loss_hat, main_loss)
                elif opts.method == 'learn_loss_ce':
                    loss = pred_loss_fn(loss_hat, bin_idx)
                loss.backward()
                loss_optimizer.step()

                train_lossAv += loss.item()
                train_ct += 1

                if log:
                    wandb.log({"train_learn_loss_loss": loss.item()})

        for (i, X) in enumerate(dataloader_val):
            x = X[0].to(opts.device)
            y = X[1].to(opts.device)
            c_1 = X[2].to(opts.device)
            c_2 = X[3].to(opts.device)
            ptb_name = X[4][0]

            with torch.no_grad():
                y_hat, x_recon, z_mu, z_logvar, z_ptb = main_model(x, c_1, c_2, return_latents=True)
            z_ptb = z_ptb.detach() # Detach to prevent gradients from flowing back to the main model
            c_1 = c_1.detach()
            assert np.array_equal(gene_latent_df[gene_latent_df['gene_name'] == ptb_name]['gene_emb_ft'].values[0], z_ptb[0].cpu().numpy())

            main_loss,  _, _ = loss_function(y_hat=y_hat, y=y,
                                             x_recon=x_recon, x=x,
                                             mu=z_mu, logvar=z_logvar,
                                             MMD_sigma=opts.MMD_sigma, kernel_num=opts.kernel_num,
                                             gamma1=1, gamma2=0)
            main_loss = main_loss.item()
            if epoch == 0:
                # Use first epoch to set min and max loss
                if main_loss < min_loss:
                    min_loss = main_loss
                if main_loss > max_loss:
                    max_loss = main_loss
                if log:
                    wandb.log({"learn_loss_min": min_loss, "learn_loss_max": max_loss})
            else:
                if opts.method == 'learn_loss_mse':
                    main_loss = torch.tensor(main_loss).expand(z_ptb.shape[0]).to(opts.device).double()  # Expand to match batch size
                elif opts.method == 'learn_loss_ce':
                    # Bin main_loss into bins from min_loss to max_loss
                    bins = torch.linspace(min_loss, max_loss, num_bins + 1).to(opts.device)  # Bin edges
                    bin_idx = torch.bucketize(main_loss, bins, right=False) - 1  # Adjust for 0-based index
                    bin_idx = torch.clamp(bin_idx, min=0, max=num_bins - 1)  # Ensure indices are valid
                    bin_idx = bin_idx.expand(z_ptb.shape[0])  # Expand to match batch size
                    
                # Now train the loss prediction model to predict loss
                loss_model.eval()

                if opts.use_prior == 'True' and opts.prior_only == 'True':
                    with torch.no_grad():
                        loss_hat = loss_model(c_1)
                elif opts.use_prior == 'False' and opts.prior_only == 'False':
                    with torch.no_grad():
                        loss_hat = loss_model(z_ptb)
                
                if opts.method == 'learn_loss_mse':
                    val_loss = pred_loss_fn(loss_hat, main_loss)
                elif opts.method == 'learn_loss_ce':
                    val_loss = pred_loss_fn(loss_hat, bin_idx)
                val_lossAv += val_loss.item()
                val_ct += 1
                if log:
                    wandb.log({"val_learn_loss_loss": val_loss.item()})

        if epoch > 0:
            print('Epoch:', epoch, 'Train avg loss:', train_lossAv / train_ct, 'Val avg loss:', val_lossAv / val_ct)
            if log:
                wandb.log({"learn_loss_epoch": epoch, 
                           "train_learn_loss_lossAv": train_lossAv / train_ct,
                           "val_learn_loss_lossAv": val_lossAv / val_ct})
            if val_lossAv / val_ct < min_val_loss:
                min_val_loss = val_lossAv / val_ct
                torch.save(loss_model, opts.output_dir + '/round_' + str(round-1) + '_best_val_learn_loss_model.pt')
                print('best val model saved at epoch', epoch)
                if log:
                    wandb.log({"best_val_learn_loss_loss": min_val_loss})
                    wandb.log({"best_val_learn_loss_epoch": epoch})
            if train_lossAv / train_ct < min_train_loss:
                min_train_loss = train_lossAv / train_ct
                torch.save(loss_model, opts.output_dir + '/round_' + str(round-1) + '_best_train_learn_loss_model.pt')
                print('best train model saved at epoch', epoch)
                if log:
                    wandb.log({"best_train_learn_loss_loss": min_train_loss})
                    wandb.log({"best_train_learn_loss_epoch": epoch})

    # Now, use the loss prediction model to predict the loss for all perturbations to select the ones with the highest loss
    loss_model = torch.load(opts.output_dir + '/round_' + str(round-1) + '_best_val_learn_loss_model.pt', map_location=opts.device)
    loss_model.eval()

    # Get the latent embeddings for all perturbations
    pool_set_df = gene_latent_df[~gene_latent_df['gene_name'].isin(train_list_current + opts.ptb_leave_out_list)]

    if opts.use_prior == 'True' and opts.prior_only == 'True':
        pool_set = np.stack(pool_set_df['gene_emb'].values)
    elif opts.use_prior == 'False' and opts.prior_only == 'False':
        pool_set = np.stack(pool_set_df['gene_emb_ft'].values)
    pool_set = torch.tensor(pool_set).to(opts.device)

    if opts.method == 'learn_loss_mse':
        with torch.no_grad():
            predicted_loss = loss_model(pool_set)
    elif opts.method == 'learn_loss_ce':
        with torch.no_grad():
            logits = loss_model(pool_set)
            predicted_loss = torch.argmax(logits, dim=1)  # Get the index of the highest logit for each sample
    predicted_loss = predicted_loss.cpu().numpy()
    pool_set_df['predicted_loss'] = predicted_loss
    
    if opts.method == 'learn_loss_mse':
        pool_set_df = pool_set_df.sort_values(by='predicted_loss', ascending=False)
        train_list_temp = pool_set_df['gene_name'].head(opts.num_ptb).tolist()
    elif opts.method == 'learn_loss_ce':
        pool_set_df = pool_set_df.sort_values(by='predicted_loss', ascending=False)
        pool_set_df = pool_set_df[['gene_name', 'predicted_loss']]
        # shuffle the samples with the same predicted loss
        shuffled_df = (
            pool_set_df
            .groupby('predicted_loss', group_keys=False)
            .apply(lambda group: group.sample(frac=1, random_state=42))  # Shuffle each group
            .sort_values(by='predicted_loss', ascending=False)  # Ensure highest predicted_loss groups are at the top
        )
        train_list_temp = shuffled_df['gene_name'].head(opts.num_ptb).tolist()

    assert len(set(train_list_temp)) == opts.num_ptb
    assert set(train_list_temp) - set(opts.all_ptb_targets_with_embeddings) == set(), 'train_list_temp contains genes not in all_ptb_targets_with_embeddings'
    assert len(set(train_list_temp) & set(train_list_current)) == 0, 'overlap between train_list_temp and train_list_current'
    assert len(set(train_list_temp) & set(opts.ptb_leave_out_list)) == 0, 'overlap between train_list_temp and ptb_leave_out_list'

    return train_list_temp

                


