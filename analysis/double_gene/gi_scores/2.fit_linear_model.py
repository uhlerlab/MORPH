import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import TheilSenRegressor
from tqdm import tqdm
import torch

### 1. First read in the delta file -------------------------------------------
torch.set_num_threads(8)
random_seed=12
dataset_name = 'norman_k562_hvg'
dataset = dataset_name.replace('_hvg', '')
use_hvg = 'True' if 'hvg' in dataset_name else 'False'

filter_zeros=False
model_type='Truth' # 'Gears' or 'morph' or 'Control' or 'Truth'
if model_type == 'morph':
    representation_type='DepMap_GeneEffect'
    model_name = 'best_model'
    recon_loss = 'mmd'
    null_label = 'zeros'
    mxAlpha = 2.0
    tolerance_epochs = 20
elif model_type == 'Gears':
    model_name='model.pt'
elif model_type == 'Control':
    model_name=None
elif model_type == 'Truth':
    model_name=None
num_epochs=100
use_pred_single = False # True if using predicted single perturbation, False if using true single perturbation

# indicate whether to calculate from the predicted data or the true data ----------------
num_gene = 2500
n_subsamples = 1000 # default set to 1000

parent_dir = f'/home/che/perturb-project/git/gene_ptb_prediction/gene_interaction_prediction/data/{dataset_name}/predict'
if model_type == 'Gears':
    if use_pred_single:
        raise ValueError('Gears model does not have predicted single perturbation')
    else:
        data_path = f'{parent_dir}/use_gt_single/num_gene_{num_gene}/Gears/seed_{random_seed}'
elif model_type == 'morph':
    if use_pred_single:
        data_path = f'{parent_dir}/use_pred_single/{representation_type}_{model_type}/epochs_{num_epochs}_seed_{random_seed}'
    else:
        data_path = f'{parent_dir}/use_gt_single/num_gene_{num_gene}/{representation_type}_{model_type}/recon_loss_{recon_loss}/null_label_{null_label}/epochs_{num_epochs}/tolerance_epochs_{tolerance_epochs}/mxAlpha_{mxAlpha}/seed_{random_seed}/{model_name}'
elif model_type == 'Control':
    data_path = f'{parent_dir}/use_gt_single/num_gene_{num_gene}/Control/seed_{random_seed}'
elif model_type == 'Truth':
    data_path = f'{parent_dir}/use_gt_single/num_gene_{num_gene}/Truth'

if filter_zeros:
    with open(f'{data_path}/delta_expression_filtered.pkl', 'rb') as f:
        delta_gene_expressions = pickle.load(f)
else:
    with open(f'{data_path}/delta_expression.pkl', 'rb') as f:
            delta_gene_expressions = pickle.load(f)
print('Loaded predictions from', data_path)


print(delta_gene_expressions[delta_gene_expressions.index == 'MAPK1+PRTG'])

df = delta_gene_expressions
df = df[df.index != 'non-targeting']


# Prepare to store results
results = []

# Function to fit model and store coefficients and residuals
def fit_model(X, y):
    model = TheilSenRegressor(fit_intercept=False, max_subpopulation=1e5, max_iter=1000, random_state=1000, n_subsamples=n_subsamples)
    model.fit(X, y)
    coefficients = model.coef_
    residuals = y - model.predict(X)
    return coefficients, residuals

# Loop through combinations of perturbations
perturbations = [col for col in df.index if '+' in col]
print('Number of double perturbations:', len(perturbations))
single_perturbations = [col for col in df.index if '+' not in col]
print('Number of single perturbations:', len(single_perturbations))

for comb in tqdm(perturbations):
    # Split combination into individual perturbations
    pert_a, pert_b = comb.split('+')
    
    if pert_a in df.index and pert_b in df.index:
        # Prepare X (features) and y (target)
        X = df.loc[[pert_a, pert_b]].values.T
        y = df.loc[comb].values

        # Fit the Theil-Sen regression model
        coeffs, resids = fit_model(X, y)
        
        # Calculate combined effect
        combined_effect = coeffs[0] * X[:, 0] + coeffs[1] * X[:, 1]

        # Store results
        results.append({
            'combination': comb,
            'g_ab': y,
            'perturbation_a': pert_a,
            'g_a': X[:, 0],
            'perturbation_b': pert_b,
            'g_b': X[:, 1],
            'c_a': coeffs[0],
            'c_b': coeffs[1],
            'epsilon': resids,
            'combined_effect': combined_effect
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

# save the results into pickle file
if filter_zeros:
    with open(f'{data_path}/theilsen_results_n_subsamples_{n_subsamples}_filtered.pkl', 'wb') as f:
        pickle.dump(results_df, f)
    print('saved to ', f'{data_path}/theilsen_results_n_subsamples_{n_subsamples}_filtered.pkl')
else:
    with open(f'{data_path}/theilsen_results_n_subsamples_{n_subsamples}.pkl', 'wb') as f:
        pickle.dump(results_df, f)
    print('saved to ', f'{data_path}/theilsen_results_n_subsamples_{n_subsamples}.pkl')