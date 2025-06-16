import torch
from torch import nn
from torchvision import transforms
from transformers import ViTConfig, ViTModel
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
import numpy as np
import sys
import json
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import torch.optim as optim
from pytorch_metric_learning import losses
import wandb
import scanpy as sc
import anndata
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt


from helper import resize_pos_embed, FineTuneViT, ViT_contrastive

sys.path.append('/home/che/perturb-project/git/gene_ptb_prediction/ebov_analysis/DeepLearning')
from tools.config_tools import seed_randomness
from tools.unsupervised_modeling_dataset_tools import UnsupervisedCellsDataset, split_dataset

torch.set_num_threads(8)

# Set-up parameters -----------------------------------------------------------
class_to_idx = {'faint': 0, 'punctate': 1, 'cytoplasmic': 2, 'peripheral': 3}
parent_folder_path = '/data/che/ops_ebov'
test_set_id = 'random_fold_5'
supervision = 'supervised' # supervised or unsupervised
loss_fn = 'SupConLoss' # CrossEntropyLoss or SupConLoss
print(f'Loss function: {loss_fn}')
contrastive_random_sample = False # indicate whether to randomly sample for Contrastive Loss
print(f'contrastive_random_sample:', contrastive_random_sample)
if loss_fn == 'SupConLoss':
    contrastive = True
else:
    contrastive = False
balanced_sample = False # indicate whether to balance the sample
assert (loss_fn == 'SupConLoss' and not balanced_sample), 'Dont set balanced sampling for SupConLoss as it by default samples balanced'
print('balanced_sample:', balanced_sample)

run_name = 'vit'
vit_name = 'google/vit-base-patch16-224'
num_epochs = 10
print(f'Starting training for {num_epochs} epochs')

# Set-up model folder path ---------------------------------------------------
model_folder_path = f'{parent_folder_path}/models/{test_set_id}/ViT/{loss_fn}'
if not os.path.exists(model_folder_path):
    os.makedirs(model_folder_path)
device='cuda:6'

# Create dataloaders ----------------------------------------------------------
config_path = f'{model_folder_path}/config.json'
with open(config_path, 'r') as f:
    config = json.load(f)
assert config['model_name'] == run_name, f'config model name {config["model_name"]} does not match run name {run_name}'
assert config['metadata_path'] == '/data/che/ops_ebov/MetadataEssential/Genomewide_Metadata_Essential_Classification_GWA.parquet'
assert config['metadata_name'] == 'Genomewide_Metadata_Essential_Classification_GWA'
assert config['loss_fn'] == loss_fn
data_path = config['data_path']
training_batch_size = config['training_batch_size']
cell_size = config['cell_size']
seed = config['seed']
rng = seed_randomness(seed)

# first check if test_set_list is in config, if not, load it from metadata
print('Loading test set list from metadata')
split_path = f"/home/che/perturb-project/predict_model/performance_eval/ops_data/{config['metadata_name']}/splits.csv"
split_df = pd.read_csv(split_path)
test_set_list = split_df[split_df['test_set_id'] == config['test_set_id']]['test_set'].values[0].split(',')
if 'test_set_list' not in config:
    config['test_set_list'] = test_set_list
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4) 
else:
    assert set(test_set_list) == set(config['test_set_list']), 'Test set list does not match'
    test_set_list = config['test_set_list']

# load dataset if exists
train_ds_path = model_folder_path+'/train_ds.pkl'
val_ds_path = model_folder_path+'/val_ds.pkl'
test_ds_path = model_folder_path+'/test_ds.pkl'

if os.path.exists(train_ds_path) and os.path.exists(val_ds_path) and os.path.exists(test_ds_path):
    train_ds = pd.read_pickle(train_ds_path)
    val_ds = pd.read_pickle(test_ds_path)
    test_ds = pd.read_pickle(test_ds_path)
else:
    print('Splitting dataset')
    metadata = pd.read_parquet(config['metadata_path'])
    train_ds, val_ds, test_ds, test_set_list = split_dataset(metadata, seed=config['seed'], val_size=config['val_size'], test_set_id=config['test_set_id'], 
                                                             metadata_path=config['metadata_path'], stratify_by_plate=True, return_p_w_t=True)
    assert set(test_set_list) == set(config['test_set_list']), 'Test set list does not match'
    train_ds.to_pickle(train_ds_path)
    val_ds.to_pickle(val_ds_path)
    test_ds.to_pickle(test_ds_path)
    print('Dataset split saved to:', train_ds_path, val_ds_path, test_ds_path)

# get train and test dataloaders
train_dl = UnsupervisedCellsDataset(train_ds, data_path, rng, mode='train', id_task=bool(config['id_task']), gene_to_one_hot=None,
                                        supervision = bool(config['supervision']), min_max_masked=bool(config['min_max_masked']), 
                                        channel=config['channel_name'], balanced_sample=balanced_sample, 
                                        contrastive=contrastive, contrastive_random_sample = contrastive_random_sample,
                                        test_set_list=test_set_list, cell_size=cell_size, batch_size=training_batch_size)
val_dl = UnsupervisedCellsDataset(val_ds, data_path, rng, mode='val', id_task=bool(config['id_task']), gene_to_one_hot=None,
                                supervision = bool(config['supervision']), min_max_masked=bool(config['min_max_masked']), 
                                channel=config['channel_name'], balanced_sample=balanced_sample, 
                                contrastive=contrastive, contrastive_random_sample = contrastive_random_sample,
                                    test_set_list=test_set_list, cell_size=cell_size, batch_size=training_batch_size)
test_dl = UnsupervisedCellsDataset(test_ds, data_path, rng, mode='test', id_task=bool(config['id_task']), gene_to_one_hot=None,
                                supervision = bool(config['supervision']), min_max_masked=bool(config['min_max_masked']), 
                                channel=config['channel_name'], balanced_sample=balanced_sample, 
                                contrastive=contrastive, contrastive_random_sample=contrastive_random_sample,
                                    test_set_list=test_set_list, cell_size=cell_size, batch_size=training_batch_size)

# Load model -----------------------------------------------------------------
model = ViTModel.from_pretrained(vit_name)
state_dict = model.state_dict()
vit_config = ViTConfig.from_pretrained(vit_name, 
                                       image_size=config['cell_size'], 
                                       label2id=class_to_idx, 
                                       id2label={v: k for k, v in class_to_idx.items()})
model = ViTModel(config=vit_config)
new_state_dict = state_dict.copy()

# Resize position embeddings
old_posemb = new_state_dict['embeddings.position_embeddings']
if model.embeddings.position_embeddings.shape != old_posemb.shape:
    new_posemb = resize_pos_embed(old_posemb, model.embeddings.position_embeddings)
    new_state_dict['embeddings.position_embeddings'] = new_posemb

# equip the model with the new state dict
model.load_state_dict(new_state_dict)

# Define model ---------------------------------------------------------------
num_classes = len(class_to_idx)
feat_dim = config['contrastive_feat_dim']
model = FineTuneViT(model, num_classes, feat_dim, loss_fn)

# Start training -------------------------------------------------------------
wandb.init(project=f"ft_vit_{config['metadata_name']}", name=f"vit_{loss_fn}_epoch_{num_epochs}")  #name should be the run time after fixing the os.makedirs bug

if loss_fn == 'SupConLoss':
    criterion = losses.SupConLoss()
elif loss_fn == 'CrossEntropyLoss':
    criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(f'Starting training for {num_epochs} epochs')
best_val_loss = np.inf
best_val_model = deepcopy(model)

for epoch in range(num_epochs):
    model.train()
    model.to(device)
    running_loss = 0.0
    i = 0

    wandb.log({'epoch': epoch+1})

    # training loop
    for images, masks, metadata in tqdm(train_dl, total=len(train_dl)):
        i += 1
        images = images.to(device)
        optimizer.zero_grad()
        cls_output, logits = model(images)
        label_indices = torch.tensor(metadata['label'].map(class_to_idx).values, device=device)
        loss = criterion(logits, label_indices)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({'training_loss': loss.item()})

    epoch_loss = running_loss / len(train_dl)
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')
    wandb.log({'epoch_avg_loss': epoch_loss})

    # validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks, metadata in tqdm(val_dl, total=len(val_dl), desc='Validation'):
            images = images.to(device)
            cls_output, logits = model(images)
            label_indices = torch.tensor(metadata['label'].map(class_to_idx).values, device=device)
            loss = criterion(logits, label_indices)
            val_loss += loss.item()
            wandb.log({'validation_loss': loss.item()})
    val_loss /= len(val_dl)
    print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
    wandb.log({'val_avg_loss': val_loss})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_model = deepcopy(model)
        wandb.log({'best_val_loss': best_val_loss})
        wandb.log({'best_val_epoch': epoch+1})

# Save model -----------------------------------------------------------------
model_path = f'{model_folder_path}/epoch_{num_epochs}'
if not os.path.exists(model_path):
    os.makedirs(model_path)

if balanced_sample:
    model_path = f'{model_path}/balanced_trained'
if not os.path.exists(model_path):
    os.makedirs(model_path)

if contrastive:
    if contrastive_random_sample:
        model_path = f'{model_path}/contrastive_random_sample'
if not os.path.exists(model_path):
    os.makedirs(model_path)

torch.save(model.state_dict(), f'{model_path}/model.pth')
torch.save(best_val_model.state_dict(), f"{model_path}/model_best.pth")
print('Model saved at:', f'{model_path}')

# Now, let's begin data extractions ------------------------------------------
plate_subset = ['A']
metadata_df_1 = pd.read_parquet(config['metadata_path'])
all_ds = metadata_df_1.groupby(["plate", "well", "tile"]).agg("first").reset_index()[["plate", "well", "tile"]]
if len(plate_subset) > 0:
    all_ds = all_ds[all_ds['plate'].isin(plate_subset)]
    metadata_df_1 = metadata_df_1[metadata_df_1['plate'].isin(plate_subset)]
    print('Subsetted to plates:', plate_subset)
print('metadata_df_1 shape:', metadata_df_1.shape)
print('all_ds (unique combination of plate, well and tile) shape:', all_ds.shape)

all_dl = UnsupervisedCellsDataset(all_ds, data_path, rng, mode='val', supervision = bool(config['supervision']),
                                  id_task=bool(config['id_task']), gene_to_one_hot=None,
                                  min_max_masked=bool(config['min_max_masked']), channel=config['channel_name'], 
                                  balanced_sample=False, contrastive=False, contrastive_random_sample=False, #set balanced_sample and contrastive to False for data extraction
                                  test_set_list=None, cell_size=cell_size, batch_size=training_batch_size)

# 1. load in trained model and extract embeddings ------------------------------
# model = best_val_model
best_val_model.eval()
embeddings = []
metadata_df = pd.DataFrame()
with torch.no_grad():
    for _, data in tqdm(enumerate(all_dl), total=len(all_dl)):
        images, masks, metadata = data
        images = images.to(device)
        cls_output, logits = best_val_model(images)
        assert images.shape[0] == masks.shape[0] == metadata.shape[0] == cls_output.shape[0]
        assert cls_output.shape[1] != 4, 'cls_output shape should not match number of classes'
        embeddings.append(cls_output.cpu().numpy())
        metadata_df = pd.concat([metadata_df, metadata], ignore_index=True)

assert metadata_df.shape[0] == metadata_df_1.shape[0], 'metadata_df shape does not match metadata_df_1 shape'
assert len(embeddings) == len(all_dl), 'embeddings length does not match all_dl length'

# 2. save embeddings ----------------------------------------------------------
adata = sc.AnnData(X=np.concatenate(embeddings, axis=0))
adata.obs = metadata_df
adata.obs['i'] = adata.obs['i'].astype(str)
adata.obs['j'] = adata.obs['j'].astype(str)
channel_str = ','.join(config['channel_name'])
adata.obs['channel'] = channel_str
adata.obs['supervision'] = config['supervision']
adata.obs['metadata_path'] = config['metadata_path']
adata.obs['split'] = 'train'
adata.obs.loc[adata.obs['gene'].isin(test_set_list), 'split'] = 'test'
if len(plate_subset) > 0:
    subsets_plate = '_'.join(plate_subset)
    adata.write_h5ad(f"{model_path}/model_best_embeddings_subset_plate_{subsets_plate}.h5ad")
else:
    adata.write_h5ad(f"{model_path}/model_best_embeddings.h5ad")


# Now, let's generate UMAP ----------------------------------------------------
adata_ctrl = adata[adata.obs['gene'] == 'non-targeting'].copy()
adata_test = adata[adata.obs['split'] == 'test'].copy()
# random select 10000 samples from each 
random_seed = 12
np.random.seed(random_seed)
# Randomly select 1000 samples from adata_ctrl
selected_indices_ctrl = np.random.choice(adata_ctrl.shape[0], 2000, replace=False)
adata_ctrl_subset = adata_ctrl[selected_indices_ctrl]

# Randomly select 10000 samples from adata_test
gene = adata_test.obs['gene'].values
unique_gene = np.unique(gene)
selected_genes = np.random.choice(unique_gene, 10, replace=False)
adata_subset = adata[adata.obs['gene'].isin(selected_genes)]

# Stack them together
adata_combined = anndata.concat([adata_ctrl_subset, adata_subset])
print(adata_combined.shape, adata_combined.X.min(), adata_combined.X.max())

# Run PCA
sc.tl.pca(adata_combined, svd_solver='arpack')
# Compute the neighborhood graph
sc.pp.neighbors(adata_combined, n_neighbors=10, n_pcs=50)
# Run UMAP
sc.tl.umap(adata_combined, min_dist=0.5)

with plt.rc_context():
    sc.pl.umap(adata_combined,
            color = ['gene', 'label'], show=False)
    plt.savefig(f"{model_path}/model_best_embeddings_subset_plate_{subsets_plate}_umap_label_explore.png")
    plt.close()

# Plot LogReg and SVM plot ----------------------------------------------------
adata_train = adata[adata.obs['split'] == 'train']
adata_test = adata[adata.obs['split'] == 'test']
X_train = adata_train.X
y_train = adata_train.obs['label']
X_test = adata_test.X
y_test = adata_test.obs['label']
label_map = {'faint': 0, 'punctate': 1, 'cytoplasmic': 2, 'peripheral': 3}
y_test = y_test.map(label_map)
y_train = y_train.map(label_map)
# random select 3000 samples from X_trian, y_train, X_test, y_test
random_seed = 12
np.random.seed(random_seed)
idx_train = np.random.choice(y_train.shape[0], 3000, replace=False)
X_train = X_train[idx_train]
y_train = y_train.iloc[idx_train]

idx_test = np.random.choice(y_test.shape[0], 8000, replace=False)
X_test = X_test[idx_test]
y_test = y_test.iloc[idx_test]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Train the logistic regression
log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train, y_train)

# Evaluate the model
y_pred_log_reg = log_reg.predict(X_test)
balanced_acc = balanced_accuracy_score(y_test, y_pred_log_reg)
print('logistic regression')
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Balanced Accuracy:", balanced_acc)

# Logistic Regression Confusion Matrix ---------------------------------------------------------
value_counts = pd.Series(y_test).value_counts()
cm = confusion_matrix(y_test, y_pred_log_reg)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Define the class names
x_class_names = ['faint', 'punctate', 'cytoplasmic', 'peripheral']
y_class_names = []
# annotate numbers in y_class_names
for c in x_class_names:
    value = value_counts[label_map[c]]
    y_class_names.append(f'{c.capitalize()} [{value}]')
x_class_names = [c.capitalize() for c in x_class_names]

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='magma', xticklabels=x_class_names, yticklabels=y_class_names)
plt.ylabel('True Phenotype [# Samples]')
plt.xlabel('Predicted Phenotype')
plt.title('Confusion Matrix')
# Annotate balanced accuracy
plt.text(0.5, 1.05, f'Balanced Accuracy: {balanced_acc:.2%}', ha='center', va='center', transform=plt.gca().transAxes)
plt.savefig(f"{model_path}/model_best_gt_features_confusion_matrix_log_reg.png")
plt.show()





