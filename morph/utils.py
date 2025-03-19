import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import Sampler

from dataset import SCDataset


## MMD LOSS
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        return
    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        # print(torch.sum(L2_distance.data) / (n_samples**2-n_samples))
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def get_data(opts):
    dataset_name=opts.dataset_name
    adata_path=opts.adata_path
    batch_size=opts.batch_size
    leave_out_test_set=opts.leave_out_test_set
    validation_set_ratio=opts.validation_set_ratio
    validation_ood_ratio=opts.validation_ood_ratio
    representation_type=opts.label
    representation_type_2 = opts.label_2
    representation_type_3 = opts.label_3
    min_counts=opts.batch_size
    random_seed=opts.seed
    
    # First get dataset and indices
    dataset = SCDataset(base_dir=opts.base_dir,
                        dataset_name=dataset_name,
                        adata_path=adata_path,
                        leave_out_test_set = leave_out_test_set, 
                        representation_type=representation_type, representation_type_2=representation_type_2,
                        representation_type_3=representation_type_3,
                        min_counts=min_counts,
                        random_seed=random_seed)
    ptb_leave_out_list = dataset.ptb_leave_out_list
    print('Generated dataset')
    print('Getting train and validation indices')
    train_idx, val_idx, infer_index = split_scdata(
        dataset, 
        ptb_targets = dataset.ptb_targets,
        ptb_leave_out_list=ptb_leave_out_list, # list of targets to leave out for testing
        validation_set_ratio=validation_set_ratio,
        validation_ood_ratio=validation_ood_ratio,
        batch_size=batch_size
    )
    print('Got train and validation indices')
    
    ptb_genes = dataset.ptb_targets #list of unique perturbation target names
    ptb_vector_dict = dataset.ptb_vector_dict #dictionary of perturbation target names and their corresponding representation vectors
    
    # Load data
    # Training dataloader
    dataset1 = Subset(dataset, train_idx) 
    train_ptb_name = dataset.ptb_names[train_idx] 
    dataloader = DataLoader(
        dataset1,
        batch_sampler=SCDATA_sampler(dataset1, batch_size, train_ptb_name),
        num_workers=0
    )
    
    dim = dataset[0][0].shape[0] # dimension of the gene expression vector: 8749 for RPE1
    cdim = dataset[0][2].shape[0] # dimension of the perturbation vector (e.g., 256 for Geneformer, and 'number of perturbation targets' for Baseline)
    if representation_type_2 is not None:
        cdim_2 = dataset[0][5].shape[0]
    else:
        cdim_2 = None

    if representation_type_3 is not None:
        cdim_3 = dataset[0][7].shape[0]
    else:
        cdim_3 = None
    
    # Testing dataloader
    dataset_val = Subset(dataset, val_idx)
    ptb_name = dataset.ptb_names[val_idx]
    dataloader_val = DataLoader(
        dataset_val,
        batch_sampler=SCDATA_sampler(dataset_val, batch_size, ptb_name),
        num_workers=0
    )

    # Inference dataloader
    if infer_index is not None:
        dataset_infer = Subset(dataset, infer_index)
        ptb_name = dataset.ptb_names[infer_index]
        assert(set(ptb_name) == set(ptb_leave_out_list)), "Perturbation targets in inference set are those specified in ptb_leave_out_list"
        assert(ptb_name.isin(train_ptb_name).any() == False), "Perturbation targets in inference set should NOT be in training set"
        dataloader_infer = DataLoader(
            dataset_infer,
            batch_sampler=SCDATA_sampler(dataset_infer, batch_size, ptb_name),
            num_workers=0
        )
    else:
        dataloader_infer = None
    
    return dataloader, train_idx, dataloader_val, val_idx, dataloader_infer, infer_index, dim, cdim, cdim_2, cdim_3, ptb_genes, ptb_vector_dict, ptb_leave_out_list

def split_scdata(scdataset, ptb_targets = None, ptb_leave_out_list=None, 
                 validation_set_ratio=None, validation_ood_ratio = None, 
                 batch_size=32, max_ood_val_genes=5,):
    'leave out some cells for testing'
    assert(len(ptb_targets) > len(ptb_leave_out_list)), "Need at least one perturbation target for training"
    assert(len(ptb_targets) - len(ptb_leave_out_list) >= max_ood_val_genes), "Not enough perturbation targets to leave out for testing"

    if len(ptb_leave_out_list) > 0:
        infer_idx = np.where(np.isin(scdataset.ptb_names, ptb_leave_out_list))[0].tolist()
    else:
        infer_idx = None
    
    # Determine training candidates and validation perturbations
    training_candidates = set(ptb_targets) - set(ptb_leave_out_list)
    validation_set_size = int(validation_set_ratio * len(training_candidates))
    print(f"Validation set size: {validation_set_size}")
    validation_ood_size = int(validation_ood_ratio * validation_set_size)
    print(f"Validation ood set size: {validation_ood_size}")
    val_ptb_list = random.sample(training_candidates, validation_set_size) #for validation
    val_ood_ptb_list = random.sample(val_ptb_list, validation_ood_size) #for validation (simulate ood)
    val_iid_ptb_list = list(set(val_ptb_list) - set(val_ood_ptb_list))  #for validation (simulate iid)

    if validation_set_size > 0:
        # Calculate validation out-of-distribution indices
        if max_ood_val_genes > 0:
            val_ood_idx = np.where(np.isin(scdataset.ptb_names, val_ood_ptb_list))[0].tolist()
        else:
            val_ood_idx = []
        
        # Calculate validation in-distribution indices
        val_iid_idx = []
        for ptb in val_iid_ptb_list:
            idx = np.where(scdataset.ptb_names == ptb)[0]
            batch_num = (len(idx) // batch_size)
            num_sample = int(batch_num/2) * batch_size
            val_iid_idx.extend(idx[:num_sample])
        val_iid_idx = list(set(val_iid_idx))
        
        # Combine validation indices
        val_idx = val_ood_idx + val_iid_idx if val_ood_idx else val_iid_idx
        print('Finished calculating validation indices')

        # Calculate training indices excluding validation and inference indices
        all_indices = np.arange(len(scdataset))
        val_set = set(val_idx)
        infer_set = set(infer_idx) if infer_idx is not None else set()
        train_idx = np.array([i for i in all_indices if i not in val_set and i not in infer_set])
    else:
        # Calculate training indices excluding only inference indices
        all_indices = np.arange(len(scdataset))
        infer_set = set(infer_idx) if infer_idx is not None else set()
        train_idx = np.array([i for i in all_indices if i not in infer_set])
        val_idx = None
    
    print('Finished calculating training indices')
    if infer_idx is not None:
        assert(len(train_idx) + len(val_idx) + len(infer_idx) == len(scdataset)), "train_idx, val_idx, infer_idx should be disjoint"
    else:
        assert(len(train_idx) + len(val_idx) == len(scdataset)), "train_idx, val_idx should be disjoint"
    return train_idx, val_idx, infer_idx


# a special batch sampler that groups only cells from the same interventional distribution into a batch
class SCDATA_sampler(Sampler):
    def __init__(self, scdataset, batchsize, ptb_name=None):
        self.intervindices = []
        self.len = 0
        if ptb_name is None:
            ptb_name = scdataset.ptb_names
        for ptb in set(ptb_name):
            idx = np.where(ptb_name == ptb)[0] # indices of cells with the same interventional distribution
            self.intervindices.append(idx) # list of indices of cells with the same interventional distribution
            self.len += len(idx) // batchsize # number of batches
        self.batchsize = batchsize
    
    def __iter__(self):
        comb = []
        # loop over each intervention
        for i in range(len(self.intervindices)):
            random.shuffle(self.intervindices[i])
        
            interv_batches = chunk(self.intervindices[i], self.batchsize)
            if interv_batches:
                comb += interv_batches

        combined = [batch.tolist() for batch in comb]
        random.shuffle(combined)
        return iter(combined)
    
    def __len__(self):
        return self.len


def chunk(indices, chunk_size):
    split = torch.split(torch.tensor(indices), chunk_size)
    
    if len(indices) % chunk_size == 0:
        return split
    elif len(split) > 0:
        # If there are some indices left over (the last chunk is smaller than chunk_size),
        # this line discards the last chunk. 
        # It returns all the chunks except the last one.
        return split[:-1]
    else:
        return None