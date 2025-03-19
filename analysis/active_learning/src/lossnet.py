'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 


class LossNet(nn.Module):
    def __init__(self, dim, output_dim=10):
        super(LossNet, self).__init__()
        
        if dim/2 > output_dim:
            hidden_dim = int(dim / 2)
        else:
            hidden_dim = output_dim
        self.dim = dim
        self.num_bins = output_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, features):
        out = self.leaky_relu(self.fc1(features))
        logits = self.fc2(out)
        return logits