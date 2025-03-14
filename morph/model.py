import torch
import torch.nn as nn
import numpy as np

class MORPH(nn.Module):
    def __init__(self, dim, c_dim, 
                 opts,
                 device=None):
        
        super(MORPH, self).__init__()
        
        if device is None:
            self.cuda = False
            self.device = 'cpu'
        else:
            self.device = device
            self.cuda = True
        
        self.z_dim_gene = opts.latdim_ctrl #latent dimension for gene decoder
        self.z_dim_ptb = opts.latdim_ptb #latent dimension for perturbation decoder
        self.c_dim = c_dim #input dimension for perturbation decoder
        self.dim = dim #input dimension for gene-expression decoder
        self.geneset_num = opts.geneset_num # specify number of geneset in the geneset matrix
        self.geneset_dim = opts.geneset_dim # specify the dimension of the gene set matrix
        self.hidden_dim = opts.hidden_dim

        if opts.null_label == 'zeros':
            self.null_label = torch.zeros(1, self.c_dim).double().to(self.device)
        elif opts.null_label == 'gaussian':
            null_folder = f'./data/gene_ptb_emb/Null/dim_{self.c_dim}/null_vector.npy'
            self.null_label = torch.tensor(np.load(null_folder)).double().to(self.device)
        elif opts.null_label == 'gaussian_normalized':
            null_folder = f'./data/gene_ptb_emb/Null/dim_{self.c_dim}/normalized_vector.npy'
            self.null_label = torch.tensor(np.load(null_folder)).double().to(self.device)

        # Gene-expression encoder
        self.fc1 = nn.Linear(self.dim, 512)
        self.fc2 = nn.Linear(512, self.hidden_dim)
        self.fc_mean = nn.Linear(self.hidden_dim, self.z_dim_gene)
        self.fc_var = nn.Linear(self.hidden_dim, self.z_dim_gene)

        # Gene Program matrix
        self.geneset_mtrx = nn.Parameter(torch.randn(self.geneset_num, self.geneset_dim))
        
        # Perturbation encoder
        if self.z_dim_ptb > 150:
            self.c1 = nn.Linear(self.c_dim, 512) #c_dim -> hids
            self.c2 = nn.Linear(512, 256) #c_dim -> hids
            self.c3 = nn.Linear(256, self.z_dim_ptb) #c_dim -> hids
            if self.z_dim_ptb > 256:
                raise ValueError("z_dim_ptb should be less than or equal to 256")
        else:
            self.c1 = nn.Linear(self.c_dim, 200) #c_dim -> hids
            self.c2 = nn.Linear(200, 150) #c_dim -> hids
            self.c3 = nn.Linear(150, self.z_dim_ptb) #c_dim -> hids

        # Q,K,V for cross-attention
        self.W_Q1 = nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.z_dim_gene + self.z_dim_ptb)
        self.W_K1 = nn.Linear(self.geneset_dim, self.z_dim_gene + self.z_dim_ptb)
        self.W_V1 = nn.Linear(self.geneset_dim, self.z_dim_gene + self.z_dim_ptb)
        
        self.W_Q2 = nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.z_dim_gene + self.z_dim_ptb)
        self.W_K2 = nn.Linear(self.geneset_dim, self.z_dim_gene + self.z_dim_ptb)
        self.W_V2 = nn.Linear(self.geneset_dim, self.z_dim_gene + self.z_dim_ptb)

        # Normalization and Feed Forward
        self.FF1 = nn.Sequential(
            nn.LayerNorm(self.z_dim_gene + self.z_dim_ptb),
            nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.z_dim_gene + self.z_dim_ptb),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.z_dim_gene + self.z_dim_ptb)
        )
        
        self.FF2 = nn.Sequential(
            nn.LayerNorm(self.z_dim_gene + self.z_dim_ptb),
            nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.z_dim_gene + self.z_dim_ptb),
            nn.LeakyReLU(0.2),
            nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.z_dim_gene + self.z_dim_ptb)
        )
        
        # Decoder MLP
        self.d1 = nn.Linear(self.z_dim_gene + self.z_dim_ptb, self.dim) #z_dim -> hids
        
        # activation functions
        self.leakyrelu = nn.LeakyReLU(0.2)

    def encode(self, x):
        'Encoder for gene-expression data'
        h = self.leakyrelu(self.fc1(x))
        h = self.leakyrelu(self.fc2(h))
        return self.fc_mean(h), self.fc_var(h)
    
    def c_encode(self, c):
        'Encoder for perturbation label'
        h = self.leakyrelu(self.c1(c))
        h = self.leakyrelu(self.c2(h))
        return self.leakyrelu(self.c3(h))
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def attention_1(self, query, gene_set_matrix):
        latent_dim = self.W_K1.out_features
        q = self.W_Q1(query).view(-1, 1, latent_dim)
        # copy gene_set_matrix to match the batch size
        gene_set_matrix = gene_set_matrix.unsqueeze(0).expand(query.size(0), -1, -1)
        k = self.W_K1(gene_set_matrix).transpose(1, 2)
        v = self.W_V1(gene_set_matrix)

        scores = torch.bmm(q, k) / np.sqrt(latent_dim)
        alpha = nn.Softmax(dim=-1)(scores)
        attn_output = torch.bmm(alpha, v).squeeze(1)
        return attn_output, alpha, q, k, v

    def attention_2(self, query, gene_set_matrix):
        latent_dim = self.W_K2.out_features
        q = self.W_Q2(query).view(-1, 1, latent_dim)
        # copy gene_set_matrix to match the batch size
        gene_set_matrix = gene_set_matrix.unsqueeze(0).expand(query.size(0), -1, -1)
        k = self.W_K2(gene_set_matrix).transpose(1, 2)
        v = self.W_V2(gene_set_matrix)

        scores = torch.bmm(q, k) / np.sqrt(latent_dim)
        alpha = nn.Softmax(dim=-1)(scores)
        attn_output = torch.bmm(alpha, v).squeeze(1)
        return attn_output, alpha, q, k, v

    def decode(self, u):
        'Decoder for gene-expression data'        
        h = self.leakyrelu(self.d1(u))
        return h

    def forward(self, x, c_1, c_2, gene_set_mtrx = None, return_alphas = False, return_latents = False):

        # encode perturbation label
        z_ptb_1 = self.c_encode(c_1)
        z_ptb_2 = None
        c_2_test = c_2.detach().cpu().numpy()
        if not np.all(np.isnan(c_2_test)):
            z_ptb_2 = self.c_encode(c_2)
            del c_2_test
        if z_ptb_2 is not None:
            z_ptb_ft = z_ptb_1 + z_ptb_2
        else:
            z_ptb_ft = z_ptb_1

        # encode control label
        control_label = self.null_label.repeat(c_1.size(0), 1)
        z_control = self.c_encode(control_label)

        # encode gene-expression data
        mu, var = self.encode(x)
        z = self.reparametrize(mu, var)
        
        if gene_set_mtrx is None:
            gene_set_mtrx = self.geneset_mtrx
        else:
            gene_set_mtrx = gene_set_mtrx

        # y_hat --------------------------------
        # Attention 1
        z_ptb = torch.cat((z, z_ptb_ft), dim=1)
        attn_out_1, alphas_1_ptb, q_1_ptb, k_1, v_1 = self.attention_1(query=z_ptb, gene_set_matrix=gene_set_mtrx)
        z_ptb = attn_out_1 + z_ptb
        z_ptb = self.FF1(z_ptb) + z_ptb

        # Attention 2
        attn_out_2, alphas_2_ptb, q_2_ptb, k_2, v_2 = self.attention_2(query=z_ptb, gene_set_matrix=gene_set_mtrx)
        z_ptb = attn_out_2 + z_ptb
        z_ptb = self.FF2(z_ptb) + z_ptb
        
        y_hat = self.decode(z_ptb)
        
        # x_recon --------------------------------
        # Attention 1
        z_control = torch.cat((z, z_control), dim=1)
        attn_out_1, alphas_1_ctrl, q_1_ctrl, k_1, v_1 = self.attention_1(query=z_control, gene_set_matrix=gene_set_mtrx)
        z_control = attn_out_1 + z_control
        z_control = self.FF1(z_control) + z_control

        # Attention 2
        attn_out_2, alphas_2_ctrl, q_2_ctrl, k_2, v_2 = self.attention_2(query=z_control, gene_set_matrix=gene_set_mtrx)
        z_control = attn_out_2 + z_control
        z_control = self.FF2(z_control) + z_control
        
        x_recon = self.decode(z_control)
        
        if return_alphas:
            alphas = [alphas_1_ptb, alphas_2_ptb, alphas_1_ctrl, alphas_2_ctrl, q_1_ptb, q_1_ctrl, k_1, v_1, q_2_ptb, q_2_ctrl, k_2, v_2]
            if return_latents:
                return y_hat, x_recon, mu, var, alphas, z_ptb_ft
            else:
                return y_hat, x_recon, mu, var, alphas
        else:
            if return_latents:
                return y_hat, x_recon, mu, var, z_ptb_ft
            else:
                return y_hat, x_recon, mu, var
