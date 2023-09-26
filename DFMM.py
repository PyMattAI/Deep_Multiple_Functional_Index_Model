import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

# import data
X = None
with open('X.dat', 'rb') as f:
    X = pickle.load(f)


Y = None
with open('Y.dat', 'rb') as f:
    Y = pickle.load(f)

# Definitions of the different classes
class LayerNorm(nn.Module):

    def __init__(self, d, eps=1e-6):
        super().__init__()
        # d is the normalization dimension
        self.d = d
        self.eps = eps
        self.alpha = nn.Parameter(torch.randn(d))
        self.beta = nn.Parameter(torch.randn(d))

    def forward(self, x):
        # x is a torch.Tensor
        # avg is the mean value of a layer
        avg = x.mean(dim=-1, keepdim=True)
        # std is the standard deviation of a layer (eps is added to prevent dividing by zero)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - avg) / std * self.alpha + self.beta
		

class FeedForward(nn.Module):

    def __init__(self, in_d=1, out_d=7,hidden=[4,4,4], dropout=0.1, activation=F.relu):
        # in_d      : input dimension, integer
        # hidden    : hidden layer dimension, array of integers
        # dropout   : dropout probability, a float between 0.0 and 1.0
        # activation: activation function at each layer
        super().__init__()
        self.sigma = activation
        dim = [in_d] + hidden + [out_d]
        self.layers = nn.ModuleList([nn.Linear(dim[i-1], dim[i]) for i in range(1, len(dim))])
        self.ln = nn.ModuleList([LayerNorm(k) for k in hidden])
        self.dp = nn.ModuleList([nn.Dropout(dropout) for _ in range(len(hidden))])

    def forward(self, t):
        for i in range(len(self.layers)-1):
            t = self.layers[i](t)
            # skipping connection
            t = t + self.ln[i](t)
            t = self.sigma(t)
            # apply dropout
            t = self.dp[i](t)
        # linear activation at the last layer
        return self.layers[-1](t)


def _inner_product(f1, f2, h):
    prod = f1 * f2 # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), torch.unsqueeze(h,dim=-1))/2

def _l1(f, h):
# f dimension : ( B bases, J )
    B, J = f.size()
    return _inner_product(torch.abs(f), torch.ones((B, J)), h)

def _l2(f, h):
        # f dimension : ( B bases, J )
        # output dimension - ( B bases, 1 )
    return torch.sqrt(_inner_product(f, f, h)) 
		

	
class AdaFNN(nn.Module):

    def __init__(self, n_base=4, base_hidden=[64, 64, 64], grid=(0, 1),
                 dropout=0.1, lambda1=0.0, lambda2=0.0,
                 device=torch.device("cuda",1) ):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        grid        : observation time grid, array of sorted floats including 0.0 and 1.0
        sub_hidden  : hidden layers in the subsequent network, array of integers
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        # grid should include both end points
        grid = np.array(grid)
        # send the time grid tensor to device
        self.t = torch.tensor(grid).to(device).float()
        self.h = torch.tensor(grid[1:] - grid[:-1]).to(device).float()
        # instantiate each basis node in the basis layer
        self.BL = nn.ModuleList([FeedForward(in_d=1, out_d=1,hidden=base_hidden, dropout=dropout, activation=F.selu)
                                 for _ in range(n_base*80)])
        # instantiate the subsequent network
    
    def forward(self, x):
        B, m, J = x.size()
        #assert J == self.h.size()[0] + 1
        T = self.t.unsqueeze(dim=-1)
        # evaluate the current basis nodes at time grid
        self.bases = [basis(T).transpose(-1, -2) for basis in self.BL] #### n_base
        """
        compute each basis node's L2 norm
        normalize basis nodes
        """
        l2_norm = _l2(torch.cat(self.bases, dim=0), self.h).detach()
        self.normalized_bases = [self.bases[i] / (l2_norm[i, 0] + 1e-6) for i in range(self.n_base*80)]
        # compute each score <basis_i, f> 
        score = torch.cat([torch.cat([_inner_product(b.repeat((B, 1)), x[:,:,i], self.h) # (B, 1)
                           for b in self.bases[(i*n_base):((i+1)*n_base)]], dim=-1)  for i in range(J)],dim=-1) # score dim = (B, n_base*m)
        return score

    def R1(self, l1_k):
        """
        L1 regularization
        l1_k : number of basis nodes to regularize, integer        
        """
        if self.lambda1 == 0: return torch.zeros(1).to(self.device)
        selected = np.random.choice(self.n_base*80, min(l1_k, self.n_base*80), replace=False)
        selected_bases = torch.cat([self.normalized_bases[i] for i in selected], dim=0) # (k, J)
        return self.lambda1 * torch.mean(_l1(selected_bases, self.h))

    def R2(self, l2_pairs):
        """
        L2 regularization
        l2_pairs : number of pairs to regularize, integer  
        """
        if self.lambda2 == 0 or self.n_base == 1: return torch.zeros(1).to(self.device)
        k = min(l2_pairs, self.n_base * 80*(self.n_base*80 - 1) // 2)
        f1, f2 = [None] * k, [None] * k
        for i in range(k):
            a, b = np.random.choice(self.n_base*80, 2, replace=False)
            f1[i], f2[i] = self.normalized_bases[a], self.normalized_bases[b]
        return self.lambda2 * torch.mean(torch.abs(_inner_product(torch.cat(f1, dim=0),
                                                                  torch.cat(f2, dim=0),
                                                                  self.h)))



class DFMM(nn.Module):

    def __init__(self, n_base=4, base_hidden=[80, 80, 80], grid=(0, 1),
                 dropout=0.1, lambda1=0.0, lambda2=0.0,
                 device=torch.device("cuda",0) ):
        """
        n_base      : number of basis nodes, integer
        base_hidden : hidden layers used in each basis node, array of integers
        grid        : observation time grid, array of sorted floats including 0.0 and 1.0
        sub_hidden  : hidden layers in the subsequent network, array of integers
        dropout     : dropout probability
        lambda1     : penalty of L1 regularization, a positive real number
        lambda2     : penalty of L2 regularization, a positive real number
        device      : device for the training
        """
        super().__init__()
        self.n_base = n_base
        self.device = device
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ln = LayerNorm(4) 
        # instantiate each basis node in the basis layer
        self.Ada = AdaFNN(n_base=n_base, base_hidden=base_hidden, grid=grid,
                 dropout=dropout, lambda1=lambda1, lambda2=lambda2,
                 device=device)
        # instantiate the subsequent network
        self.FF = FeedForward(self.n_base*80,4,[self.n_base*80,160,128])
        # instantiate the initial network
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=20, nhead=4, dim_feedforward=1024, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, 4)
        

    def forward(self, x): 
        transform = self.transformer_encoder(x) 
        transform = torch.swapaxes(transform,1,-1) 
        out = self.Ada(transform) 
        out = self.FF(out)
        out = self.ln(out)
        return out
    
    def L1(self, l1_k):
        return self.Ada.R1(l1_k) 

    def L2(self, l2_pairs):
        return self.Ada.R2(l2_pairs)




		
# Example of parameters values 


batch_size = 128

T = pd.DataFrame([np.linspace(0,1,20)])
grid = T.iloc[0, :].to_list()


# set up CPU/GPU
device = torch.device('cuda') 

base_hidden = [256, 256, 256, 256]
n_base = 2
lambda1, l1_k = 0.0, 2
lambda2, l2_pairs = 0.5, 3
dropout = 0.2

 
