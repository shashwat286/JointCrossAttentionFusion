import torch
import torch.nn as nn
from scipy.spatial.distance import pdist, squareform
import torch.nn.init as torch_init
import numpy as np
import math




class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=2500):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class JointCrossAttention(nn.Module):
    def __init__(self, hid_dim, n_heads=1):
        super(JointCrossAttention, self).__init__()
        self.hid_dim =hid_dim
        self.n_heads = n_heads

        self.ja = None
        self.jv = None
        
        self.ca = None
        self.a = None
        self.cv = None
        self.v = None
        
        self.ha = None
        self.hv = None
        
        self.tanh = nn.Tanh()
        self.ReLU = nn.ReLU()

    def forward(self, f, f_v, f_a):
        self.ja = nn.Linear(f.shape[0],f.shape[0]).to('cuda')
        self.jv = nn.Linear(f.shape[0],f.shape[0]).to('cuda')
        
        self.ca = nn.Linear(f.shape[2],self.hid_dim).to('cuda')
        self.a = nn.Linear(f.shape[0],self.hid_dim).to('cuda')
        self.cv = nn.Linear(f.shape[2],self.hid_dim).to('cuda')
        self.v = nn.Linear(f.shape[0],self.hid_dim).to('cuda')
        
        self.ha = nn.Linear(self.hid_dim,f.shape[0]).to('cuda')
        self.hv = nn.Linear(self.hid_dim,f.shape[0]).to('cuda')
        self.norm_fact = 1 / math.sqrt(f.shape[2])


        c_a = torch.matmul(self.ja(f_a.permute(1,2,0)),f.permute(1,0,2))*self.norm_fact
        c_a = self.tanh(c_a)
       
        c_v = torch.matmul(self.jv(f_v.permute(1,2,0)),f.permute(1,0,2))*self.norm_fact
        c_v = self.tanh(c_v)
       
        
        h_a = self.a(f_a.permute(1,2,0)) + self.ca(c_a)
        h_a = self.ReLU(h_a)
        
        h_v = self.v(f_v.permute(1,2,0)) + self.cv(c_v)
        h_v = self.ReLU(h_v)
      
        o_a = f_a + self.ha(h_a).permute(2,0,1)
        o_v = f_v + self.hv(h_v).permute(2,0,1)
        
        output = torch.cat((o_v,o_a), dim=2)
        
        return output

    