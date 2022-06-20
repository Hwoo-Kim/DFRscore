import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import rdmolops

class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims,dropout):
        super().__init__()
        self.num_hidden = len(hidden_dims)
        self.fcs = nn.ModuleList()
        for i in range(self.num_hidden+1):
            if i ==0: self.fcs.append(nn.Linear(in_dim,hidden_dims[i]))
            elif i==self.num_hidden: self.fcs.append(nn.Linear(hidden_dims[-1],out_dim))
            else: self.fcs.append(nn.Linear(hidden_dims[i-1],hidden_dims[i]))
        self.act = nn.ReLU()
        self.dropout = dropout

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = self.act(layer(x))
            if self.dropout: x = self.dropout(x)
        x = self.fcs[-1](x)
        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, alpha=0.2,bias=True, dropout=nn.Dropout(p=0.2)):
        super().__init__()

        assert emb_dim % num_heads == 0, "For GAT layer, emb_dim must be dividable by num_heads."
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.alpha = alpha
        self.bias = bias
        self.leakyrelu=nn.LeakyReLU(negative_slope=alpha)
        self.act = nn.ReLU()
        self.W = nn.Linear(emb_dim, emb_dim, bias=self.bias)     # each W_k = [emb_dim, emb_dim/num_heads]
        self.a= nn.Linear(emb_dim, 2*num_heads, bias=self.bias)
        self.dropout = dropout

    def forward(self, x, A):
        Wh = self.W(x)                      # [B,N,F]
        e = self.get_attention_coeff(Wh)    # [B,H,N,N]
        Wh = self.transform(Wh)             # [B,H,N,F/H] 

        inf = torch.tensor(float('-inf')).to(x.device)
        A = A.unsqueeze(1).repeat(1,self.num_heads,1,1)
        att_coeff = torch.where(A==1.0, self.leakyrelu(e), inf)
        attention = torch.softmax(att_coeff, dim=-1)                # [B,H,N,N]
        attention = self.nan_to_num(attention)
        retval = torch.matmul(attention,Wh)                         # [B,H,N,F/H]
        retval = self.restore(retval)
        if self.dropout: retval = self.dropout(retval)
        return self.act(retval)

    def transform(self, tensor):
        # input: [B,N,F] -> output: [B,H,N,F/H]
        new_shape = tensor.size()[:-1] + (self.num_heads, -1)
        tensor = tensor.view(new_shape)         # [B,N,H,F/H]
        return tensor.transpose(1,2)            # [B,H,N,F/H]

    def restore(self, tensor):
        # input: [B,H,N,F/H] -> output: [B,N,F]
        tensor = tensor.transpose(1,2)          # [B,N,H,F/H]
        new_shape = tensor.size()[:-2] + (self.emb_dim,)
        return tensor.reshape(new_shape)           # [B,N,F]

    def nan_to_num(self, attention):
        attention = torch.where(attention.isnan(), torch.Tensor([0.0]).to(attention.data.device), attention)
        return attention

    def get_attention_coeff(self, Wh):
        # input: [B,N,F], output: [B,H,N,N]
        e12 = self.a(Wh)                # [B,N,F] -> [B,N,2*H]
        e12 = self.transform(e12)       # [B,N,2*H] -> [B,H,N,2]
        e1, e2 = e12[:,:,:,0].unsqueeze(-1), e12[:,:,:,1].unsqueeze(-1)
        return e1 + e2.transpose(-2,-1)    

    def __repr__(self):
        return f'{self.__class__.__name__}(\n' + \
                f'  activation: {self.leakyrelu}\n' + \
                f'  W: Linear({self.emb_dim} -> {self.emb_dim}, bias={self.bias})\n' + \
                f'  a: Linear({self.emb_dim} -> {2*self.num_heads}, bias={self.bias})\n)'

if __name__=='__main__':
    att = GraphAttentionLayer(
            emb_dim=128,
            num_heads=8,
            dropout=0.2,
            alpha=0.2
            )
    print(att)
