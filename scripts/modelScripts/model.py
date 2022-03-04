import torch
import torch.nn as nn
import torch.nn.functional as F

from rdkit import Chem
from rdkit.Chem import rdmolops
from scripts.layers import GraphAttentionLayer

class SVS(nn.Module):
    def __init__(
            self,
            conv_dim,
            fc_dim,
            n_GAT_layer,
            n_fc_layer,
            num_heads,
            len_features,
            num_class,
            dropout:float,
            residual=True
            ):
        super().__init__()
        self.len_feature = len_features
        self.embedding = nn.Linear(len_features,conv_dim)
        self.GAT_layers=nn.ModuleList(
                [GraphAttentionLayer(
                    emb_dim=conv_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    alpha=0.2,
                    bias=True
                    )
                    for i in range(n_GAT_layer)]
                )
        self.fc_layers=nn.ModuleList([nn.Linear(conv_dim,fc_dim)])
        for i in range(n_fc_layer-2):
            self.fc_layers.append(nn.Linear(fc_dim,fc_dim))
        self.pred_layer = nn.Linear(fc_dim,num_class)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

    def forward(self,x,A):
        x = self.embedding(x)
        for layer in self.GAT_layers:
            x = layer(x, A)         # output was already applied with ELU.
        x = retval = x.mean(1)
        for layer in self.fc_layers:
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        retval = self.pred_layer(x)
        return retval

    def restore(self):
        pass

    def smi_to_graph_feature(self):
        pass
