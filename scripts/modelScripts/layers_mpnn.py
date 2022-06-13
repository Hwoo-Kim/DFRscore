import torch
import torch.nn as nn

import numpy as np
import os
import argparse

from torch.autograd.variable import Variable

class FeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dims=(128, 128)):
        super().__init__()
        self.num_hidden = len(hidden_dims)
        self.fcs = nn.ModuleList()
        for i in range(self.num_hidden+1):
            if i ==0: self.fcs.append(nn.Linear(in_dim,hidden_dims[i]))
            elif i==self.num_hidden: self.fcs.append(nn.Linear(hidden_dims[-1],out_dim))
            else: self.fcs.append(nn.Linear(hidden_dims[i-1],hidden_dims[i]))
        self.relu = nn.ReLU()

    def forward(self, x):
        #x = x.contiguous().view(x.size()[0], -1)    # [B,N,H] -> [B,N*H]
        for layer in self.fcs:
            x = self.relu(layer(x))
        return x

class MessageFunction(nn.Module):
    """
    M(h_v,h_w,e_vw) = A(e_vw)h_w, where A is a neural network which maps the edge vector to a dxd matrix.
    Here, A is implemented using FeedForward class as a edge embedding function.
    """
    def __init__(self, edge_dim, hidden_dim, message_dim):
        super().__init__()
        self.args = self._args_setting(edge_dim, hidden_dim, message_dim)
        #self.edge_embed = FeedForward(in_dim=self.args['edge'], out_dim=self.args['in']*self.args['out'])

    def edge_embedding(self):
        return FeedForward(in_dim=self.args['edge'], out_dim=self.args['in']*self.args['out'])

    @staticmethod
    def _args_setting(edge_dim, hidden_dim, message_dim):
        return {'edge': edge_dim, 'in':hidden_dim, 'out':message_dim}

    def forward(self, hidden, edge, edge_embedded=True):
        """
        hidden: [B,N,F].
        edge:   [B,N,N,message,F]. already embedded.
        """
        #edge_output = self.edge_embed(edge)                                     # [B*N*N,edge] -> [B*N*N,F*message]
        #edge_output = edge_output.view(-1, self.args['out'], self.args['in'])   # [B*N*N,message,F]
        if edge_embedded == False:
            edge = self.edge_embedding()(edge)

        hidden = hidden.unsqueeze(1).repeat(1,hidden.size(1),1,1)           # [B,N,F] -> [B,N,N,F]
        #hidden = hidden.view(-1, self.args['in'])                           # [B*N*N,F]
        
        # [B*N*N,message,F]x[B*N*N,F,1] -> [B*N*N,message]
        message = torch.matmul(edge, hidden.unsqueeze(-1)).squeeze()     #[B,N,N,M]
        return message

class UpdateFunction(nn.Module):
    def __init__(self, hidden_dim, message_dim):
        super().__init__()
        self.args = self._args_setting(hidden_dim, message_dim)
        self.gru = nn.GRU(self.args['in'], self.args['out'])

    @staticmethod
    def _args_setting(hidden_dim, message_dim):
        return {'in':message_dim, 'out':hidden_dim}

    def forward(self, hidden, message):
        """
        hidden:  [B,N,F].
        message: [B,N,message]. embedded into [B*N*N, message*F]
        """
        h = hidden.view(1,-1, hidden.size(2))      # [1,B*N,F]
        m = message.view(1,-1, message.size(2))    # [1,B*N,message]
        h_new = self.gru(m, h)[0]                   # [B*N,F]
        return h_new.view(hidden.size())            # [B,N,F]

class Readout(nn.Module):
    def __init__(self, hidden_dim, target_dim):
        super().__init__()
        self.args = self._args_setting(hidden_dim, target_dim)
        #self.layer_i = FeedForward(in_dim=2*self.args['in'], out_dim=self.args['out'],hidden_dims=[128,128])
        #self.layer_j = FeedForward(in_dim=self.args['in'], out_dim=self.args['out'],hidden_dims=[128,128])
        self.layer_i = FeedForward(in_dim=2*self.args['in'], out_dim=self.args['out'])
        self.layer_j = FeedForward(in_dim=self.args['in'], out_dim=self.args['out'])
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _args_setting(hidden_dim, target_dim):
        return {'in': hidden_dim, 'out': target_dim}

    def forward(self, h_0, h_last,masking): # [B,N,F], [B,N,F]
        retval = self.sigmoid(self.layer_i(torch.cat([h_0, h_last], 2)))*self.layer_j(h_last)
        retval = retval*masking
        retval = retval.sum(1)
        return retval
