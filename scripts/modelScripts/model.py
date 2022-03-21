import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolops import GetFormalCharge

from scripts.modelScripts.layers import GraphAttentionLayer
from scripts.modelScripts.preprocessing import get_atoms_feature
from scripts.modelScripts.data import InferenceDataset

import os, sys, pickle
import numpy as np

#MAX_NUM_ATOM = 64
#CONV_DIM = 512
#FC_DIM = 256
#NUM_GAT_LAYER = 5
#NUM_FC_LAYER = 3
#NUM_HEADS= 8
#NUM_CLASS= 5
#LEN_FEATURES = 30
#NUM_CORES= 4

class SVS(nn.Module):
    def __init__(
            self,
            conv_dim,
            fc_dim,
            n_GAT_layer,
            n_fc_layer,
            num_heads,
            len_features,
            max_num_atoms,
            num_class,
            dropout:float,
            residual=True
            ):
        super().__init__()
        self.max_num_atoms = max_num_atoms
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

    def restore(self, path_to_model):
        self.load_state_dict(torch.load(path_to_model))
        pass

    def mol_to_graph_feature(self, mol):
        sssr = Chem.GetSymmSSSR(mol)
        num_atoms = mol.GetNumAtoms()
        adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
        padded_adj = np.zeros((self.max_num_atoms, self.max_num_atoms))
        padded_adj[:num_atoms,:num_atoms] = adj
        feature = []
        atoms = mol.GetAtoms()
        for atom in atoms:
            feature.append(get_atoms_feature(sssr, atom))
        feature = np.array(feature)
        padded_feature = np.zeros((self.max_num_atoms, self.len_features))
        padded_feature[:num_atoms,:self.len_features] = feature
        padded_feature = torch.from_numpy(padded_feature)
        padded_adj = torch.from_numpy(padded_adj)

        return padded_feature, padded_adj

    def mols_to_graph_feature(self, mol_list):
        save_dir = os.join(os.path.dirname(__file__), 'tmp')
        os.mkdir(save_dir)
        for idx, mol in enumerate(mol_list):
            if mol:
                padded_feature, padded_adj = self.mol_to_graph_feature(mol)
                with open(f'{save_dir}/{idx}.pkl','wb') as fw:
                    pickle.dump({'feature':padded_feature,
                            'adj':padded_adj},
                            fw)
            else:
                with open(f'{save_dir}/{idx}.pkl','wb') as fw:
                    pickle.dump({'feature':torch.nan,
                            'adj':torch.nan},
                            fw)

        return save_dir

    def smiToScore(self, smi:str) -> tuple:
        assert isinstance(smi, str), 'input of smiToScore method must be a string of SMILES.'
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            raise AssertionError('input of smiToScore method must be a string of SMILES.')
        if not mol:
            raise AssertionError('Failed to generate rdchem.Mol object from given string.')
        return self.molToScore(mol)

    def molToScore(self, mol:object) -> tuple:
        assert isinstance(mol, rdkit.Chem.rdchem.Mol), \
            'input of molToScore method must be an instance of rdkit.Chem.rdchem.Mol.'
        feature, adj = self.mol_to_graph_feature(mol)
        retval = self.forward(feature, adj)
        return retval

    def smiListToScores(self, smi_list):
        mol_list = []
        for smi in smi_list:
            try:
                mol_list.append(Chem.MolFromSmiles(smi)) 
            except:
                mol_list.append(None)
        return self.molListToScores(mol_list)

    def molListToScores(self, mol_list, batch_size=128):
        save_dir = self.mols_to_graph_feature(mol_list)
        data_set = InferenceDataset
        data_loader = DataLoader(data_set, batch_size = batch_size, shuffle=False)
        scores = np.empty(0)
        device = self.device
        for i_batch,batch in enumerate(data_loader):
            x = batch['feature'].float().to(device)
            A = batch['adj'].float().to(device)
            y = batch['label'].long().to(device)
            scores = np.concatenate(scores, self.forward(x,A))
        return scores

