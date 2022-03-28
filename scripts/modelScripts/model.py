import os, pickle
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
np.set_printoptions(suppress=True)

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
#from rdkit.Chem.rdmolops import GetFormalCharge

from scripts.modelScripts.layers import GraphAttentionLayer
from scripts.modelScripts.preprocessing import get_atoms_feature, sssr_to_ring_feature
from scripts.modelScripts.data import InferenceDataset

#MAX_NUM_ATOM = 64
#CONV_DIM = 256
#FC_DIM = 128
#NUM_GAT_LAYER = 6
#NUM_FC_LAYER = 4
#NUM_HEADS= 8
#NUM_CLASS= 5
#LEN_FEATURES = 36

class SVS(nn.Module):
    """
    Model to predict synthesizability for virtual screening.
    You can select which model to load using 'restore' method.
      (It needs path to model state_dict file)
    If you want to get SVS score using this model,
      1) model = SVS(args)  (if you have CUDA system, you can use model=SVS(args).cuda())
      2-1) From SMILES string:
        score = model.smiToScore(<SMILES>) / scores = model.smiListToScores(<list of SMILES>)
      2-2) From RDKit Molecule object:
        score = model.molToScore(<Molecule object>) / scores = model.molListToScores(<list of Molecule objects>)
    """
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
            dropout:float=0
            ):
        super().__init__()
        self.conv_dim = conv_dim
        self.fc_dim = fc_dim
        self.n_GAT_layer = n_GAT_layer
        self.n_fc_layer = n_fc_layer
        self.num_heads = num_heads
        self.len_features = len_features
        self.max_num_atoms = max_num_atoms
        self.num_class = num_class

        self.embedding = nn.Linear(len_features,conv_dim)
        self.GAT_layers=nn.ModuleList(
                [GraphAttentionLayer(
                    emb_dim=conv_dim,
                    num_heads=num_heads,
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
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.path_to_model = 'Not restored'

    def forward(self,x,A):
        x = self.embedding(x)
        for layer in self.GAT_layers:
            x = layer(x, A)         # output was already applied with ELU.
            x = self.dropout(x)
        x = retval = x.mean(1)
        for idx, layer in enumerate(self.fc_layers):
            x = layer(x)
            x = self.relu(x)
            x = self.dropout(x)
        retval = self.pred_layer(x)
        return retval

    def restore(self, path_to_model):
        if self._cuda_is_available():
            self.load_state_dict(torch.load(path_to_model))
        else:
            self.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        self.path_to_model = path_to_model
        pass

    def mol_to_graph_feature(self, mol):
        sssr = Chem.GetSymmSSSR(mol)
        num_atoms = mol.GetNumAtoms()
        adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
        padded_adj = np.zeros((self.max_num_atoms, self.max_num_atoms))
        padded_adj[:num_atoms,:num_atoms] = adj
        feature = []
        atoms = mol.GetAtoms()
        ring_feature = sssr_to_ring_feature(sssr, num_atoms)
        for atom in atoms:
            feature.append(get_atoms_feature(atom))
        feature = np.concatenate([np.array(feature), ring_feature],axis=1)  # 30 + 6 = 36

        padded_feature = np.zeros((self.max_num_atoms, self.len_features))
        padded_feature[:num_atoms,:self.len_features] = feature
        padded_feature = torch.from_numpy(padded_feature)
        padded_adj = torch.from_numpy(padded_adj)

        return padded_feature, padded_adj

    def mols_to_graph_feature(self, mol_list):
        """
        Args:
          mol_list: list of RDKit molecule objects.
        Returns: 
          padded_features: torch tensor.
          padded_adjs: torch tensor.
        """
        padded_features, padded_adjs = [], []
        for mol in mol_list:
            if mol:
                padded_feature, padded_adj = self.mol_to_graph_feature(mol)
                padded_features.append(padded_feature)
                padded_adjs.append(padded_adj)
            else:
                padded_features.append(torch.nan)
                padded_adjs.append(torch.nan)
        padded_features = torch.stack(padded_features,dim=0)
        padded_adjs = torch.stack(padded_adjs,dim=0)

        return padded_features, padded_adjs

    def smiToScore(self, smi:str, get_probs=False) -> tuple:
        assert isinstance(smi, str), 'input of smiToScore method must be a string of SMILES.'
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            raise AssertionError('input of smiToScore method must be a string of SMILES.')
        if not mol:
            raise AssertionError('Failed to generate rdchem.Mol object from given string.')
        return self.molToScore(mol, get_probs)

    def molToScore(self, mol:object, get_probs=False):
        assert isinstance(mol, rdkit.Chem.rdchem.Mol), \
            'input of molToScore method must be an instance of rdkit.Chem.rdchem.Mol.'
        self.device = self.embedding.weight.device
        feature, adj = self.mol_to_graph_feature(mol)
        feature = feature.float().unsqueeze(0).to(self.device)
        adj= adj.float().unsqueeze(0).to(self.device)
        probs = self.softmax(self.forward(feature, adj)).to('cpu').detach().numpy()[0]
        if get_probs:
            retval = np.round_(probs, decimals=4)
        else:
            labels = np.array(range(self.num_class))
            labels[0] = self.num_class
            retval = np.sum(np.multiply(probs, labels),axis=0)
            retval = np.round_(retval, decimals=4)
        return retval

    def smiListToScores(self, smi_list, batch_size=256, get_probs=False):
        mol_list = []
        for smi in smi_list:
            try:
                mol_list.append(Chem.MolFromSmiles(smi)) 
            except:
                mol_list.append(None)
        return self.molListToScores(mol_list, batch_size, get_probs)

    def molListToScores(self, mol_list, batch_size=256, get_probs=False):
        # TODO: 너무 크다 싶으면 10000개 씩 잘라서 하는 등 활용 가능.
        self.device = self.embedding.weight.device
        padded_features, padded_adjs = self.mols_to_graph_feature(mol_list)
        data_set = InferenceDataset(features=padded_features, adjs=padded_adjs)
        data_loader = DataLoader(data_set, batch_size = batch_size, shuffle=False)
        scores = []

        for i_batch,batch in enumerate(data_loader):
            x = batch['feature'].float().to(self.device)
            A = batch['adj'].float().to(self.device)
            scores.append(self.forward(x,A).to('cpu').detach())
        probs = self.softmax(torch.cat(scores)).numpy()
        #probs = np.round_(probs, decimals=4)
        if get_probs:
            retval = np.round_(probs, decimals=4)
        else:
            labels = np.array(range(self.num_class))
            labels[0] = self.num_class
            retval = np.sum(np.multiply(probs, labels),axis=1)
            #retval = np.sum(np.multiply(probs, np.array(range(self.num_class))),axis=1)
            retval = np.round_(retval, decimals=4)
        return retval
        #return scores

    def _cuda_is_available(self):
        return torch.cuda.is_available()

    def __repr__(self):
        return f'{self.__class__.__name__}(\n' + \
                f'  loaded_model: {self.path_to_model}\n'+ \
                f'  conv_dim: {self.conv_dim}\n'+ \
                f'  fc_dim: {self.fc_dim}\n'+ \
                f'  n_GAT_layer: {self.n_GAT_layer}\n'+ \
                f'  n_fc_layer: {self.n_fc_layer}\n'+ \
                f'  num_heads: {self.num_heads}\n'+ \
                f'  len_features: {self.max_num_atoms}\n'+ \
                f'  num_class: {self.num_class}\n' + \
                ')'
