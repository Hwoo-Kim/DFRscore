import os, pickle, sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
np.set_printoptions(suppress=True)

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
#from rdkit.Chem.rdmolops import GetFormalCharge

sys.path.append(f'{os.path.abspath(os.path.dirname(__file__))}')
#from .layers import GraphAttentionLayer
from layers_mpnn import MessageFunction, UpdateFunction, Readout

from preprocessing_mpnn import get_node_feature, get_edge_feature, sssr_to_ring_feature
from data_mpnn import InferenceDataset


class DFRscore(nn.Module):
    """
    Model to predict synthesizability for virtual screening.
    You can select which model to load using 'restore' method.
      (It needs path to model state_dict file)
    If you want to get DFRscore using this model,
      1) model = DFRscore(args)  (if you have CUDA system, you can use model=DFRscore(args).cuda())
      2-1) From SMILES string:
        score = model.smiToScore(<SMILES>) / scores = model.smiListToScores(<list of SMILES>)
      2-2) From RDKit Molecule object:
        score = model.molToScore(<Molecule object>) / scores = model.molListToScores(<list of Molecule objects>)
    """
    _EDGE_FEATURE=5
    _NODE_FEATURE=30
    _HIDDEN_DIM=80
    _MESSAGE_DIM=80
    _NUM_LAYER=3
    _MAX_STEP = 4

    def __init__(
            self,
            edge_dim=_EDGE_FEATURE,
            node_dim=_NODE_FEATURE,
            hidden_dim=_HIDDEN_DIM,
            message_dim=_MESSAGE_DIM,
            num_layers=_NUM_LAYER,
            max_step=_MAX_STEP,

            dropout:float=0
            ):
        super().__init__()
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        self.num_layers = num_layers
        self.max_step = max_step
        self.out_dim = 1
        self.dropout = nn.Dropout(dropout)

        self.message = MessageFunction(self.edge_dim,self.hidden_dim,self.message_dim)
        self.update = UpdateFunction(self.hidden_dim,self.message_dim)
        self.readout = Readout(self.hidden_dim,self.out_dim)
        self.node_embedding = nn.Linear(self.node_dim,self.hidden_dim)
        self.edge_embedding = self.message.edge_embedding()

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)

        self.path_to_model = 'Not restored'
        self.device = torch.device('cpu')

    def forward(self, x_in, e, A):
        """
        x_in: [B,N,node]
        e:    [B,N,N,edge]
        A:    [B,N,N]
        """
        masking = (torch.sum(x_in,2)>0).unsqueeze(-1)
        x_hidden = [self.node_embedding(x_in)*masking]              # [B,N,F]
        B,N,F = x_hidden[0].size()
        e_embed = self.edge_embedding(e)
        e_embed = e_embed.view(B,N,N,self.message_dim,self.hidden_dim)

        for l in range(self.num_layers):
            message = self.message(x_hidden[l], e_embed)            # [B,N,F], [B*N*N,edge] -> [B*N*N,message]
            message = torch.einsum('ijk,ijkl->ijl',A,message)       # [B,N,message]

            h_l = self.update(x_hidden[l],message)                  # [B,N,F]
            h_l = h_l * masking
            x_hidden.append(h_l)

        retval = self.readout(x_hidden[0], x_hidden[-1],masking)      # [B,N,1]
        return self.elu(retval).squeeze(-1)+1.5

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
        if num_atoms > self.max_num_atoms:
            return None, None
        adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
        padded_adj = np.zeros((self.max_num_atoms, self.max_num_atoms))
        padded_adj[:num_atoms,:num_atoms] = adj
        feature = []
        atoms = mol.GetAtoms()
        ring_feature = sssr_to_ring_feature(sssr, num_atoms)
        for atom in atoms:
            feature.append(get_node_feature(atom))
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
                if padded_feature is None:
                    padded_features.append(torch.tensor(float('nan')).repeat([self.max_num_atoms, self.len_features]))
                    padded_adjs.append(torch.tensor(float('nan')).repeat([self.max_num_atoms, self.max_num_atoms]))
                    continue
                padded_features.append(padded_feature)
                padded_adjs.append(padded_adj)
            else:
                padded_features.append(torch.tensor(float('nan')).repeat([self.max_num_atoms, self.len_features]))
                padded_adjs.append(torch.tensor(float('nan')).repeat([self.max_num_atoms, self.max_num_atoms]))
        padded_features = torch.stack(padded_features,dim=0)
        padded_adjs = torch.stack(padded_adjs,dim=0)

        return padded_features, padded_adjs

    def smiToScore(self, smi:str) -> torch.tensor:
        assert isinstance(smi, str), 'input of smiToScore method must be a string of SMILES.'
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            raise AssertionError('input of smiToScore method must be a string of SMILES.')
        if not mol:
            raise AssertionError('Failed to generate rdchem.Mol object from given string.')
        return self.molToScore(mol)

    def molToScore(self, mol:object):
        assert isinstance(mol, rdkit.Chem.rdchem.Mol), \
            'input of molToScore method must be an instance of rdkit.Chem.rdchem.Mol.'
        feature, adj = self.mol_to_graph_feature(mol)
        feature = feature.float().unsqueeze(0).to(self.device)
        adj= adj.float().unsqueeze(0).to(self.device)
        score = self.forward(feature, adj).to('cpu').detach().numpy()
        return torch.where(score.isnan(), torch.tensor(float(self.max_stpe+1)), score)

    def smiListToScores(self, smi_list, batch_size=256):
        mol_list = []
        for smi in smi_list:
            try:
                mol_list.append(Chem.MolFromSmiles(smi)) 
            except:
                mol_list.append(None)
        return self.molListToScores(mol_list, batch_size)

    def molListToScores(self, mol_list, batch_size=256):
        # TODO: 너무 크다 싶으면 10000개 씩 잘라서 하는 등 활용 가능.
        padded_features, padded_adjs = self.mols_to_graph_feature(mol_list)
        data_set = InferenceDataset(features=padded_features, adjs=padded_adjs)
        data_loader = DataLoader(data_set, batch_size = batch_size, shuffle=False)
        scores = []

        for i_batch,batch in enumerate(data_loader):
            x = batch['feature'].float().to(self.device)
            A = batch['adj'].float().to(self.device)
            scores.append(self.forward(x,A).to('cpu').detach())
        scores = torch.cat(scores).squeeze(-1)
        scores = torch.where(scores.isnan(), torch.tensor(float(self.max_step+1)), scores)
        return scores.numpy()

    def cuda(self):
        _DFR_SCORE = super().cuda()
        self.device = _DFR_SCORE.node_embedding.weight.device
        return _DFR_SCORE

    def to(self, torch_device):
        _DFR_SCORE = super().to(torch_device)
        self.device = _DFR_SCORE.node_embedding.weight.device
        return _DFR_SCORE

    @staticmethod
    def _cuda_is_available():
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
                f'  out_dim: {self.out_dim}\n' + \
                ')'


if __name__=='__main__':
    dfr = DFRscore()
    x_in = torch.zeros(2,20,36)
    x_in[:,:16,:] = torch.ones(2,16,36)
    x_in[0,:,:] = x_in[0,:,:]*2
    print(x_in)
    e = torch.ones(2,20,20,4)
    A = torch.ones(2,20,20)
    print(dfr(x_in,e,A))
