import multiprocessing as mp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
np.set_printoptions(suppress=True)
import time

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from .layers import FeedForward, GraphAttentionLayer
from .preprocessing import get_node_feature, sssr_to_ring_feature
from .data import InferenceDataset, infer_collate_fn


class DFRscore(nn.Module):
    """
    Model to estimate synthetic complexity for large-scale virtual screening.
    You can select which model to load using 'restore' method.
      (It needs path to model state_dict file)

    To get score using this class,
      1. model = DFRscore(args)  (if you have CUDA system, you can use model=DFRscore(args).cuda())
      2. model.restore(<path_to_model>)
      3-1. From SMILES string:
        score = model.smiToScore(<SMILES>) / scores = model.smiListToScores(<list of SMILES>)
      3-2. From RDKit Molecule object:
        score = model.molToScore(<Molecule object>) / scores = model.molListToScores(<list of Molecule objects>)
    """
    _CONV_DIM = 512
    _FC_DIM = 256
    _NUM_GAT_LAYER = 5
    _NUM_FC_LAYER = 4
    _NUM_HEADS = 8
    _LEN_FEATURES = 36
    _MAX_STEP = 4
    _NUM_CORES = 4
    _OUT_DIM = 1
    _DROP_OUT = 0
    _PATH_TO_MODEL = 'Not restored'

    def __init__(self, **kwargs):
        super().__init__()
        self._init_default_setting()
        self.__dict__.update(kwargs)

        self.dropout = nn.Dropout(self.dropout)
        self.embedding = nn.Linear(self.len_features, self.conv_dim, bias=False)
        self.GAT_layers=nn.ModuleList(
                [GraphAttentionLayer(
                    emb_dim=self.conv_dim,
                    num_heads=self.num_heads,
                    alpha=0.2,
                    bias=True,
                    dropout=self.dropout
                    )
                    for i in range(self.n_GAT_layer)]
                )

        self.dense = FeedForward(
                in_dim=self.conv_dim,
                out_dim=self.out_dim,
                hidden_dims=[self.fc_dim]*(self.n_fc_layer-1),
                dropout=self.dropout
                )
        #self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device('cpu')

        # zero vectors
        self._zero_node_feature = torch.zeros((1,self.len_features))*torch.tensor(float('nan'))
        self._zero_adj = torch.zeros((1,1))

    def _init_default_setting(self):
        args = {'conv_dim': self._CONV_DIM,
                'fc_dim': self._FC_DIM,
                'n_GAT_layer': self._NUM_GAT_LAYER,
                'n_fc_layer': self._NUM_FC_LAYER,
                'num_heads': self._NUM_HEADS,
                'len_features': self._LEN_FEATURES,
                'max_step': self._MAX_STEP,
                'num_cores': self._NUM_CORES,
                'out_dim': self._OUT_DIM,
                'dropout': self._DROP_OUT,
                'path_to_model': self._PATH_TO_MODEL}

        self.__dict__.update(args)

    @classmethod
    def from_trained_model(cls, **kwargs):
        model = cls()
        for key in kwargs:
            if not key in model.__dict__:
                raise KeyError(f'{key} is an unknown key.')
        if 'path_to_model' in kwargs:
            model.restore(kwargs['path_to_model'])
        model.__dict__.update(kwargs)
        return model

    def forward(self, x, A):
        x = self.embedding(x)
        for layer in self.GAT_layers:
            x = layer(x, A)
        x = x.mean(1)
        x = self.dense(x)
        retval = self.elu(x).squeeze(-1)+1.5
        return retval

    def restore(self, path_to_model):
        if self.device==torch.device('cpu'):
            self.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(path_to_model))
        self.path_to_model = path_to_model

    def mol_to_graph_feature(self, mol):
        if mol:
            num_atoms = mol.GetNumAtoms()

            # 1. Adjacency
            adj = torch.from_numpy(GetAdjacencyMatrix(mol) + np.eye(num_atoms))

            # 2. Node Feature
            sssr = Chem.GetSymmSSSR(mol)
            node_feature = []
            for atom in mol.GetAtoms():
                node_feature.append(get_node_feature(atom))
            ring_feature = sssr_to_ring_feature(sssr, num_atoms)
            node_feature = np.concatenate([np.array(node_feature), ring_feature],axis=1)
            node_feature = torch.from_numpy(node_feature).bool()
            return node_feature, adj, num_atoms
        else:
            return self._zero_node_feature, self._zero_adj, 0

    def do_mols_to_graph_feature(self, mol_list, worker, return_dict):
        node_feats , adjs, N_atoms = [], [], []
        for mol in mol_list:
            result = self.mol_to_graph_feature(mol)
            node_feats.append(result[0])
            adjs.append(result[1])
            N_atoms.append(result[2])
        return_dict[worker] = (node_feats, adjs, N_atoms)
        return

    # TODO: Process implementation
    #def mols_to_graph_feature(self, mol_list):
    #    """
    #    Args:
    #      mol_list: list of RDKit molecule objects.
    #    Returns: 
    #      node_feats: torch tensor
    #      adjs: torch tensor
    #      N_atoms: list
    #    """
    #    print('Mol To Graph Feature')
    #    t1 = time.time()
    #    num_mols = len(mol_list)
    #    batch_size = num_mols//self.num_cores
    #    if batch_size*self.num_cores!=num_mols: batch_size+=1

    #    manager = mp.Manager()
    #    return_dict = manager.dict()
    #    procs = []
    #    for worker in range(self.num_cores):
    #        p = mp.Process(target=self.do_mols_to_graph_feature, args=(mol_list[batch_size*worker:batch_size*(worker+1)], worker, return_dict))
    #        procs.append(p)
    #        p.start()

    #    for p in procs:
    #        p.join()
    #    print(f'Fin {time.time()-t1}')

    #    node_feats , adjs, N_atoms = [], [], []
    #    for node_feature, adj, num_atoms in list(return_dict.values()):
    #        node_feats += node_feature
    #        adjs += adj
    #        N_atoms += num_atoms

    #    return node_feats, adjs, N_atoms

    def mols_to_graph_feature(self, mol_list):
        """
        Args:
          mol_list: list of RDKit molecule objects.
        Returns: 
          node_feats: torch tensor
          adjs: torch tensor
          N_atoms: list
        """

        print('Mol To Graph Feature')
        since = time.time()
        with mp.Pool(processes=self.num_cores) as p:
            result = p.map(self.mol_to_graph_feature, mol_list)
        print(f'Fin {time.time()-since}')

        node_feats , adjs, N_atoms = [], [], []
        for node_feature, adj, num_atoms in result:
            node_feats.append(node_feature)
            adjs.append(adj)
            N_atoms.append(num_atoms)

        return node_feats, adjs, N_atoms

    def smiToScore(self, smi:str) -> torch.tensor:
        assert isinstance(smi, str), 'input of smiToScore method must be a string of SMILES.'
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            raise AssertionError('Failed to generate rdchem.Mol object from given string.')
        if not mol:
            raise AssertionError('Failed to generate rdchem.Mol object from given string.')
        return self.molToScore(mol)

    def molToScore(self, mol:object):
        assert isinstance(mol, rdkit.Chem.rdchem.Mol), \
            'input of molToScore method must be an instance of rdkit.Chem.rdchem.Mol.'
        feature, adj = self.mol_to_graph_feature(mol)
        feature = feature.float().unsqueeze(0).to(self.device)
        adji = adj.float().unsqueeze(0).to(self.device)
        score = self.forward(feature, adj).to('cpu').detach().numpy()
        return torch.where(score.isnan(), torch.tensor(float(self.max_stpe+1)), score)

    def smiListToScores(self, smi_list, batch_size=256):
        with mp.Pool(self.num_cores) as p:
            mol_list = p.map(self._mol_from_smiles, smi_list)
        return self.molListToScores(mol_list, batch_size)

    def molListToScores(self, mol_list, batch_size=256):
        node_feats, adjs, N_atoms = self.mols_to_graph_feature(mol_list)

        data_set = InferenceDataset(node_feats, adjs, N_atoms)
        data_loader = DataLoader(data_set,
                batch_size = batch_size,
                shuffle=False, 
                collate_fn=infer_collate_fn,
                num_workers=0
                )

        scores = []
        for batch in data_loader:
            x = batch['feature'].float().to(self.device)
            A = batch['adj'].float().to(self.device)
            scores.append(self.forward(x,A).to('cpu').detach())
        scores = torch.cat(scores).squeeze(-1)
        scores = torch.where(scores.isnan(), torch.tensor(float(self.max_step+1)), scores)
        return scores.numpy()

    def cuda(self):
        _DFR = super().cuda()
        self.device = _DFR.embedding.weight.device
        return _DFR

    def to(self, torch_device):
        _DFR = super().to(torch_device)
        self.device = _DFR.embedding.weight.device
        return _DFR
    
    @staticmethod
    def _mol_from_smiles(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            return None
        return mol

    @staticmethod
    def _cuda_is_available():
        return torch.cuda.is_available()

    @staticmethod
    def get_all_indice(L, query):
        return [i for i, x in enumerate(L) if x == query]

    def __repr__(self):
        return f'{self.__class__.__name__}(\n' + \
                f'  loaded_model: {self.path_to_model}\n'+ \
                f'  conv_dim: {self.conv_dim}\n'+ \
                f'  fc_dim: {self.fc_dim}\n'+ \
                f'  n_GAT_layer: {self.n_GAT_layer}\n'+ \
                f'  n_fc_layer: {self.n_fc_layer}\n'+ \
                f'  num_heads: {self.num_heads}\n'+ \
                f'  len_features: {self.len_features}\n' + \
                f'  max_step: {self.max_step}\n' + \
                f'  num_cores: {self.num_cores}\n' + \
                f'  dropout: {self.dropout}\n' + \
                f'  device: {self.device}\n' + \
                ')'
if __name__ == '__main__':
    model = DFRscore()
    print(model)
