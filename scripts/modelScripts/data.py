import pickle
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence as pad
import numpy as np

class TrainDataset():
    def __init__(self,data_dir, key_dir, mode):
        assert mode in ['train','val','test'], 'mode must be one of the followings: train, val, test'
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.key_dir = key_dir
        with open(f'{key_dir}/{mode}_keys.pkl','rb') as fr:
            self.key_list = pickle.load(fr) # list of keys

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self,idx):
        key = self.key_list[idx]
        with open(f'{self.data_dir}/{key}','rb') as fr:
            data=pickle.load(fr)
        return data

class InferenceDataset():
    def __init__(self, features, adjs, N_atoms):
        super().__init__()
        self.features = features
        self.adjs = adjs
        self.N_atoms = N_atoms

    def __len__(self):
        return len(self.features)

    def __getitem__(self,idx):
        data = dict()
        data['feature'] = self.features[idx]
        data['adj'] = self.adjs[idx]
        data['N_atom'] = self.N_atoms[idx]
        return data

def gat_collate_fn(batch):
    # adjacency: [N,N]
    # node_feature: [N,node]
    sample = dict()
    adj_batch=[]
    node_batch=[]
    label_batch=[]

    #max_num_atom = np.max(np.array([b['N_atom'] for b in batch]))
    max_num_atom = np.max(np.array([b['feature'].size(0) for b in batch]))
    node_dim = batch[0]['feature'].size(-1)
    for b in batch:
        num_atoms = b['feature'].size(0)

        adj = torch.zeros((max_num_atom,max_num_atom))
        adj[:num_atoms, :num_atoms] = b['adj']
        adj_batch.append(adj)

        node_batch.append(b['feature'])
        label_batch.append(b['label'])

    sample['adj']=torch.stack(adj_batch,0)
    sample['feature']=pad(node_batch,batch_first=True,padding_value=0.0)
    sample['label']=torch.tensor(label_batch)
    return sample        

def infer_collate_fn(batch):
    # adjacency: [N,N]
    # node_feature: [N,node]
    sample = dict()
    adj_batch=[]
    node_batch=[]

    max_num_atom = np.max(np.array([b['N_atom'] for b in batch]))
    node_dim = batch[0]['feature'].size(-1)
    for b in batch:
        num_atoms = b['feature'].size(0)

        adj = torch.zeros((max_num_atom,max_num_atom))
        adj[:num_atoms, :num_atoms] = b['adj']
        adj_batch.append(adj)

        node_batch.append(b['feature'])

    sample['adj']=torch.stack(adj_batch,0)
    sample['feature']=pad(node_batch,batch_first=True,padding_value=0.0)
    return sample        

#if __name__=='__main__':
#    from rdkit.Chem import MolFromSmiles as Mol
#    from rdkit.Chem.rdmolops import GetAdjacencyMatrix
#
#    def get_node_feature(mol):
#        num_atoms = mol.GetNumAtoms()
#        adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
#        adj = torch.from_numpy(adj).bool()
#        feature = torch.from_numpy(np.array([np.ones(36) for atom in mol.GetAtoms()])).bool()
#
#        return {'feature':feature, 'adj':adj, 'N_atom': num_atoms, 'label':1}
#
#    smi1, smi2, smi3 = 'CCCO', 'C1CCCCC1', 'CCCCCOCC'
#    print(smi1, smi2, smi3)
#    mol1, mol2, mol3 = Mol(smi1), Mol(smi2), Mol(smi3)
#    batch = [get_node_feature(mol1), get_node_feature(mol2), get_node_feature(mol3)]
#    new_batch = gat_collate_fn(batch)
#    print(new_batch['adj'])
#    #print(new_batch['feature'])
