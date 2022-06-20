import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from torch.nn.utils.rnn import pad_sequence as pad

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
    def __init__(self, features, adjs):
        super().__init__()
        self.features = features
        self.adjs = adjs

    def __len__(self):
        return int(self.features.shape[0])

    def __getitem__(self,idx):
        data = dict()
        data['feature'] = self.features[idx]
        data['adj'] = self.adjs[idx]
        return data

def mpnn_collate_fn(batch):
    # adjacency: [N,N]
    # node_feature: [N,node]
    # edge_feature: [N,N,edge]
    sample = dict()
    adj_batch=[]
    node_batch=[]
    edge_batch=[]
    label_batch=[]

    max_num_atom = np.max(np.array([b['N_atom'] for b in batch]))
    node_dim, edge_dim = batch[0]['node_feat'].size(-1), batch[0]['edge_feat'].size(-1)
    for b in batch:
        num_atoms = b['node_feat'].size(0)

        adj = torch.zeros((max_num_atom,max_num_atom))
        #node = np.zeros((max_num_atom,node_dim))
        edge = torch.zeros((max_num_atom,max_num_atom,edge_dim))
        
        adj[:num_atoms, :num_atoms] = b['adj']
        adj_batch.append(adj)

        node_batch.append(b['node_feat'])

        edge[:num_atoms, :num_atoms, :edge_dim] = b['edge_feat']
        edge_batch.append(edge)

        label_batch.append(b['label'])

    sample['adj']=torch.stack(adj_batch,0)
    sample['node']=pad(node_batch,batch_first=True,padding_value=0.0)
    sample['edge']=torch.stack(edge_batch,0)
    sample['label']=torch.tensor(label_batch)
    return sample        

