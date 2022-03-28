import torch
from torch.utils.data import Dataset
import pickle
import os

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
