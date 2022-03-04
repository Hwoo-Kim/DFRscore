import torch
from torch.utils.data import Dataset
import pickle
import os

#class GraphDataset(Dataset):
class TrainDataset():
    def __init__(self,data_dir, key_dir, mode):
        assert mode in ['train','val','test'], 'mode must be one of the followings: train, val, test'
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.key_dir = key_dir
        self.data_list = os.listdir(data_dir)
        with open(f'{key_dir}/{mode}_keys.pkl','rb') as fr:
            self.key_list = pickle.load(fr) # list of keys

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self,idx):
        key = self.key_list[idx]
        with open(f'{self.data_dir}/{key}','rb') as fr:
            data=pickle.load(fr)
        #data['key']=key
        return data

class InferenceDataset():
    def __init__(self, save_dir):
        super().__init__()
        self.save_dir = save_dir
        self.file_names = os.lisrdir(self.save_dir)

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self,idx):
        with open(f'{self.save_dir}/{idx}.pkl','rb') as fr:
            data=pickle.load(fr)
        #data['key']=key
        return data
