import pickle

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.nn.utils.rnn import pad_sequence as pad
from torch.utils.data import Dataset

from .preprocessing import get_node_feature, sssr_to_ring_feature


class TrainDataset(Dataset):
    def __init__(self, data_dir, key_dir, mode):
        assert mode in [
            "train",
            "val",
            "test",
        ], "mode must be one of the followings: train, val, test"
        super().__init__()
        self.mode = mode
        self.data_dir = data_dir
        self.key_dir = key_dir
        with open(f"{key_dir}/{mode}_keys.pkl", "rb") as fr:
            self.key_list = pickle.load(fr)  # list of keys

    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, idx):
        key = self.key_list[idx]
        with open(f"{self.data_dir}/{key}", "rb") as fr:
            data = pickle.load(fr)
        return data


class InferenceDataset(Dataset):
    def __init__(self, smi_list: list = None, mol_list: list = None):
        super().__init__()
        if all([smi_list, mol_list]) or not any([smi_list, mol_list]):
            raise "input for InferenceDataset is wrong."

        if smi_list:
            self.data_list = smi_list
            self.data_type = "SMILES"
        elif mol_list:
            self.data_list = mol_list
            self.data_type = "Mol"

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.data_type == "SMILES":
            return self._smi_to_graph_feature(self.data_list[idx])

        elif self.data_type == "Mol":
            return self._mol_to_graph_feature(self.data_list[idx])

    @classmethod
    def _smi_to_graph_feature(cls, smi: str):
        # initialize return value
        data = {"feature": None, "adj": None, "N_atom": None}

        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            return data
        if mol:
            return cls._mol_to_graph_feature(mol)
        else:
            return data

    @staticmethod
    def _mol_to_graph_feature(mol: Chem.rdchem.Mol):
        # initialize return value
        data = {"feature": None, "adj": None, "N_atom": None}

        if not isinstance(mol, Chem.rdchem.Mol):
            return data

        num_atoms = mol.GetNumAtoms()

        # 1. Adjacency
        adj = torch.from_numpy(
            np.asarray(GetAdjacencyMatrix(mol), dtype=bool)
            + np.eye(num_atoms, dtype=bool)
        )

        # 2. Node Feature
        sssr = Chem.GetSymmSSSR(mol)
        node_feature = []
        for atom in mol.GetAtoms():
            node_feature.append(get_node_feature(atom))
        ring_feature = sssr_to_ring_feature(sssr, num_atoms)
        node_feature = np.concatenate(
            [np.stack(node_feature, axis=0), ring_feature], axis=1
        )
        node_feature = torch.from_numpy(node_feature)

        data["feature"] = node_feature
        data["adj"] = adj
        data["N_atom"] = num_atoms

        return data


class DFRscoreCollator:
    def __init__(self, mode: str):
        assert mode in ["train", "inference"]
        self.mode = mode

    def __call__(self, batch):
        # adjacency: [N,N]
        # node_feature: [N,node]
        sample = dict()
        adj_batch = []
        node_batch = []
        label_batch = []
        none_idx = []

        max_num_atom = np.max(
            np.array(
                [b["feature"].size(0) if b["feature"] is not None else 0 for b in batch]
            )
        )

        for data_idx, b in enumerate(batch):
            if b["feature"] is None:
                none_idx.append(data_idx)
                continue

            else:
                num_atoms = b["feature"].size(0)

                adj = torch.zeros((max_num_atom, max_num_atom))
                adj[:num_atoms, :num_atoms] = b["adj"]
                adj_batch.append(adj)

                node_batch.append(b["feature"])
                if self.mode == "train":
                    label_batch.append(b["label"])

        sample["adj"] = torch.stack(adj_batch, 0)
        sample["feature"] = pad(node_batch, batch_first=True, padding_value=0.0)
        sample["none_idx"] = none_idx

        if self.mode == "train":
            sample["label"] = torch.tensor(label_batch)

        return sample


# def infer_collate_fn(batch):
#    # adjacency: [N,N]
#    # node_feature: [N,node]
#    sample = dict()
#    adj_batch = []
#    node_batch = []
#
#    max_num_atom = np.max(np.array([b["N_atom"] for b in batch]))
#    node_dim = batch[0]["feature"].size(-1)
#    for b in batch:
#        num_atoms = b["feature"].size(0)
#
#        adj = torch.zeros((max_num_atom, max_num_atom))
#        adj[:num_atoms, :num_atoms] = b["adj"]
#        adj_batch.append(adj)
#
#        node_batch.append(b["feature"])
#
#    sample["adj"] = torch.stack(adj_batch, 0)
#    sample["feature"] = pad(node_batch, batch_first=True, padding_value=0.0)
#    return sample


# if __name__=='__main__':
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
