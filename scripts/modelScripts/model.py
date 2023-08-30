from typing import List, Tuple

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader

mp.set_sharing_strategy("file_system")
import numpy as np

np.set_printoptions(suppress=True)
import time

import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

from .data import DFRscoreCollator, InferenceDataset
from .layers import FeedForward, GraphAttentionLayer
from .preprocessing import get_node_feature, sssr_to_ring_feature


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

    _CONV_DIM = 256
    _FC_DIM = 128
    _NUM_GAT_LAYER = 6
    _NUM_FC_LAYER = 2
    _NUM_HEADS = 8
    _FEATURE_SIZE = 49
    _MAX_STEP = 4
    _NUM_CORES = 4
    _OUT_DIM = 1
    _DROP_OUT = 0
    _PATH_TO_MODEL = "Not restored"

    def __init__(self, **kwargs):
        super().__init__()
        self._init_default_setting()
        for key in kwargs:
            if key not in self.__dict__:
                raise KeyError(f"{key} is an unknown key.")
        self.__dict__.update(kwargs)
        torch.set_num_threads(int(self.num_cores))

        self.dropout_layer = nn.Dropout(self.dropout)
        self.embedding = nn.Linear(self.feature_size, self.conv_dim, bias=False)
        self.GAT_layers = nn.ModuleList(
            [
                GraphAttentionLayer(
                    emb_dim=self.conv_dim,
                    num_heads=self.num_heads,
                    alpha=0.2,
                    bias=True,
                )
                for i in range(self.n_GAT_layer)
            ]
        )

        self.dense = FeedForward(
            in_dim=self.conv_dim,
            out_dim=self.out_dim,
            hidden_dims=[self.fc_dim] * (self.n_fc_layer),
            dropout=self.dropout,
        )
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.device = torch.device("cpu")

        # zero vectors
        self._zero_node_feature = torch.zeros((1, self.feature_size)) * torch.tensor(
            float("nan")
        )
        self._zero_adj = torch.zeros((1, 1))

        # parameter initialize
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            if param.dim() == 1:
                continue
            else:
                nn.init.xavier_normal_(param)

    def _init_default_setting(self):
        args = {
            "conv_dim": self._CONV_DIM,
            "fc_dim": self._FC_DIM,
            "n_GAT_layer": self._NUM_GAT_LAYER,
            "n_fc_layer": self._NUM_FC_LAYER,
            "num_heads": self._NUM_HEADS,
            "feature_size": self._FEATURE_SIZE,
            "max_step": self._MAX_STEP,
            "num_cores": self._NUM_CORES,
            "out_dim": self._OUT_DIM,
            "dropout": self._DROP_OUT,
            "path_to_model": self._PATH_TO_MODEL,
        }

        self.__dict__.update(args)

    def set_processors(self, num_cores: int):
        self.num_cores = num_cores
        torch.set_num_threads(int(self.num_cores))
        return

    @classmethod
    def from_trained_model(cls, *args, **kwargs):
        if "path_to_model" in kwargs and args:
            raise Exception("'from_trained_model' method got 2 models.")
        model = cls(**kwargs)
        if "path_to_model" in kwargs:
            model.restore(kwargs["path_to_model"])
        if args:
            model.restore(args[0])
        return model

    def forward(self, x, A):
        x = self.embedding(x)
        # convolution
        for layer in self.GAT_layers:
            x = layer(x, A)
            x = self.dropout_layer(x)

        # readout
        x = self.readout(x, A)

        # feedforward
        x = self.dense(x)
        retval = self.elu(x).squeeze(-1) + 1.5
        return retval

    def restore(self, path_to_model):
        if self.device == torch.device("cpu"):
            self.load_state_dict(
                torch.load(path_to_model, map_location=torch.device("cpu"))
            )
        else:
            self.load_state_dict(torch.load(path_to_model))
        self.path_to_model = path_to_model

    def readout(self, x, A):
        weighted_mask = self.get_mask(A)
        x = torch.mul(x, weighted_mask)
        x = x.sum(1)
        return x

    @staticmethod
    def get_mask(adj):
        # adj: [B,N,N]
        adj_sum = adj.sum(-1)
        mask = torch.where(
            adj_sum > 0,
            torch.tensor(1).to(adj_sum.device),
            torch.tensor(0).to(adj_sum.device),
        )
        num_atoms = mask.sum(-1).unsqueeze(-1)  # num_atoms: [B,1]
        mask = mask / num_atoms
        return mask.unsqueeze(-1)  # retval: [B,N,1]

    @staticmethod
    def mol_to_graph_feature(mol):
        if mol:
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
            return node_feature, adj, num_atoms
        else:
            return None, None, None

    # def mols_to_graph_feature(self, mol_list):
    #    """
    #    Args:
    #      mol_list: list of RDKit molecule objects.
    #    Returns:
    #      node_feats: torch tensor
    #      adjs: torch tensor
    #      N_atoms: list
    #    """

    #    since = time.time()
    #    with mp.Pool(processes=self.num_cores) as p:
    #        result = p.map(self.mol_to_graph_feature, mol_list)

    #    node_feats, adjs, N_atoms = [], [], []
    #    for node_feature, adj, num_atoms in result:
    #        if node_feature == None:
    #            node_feats.append(self._zero_node_feature)
    #            adjs.append(self._zero_adj)
    #            N_atoms.append(0)
    #        else:
    #            node_feats.append(node_feature)
    #            adjs.append(adj)
    #            N_atoms.append(num_atoms)

    #    return node_feats, adjs, N_atoms

    def smiToScore(self, smi: str, return_max_score_to_nan: bool = False) -> float:
        assert isinstance(
            smi, str
        ), "input of smiToScore method must be a string of SMILES."
        try:
            mol = Chem.MolFromSmiles(smi)
        except:
            raise AssertionError(
                "Failed to generate rdchem.Mol object from given string."
            )
        if not mol:
            raise AssertionError(
                "Failed to generate rdchem.Mol object from given string."
            )
        return self.molToScore(mol)

    def molToScore(self, mol: object, return_max_score_to_nan: bool = False) -> float:
        assert isinstance(
            mol, rdkit.Chem.rdchem.Mol
        ), "input of molToScore method must be an instance of rdkit.Chem.rdchem.Mol."
        feature, adj, _ = self.mol_to_graph_feature(mol)
        feature = feature.float().unsqueeze(0).to(self.device)
        adj = adj.float().unsqueeze(0).to(self.device)
        score = self.forward(feature, adj).squeeze(-1).to("cpu").detach()
        if score.isnan():
            if return_max_score_to_nan:
                return float(self.max_step + 1)
            else:
                return None
        else:
            return score.item()

    def smiListToScores(
        self, smi_list: list, batch_size=256, return_max_score_to_nan: bool = False
    ) -> np.array:
        data_set = InferenceDataset(smi_list=smi_list)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DFRscoreCollator(mode="inference"),
            num_workers=self.num_cores,
        )

        all_scores = []
        for batch in data_loader:
            x = batch["feature"].float().to(self.device)
            A = batch["adj"].float().to(self.device)
            scores = self.forward(x, A).to("cpu").detach().tolist()

            # Handling Mol object is not generated (None) cases
            for idx in batch["none_idx"]:
                scores.insert(idx, torch.nan)
            all_scores.extend(scores)

        all_scores = torch.tensor(all_scores)
        if return_max_score_to_nan:
            all_scores = torch.where(
                all_scores.isnan(), torch.tensor(float(self.max_step + 1)), all_scores
            )
        return all_scores.numpy()

    def molListToScores(
        self, mol_list: list, batch_size=256, return_max_score_to_nan: bool = False
    ) -> np.array:
        data_set = InferenceDataset(mol_list=mol_list)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DFRscoreCollator(mode="inference"),
            num_workers=self.num_cores,
        )

        all_scores = []
        for batch in data_loader:
            x = batch["feature"].float().to(self.device)
            A = batch["adj"].float().to(self.device)

            scores = self.forward(x, A).to("cpu").detach().tolist()

            # Handling Mol object is not generated (None) cases
            for idx in batch["none_idx"]:
                scores.insert(idx, torch.nan)
            all_scores.extend(scores)

        all_scores = torch.tensor(all_scores)
        if return_max_score_to_nan:
            all_scores = torch.where(
                all_scores.isnan(), torch.tensor(float(self.max_step + 1)), all_scores
            )

        return scores.numpy()

    def _preprocessing_time_check(self, smi_list: list, batch_size=256) -> float:
        since = time.time()
        data_set = InferenceDataset(smi_list=smi_list)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=DFRscoreCollator(mode="inference"),
            num_workers=self.num_cores,
        )
        for batch in data_loader:
            continue
        end = time.time()
        return end - since

    def filterWithScore(self, data_list, criteria) -> Tuple[List, List]:
        """
        Args:
          data_list: list of SMILES or rdchem.Mol.
          criteria: refine within what stpes(scalar value).
        Returns:
          passed_data: list of passed SMILES or rdchem.Mol.
          passed_idx: list of passed indice.
        """
        if type(data_list[0]) == str:
            scores = self.smiListToScores(data_list)
        elif isinstance(data_list[0], Chem.rdchem.Mol):
            scores = self.molListToScores(data_list)
        else:
            raise TypeError("Given data is neither SMILES or rdchem.Mol object.")

        bPassed = scores < criteria + 0.5
        passed_data, passed_idx = [], []
        for i, data in enumerate(data_list):
            if bPassed[i]:
                passed_data.append(data)
                passed_idx.append(i)

        return passed_data, passed_idx

    def cuda(self):
        super().cuda()
        self.device = self.embedding.weight.device
        return self

    def to(self, torch_device):
        super().to(torch_device)
        self.device = self.embedding.weight.device
        return self

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
        return (
            f"{self.__class__.__name__}(\n"
            + f"loaded_model: {self.path_to_model}\n"
            + f"conv_dim: {self.conv_dim}\n"
            + f"fc_dim: {self.fc_dim}\n"
            + f"num_GAT_layer: {self.n_GAT_layer}\n"
            + f"num_FC_layer: {self.n_fc_layer}\n"
            + f"num_heads: {self.num_heads}\n"
            + f"feature_size: {self.feature_size}\n"
            + f"max_step: {self.max_step}\n"
            + f"num_cores: {self.num_cores}\n"
            + f"dropout: {self.dropout}\n"
            + f"device: {self.device}\n"
            + ")"
        )


if __name__ == "__main__":
    from rdkit.Chem import MolFromSmiles as Mol

    smi1, smi2, smi3 = "CCCO", "C1CCCCC1", "CCCCCOCC"
    print(smi1, smi2, smi3)
    mol1, mol2, mol3 = Mol(smi1), Mol(smi2), Mol(smi3)

    x = torch.zeros(3, 8, 10)
    x[0, :4] = torch.arange(10).repeat(4, 1)
    x[1, :6] = torch.arange(10).repeat(6, 1)
    x[2, :8] = torch.arange(10).repeat(8, 1)
    print(x)

    adjs = torch.zeros((3, 8, 8))
    adjs[0, :4, :4] = torch.from_numpy(
        GetAdjacencyMatrix(mol1) + np.eye(mol1.GetNumAtoms())
    )
    adjs[1, :6, :6] = torch.from_numpy(
        GetAdjacencyMatrix(mol2) + np.eye(mol2.GetNumAtoms())
    )
    adjs[2, :8, :8] = torch.from_numpy(
        GetAdjacencyMatrix(mol3) + np.eye(mol3.GetNumAtoms())
    )
    weighted_mask = DFRscore.get_mask(adjs)
    print(weighted_mask)

    x = torch.mul(x, weighted_mask)
    print(x)
    x = x.sum(1)
    print(x)
