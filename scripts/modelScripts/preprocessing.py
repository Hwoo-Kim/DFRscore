import os
import pickle
import queue
import random
import time
from multiprocessing import Process, Queue

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix

DATA_SPLIT_SEED = 1024
PROBLEM = "regression"


# 1. Data splitting functions
def processing_data(data_dir, max_step, logger, num_data):
    global DATA_SPLIT_SEED, PROBLEM
    random.seed(DATA_SPLIT_SEED)
    # splitting
    smis_w_label = []
    for i in range(max_step + 1):
        label = float(i)
        if i == 0:
            with open(f"{data_dir}/neg{max_step}.smi", "r") as fr:
                smis = fr.read().splitlines()
            if PROBLEM == "regression":
                label = float(max_step + 1)
        else:
            with open(f"{data_dir}/pos{i}.smi", "r") as fr:
                smis = fr.read().splitlines()
        random.shuffle(smis)
        each_class = []
        for smi in smis:
            each_class.append(f"{smi}\t{label}\n")
        smis_w_label.append(each_class)

    # handling data size
    logger("  Number of data for each class (Neg, pos1, pos2, ...):")
    for i in range(max_step + 1):
        logger(f"  {len(smis_w_label[i])}\t", end="")
    each_class_size = num_data // (max_step * 2)
    class_sizes = []
    for i in range(max_step + 1):
        if i == 0:
            class_sizes.append(each_class_size * max_step)
        else:
            class_sizes.append(each_class_size)
    logger(f"\n  Given total number of data is: {num_data}")
    logger("  To achieve that, each class should be lager than (Neg, pos1, pos2, ...):")
    for i in range(max_step + 1):
        logger(f"  {class_sizes[i]}", end="\t")
    for i in range(max_step + 1):
        if len(smis_w_label[i]) < class_sizes[i]:
            logger("  Fail. You can choose one of the following options:")
            logger("   1) Decrease the number of training data (args.num_data)")
            logger("   2) Generate more retro_analysis data")
            raise Exception("Error in data preproessing.")
    else:
        logger("  Fine.")
    logger(f"  Data were randomly chosen using random seed: {DATA_SPLIT_SEED}")

    labeled_data = []
    for idx, each_class in enumerate(smis_w_label):
        labeled_data += each_class[: class_sizes[idx]]

    return labeled_data, class_sizes


def generate_keys(processed_data_dir, preprocess_dir, ratio, class_sizes):
    key_dir = os.path.join(preprocess_dir, "data_keys")
    os.mkdir(key_dir)
    train_key_dicts, val_key_dicts, test_key_dicts = [], [], []

    all_data_dicts = {}
    tmp = 0
    global PROBLEM

    for label, size in enumerate(class_sizes):
        if label == 0 and PROBLEM == "regression":
            label = len(class_sizes)
        names = [f"{label}_{tmp+idx}.pkl" for idx in range(size)]
        all_data_dicts[label] = names
        tmp += size
    ratio = np.array(ratio)
    ratio = ratio / np.sum(ratio)
    for k, v in all_data_dicts.items():
        train, val, test = (
            v[: int(len(v) * ratio[0])],
            v[int(len(v) * ratio[0]) : int(len(v) * (ratio[0] + ratio[1]))],
            v[int(len(v) * (ratio[0] + ratio[1])) :],
        )
        train_key_dicts += train
        val_key_dicts += val
        test_key_dicts += test

    with open(f"{key_dir}/train_keys.pkl", "wb") as fw:
        pickle.dump(train_key_dicts, fw)
    with open(f"{key_dir}/val_keys.pkl", "wb") as fw:
        pickle.dump(val_key_dicts, fw)
    with open(f"{key_dir}/test_keys.pkl", "wb") as fw:
        pickle.dump(test_key_dicts, fw)

    # for splitted smiles data with label
    smi_with_label = {"train": {}, "val": {}, "test": {}}
    for name in train_key_dicts:
        label, _ = name.split("_")
        label = int(label)
        with open(os.path.join(processed_data_dir, name), "rb") as fr:
            try:
                smi_with_label["train"][label].append(pickle.load(fr)["smi"])
            except:
                smi_with_label["train"][label] = [pickle.load(fr)["smi"]]
    for name in val_key_dicts:
        label, _ = name.split("_")
        label = int(label)
        with open(os.path.join(processed_data_dir, name), "rb") as fr:
            smi = pickle.load(fr)["smi"]
        try:
            smi_with_label["val"][label].append(smi)
        except:
            smi_with_label["val"][label] = [smi]
    for name in test_key_dicts:
        label, _ = name.split("_")
        label = int(label)
        with open(os.path.join(processed_data_dir, name), "rb") as fr:
            smi = pickle.load(fr)["smi"]
        try:
            smi_with_label["test"][label].append(smi)
        except:
            smi_with_label["test"][label] = [smi]
    with open(f"{preprocess_dir}/smi_split_result.pkl", "wb") as fw:
        pickle.dump(smi_with_label, fw)

    return True


# 2. Graph feature generating functions
def do_get_graph_feature(tasks, save_dir, batch_size):
    while True:
        try:
            args = tasks.get(timeout=1)
        except queue.Empty:
            break
        else:
            data_list, task_idx = args[0], args[1]
            get_graph_feature(data_list, save_dir, batch_size, task_idx)
    return


def get_graph_feature(data_list, save_dir, batch_size, task_idx, for_inference=False):
    for idx, line in enumerate(data_list):
        idx += batch_size * task_idx
        line = line.rstrip()
        if for_inference:
            smi = line
        else:
            smi, label = line.split("\t")
            label = int(float(label))
        mol = Mol(smi)
        num_atoms = mol.GetNumAtoms()

        # 1. Adjacency
        adj = torch.from_numpy(
            np.asarray(GetAdjacencyMatrix(mol), dtype=bool)
            + np.eye(num_atoms, dtype=bool)
        )

        # 2. Node Feature
        node_feature = []
        for atom in mol.GetAtoms():
            node_feature.append(get_node_feature(atom))
        sssr = Chem.GetSymmSSSR(mol)
        # ring_feature = np.zeros([num_atoms, 6], dtype=bool)   # For ablation study
        ring_feature = sssr_to_ring_feature(sssr, num_atoms)
        node_feature = np.concatenate(
            [np.stack(node_feature, axis=0), ring_feature], axis=1
        )
        # node_feature=np.stack(node_feature, axis=0)

        node_feature = torch.from_numpy(node_feature)

        if for_inference:
            with open(f"{save_dir}/{idx}.pkl", "wb") as fw:
                pickle.dump(
                    {
                        "smi": smi,
                        "N_atom": num_atoms,
                        "feature": node_feature,
                        "adj": adj,
                    },
                    fw,
                )
        else:
            with open(f"{save_dir}/{label}_{idx}.pkl", "wb") as fw:
                pickle.dump(
                    {
                        "smi": smi,
                        "N_atom": num_atoms,
                        "feature": node_feature,
                        "adj": adj,
                        "label": label,
                    },
                    fw,
                )
    return True


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def sssr_to_ring_feature(sssr, num_atoms):
    ring_feature = np.zeros([num_atoms, 6], dtype=bool)
    for ring in sssr:
        r_size = min(len(ring) - 3, 5)
        for idx in list(ring):
            ring_feature[idx][r_size] = 1
    return ring_feature


HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.UNSPECIFIED,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
]

CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]


def get_node_feature(atom):
    return np.array(
        one_of_k_encoding(
            str(atom.GetSymbol()),
            ["C", "N", "O", "F", "S", "Cl", "Br", "I", "B", "P", "ELSE"],
        )
        + one_of_k_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5, 6, "ELSE"])
        + one_of_k_encoding(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, "ELSE"])
        + one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "ELSE"])
        + one_of_k_encoding(atom.GetHybridization(), HYBRIDIZATION)
        + one_of_k_encoding(atom.GetChiralTag(), CHIRALITY)
        + [atom.GetIsAromatic()],
        dtype=bool,
    )
    # 11+8+6+6+7+4+1 = 43


# 3. Main functions
def train_data_preprocess(args):
    # 1. Reading data
    since = time.time()
    preprocess_dir = args.preprocess_dir
    log = args.preprocess_logger
    global PROBLEM

    ratio = [8, 2, 0]  # train : val : test
    log()
    labeled_data, class_sizes = processing_data(
        args.data_dir, args.max_step, log, args.num_data
    )  # lists of 'smi\tlabel\n'
    log("  Data preprocessing continued.")

    # 2. Get Feature Tensors
    save_dir = os.path.join(preprocess_dir, "generated_data")
    os.mkdir(save_dir)
    log("  Generating features...", end="")
    batch_size = 10000
    tasks = Queue()
    procs = []
    num_batch = len(labeled_data) // batch_size
    if len(labeled_data) % batch_size != 0:
        num_batch += 1
    since = time.time()
    # Creating Tasks
    for batch_idx in range(num_batch):
        task = (
            labeled_data[batch_size * batch_idx : batch_size * (batch_idx + 1)],
            batch_idx,
        )
        tasks.put(task)

    # Creating Procs
    for p_idx in range(args.num_cores):
        p = Process(target=do_get_graph_feature, args=(tasks, save_dir, batch_size))
        procs.append(p)
        p.start()
        time.sleep(0.1)

    # Completing Procs
    for p in procs:
        p.join()
    log("  Done.")

    # 3. Split into train/val/test and generate keys
    log("  Generating keys...", end="")
    generate_keys(save_dir, preprocess_dir, ratio, class_sizes)
    log("  Done.")
    log(f"  Elapsed time: {time.time()-since}\n")
    return preprocess_dir


if __name__ == "__main__":
    mol = Chem.MolFromSmiles("Cc1ccccc1-C2CCC2")
    for atom in mol.GetAtoms():
        print(get_node_feature(atom))
        print(len(get_node_feature(atom)))
        break

    sssr = Chem.GetSymmSSSR(mol)
    print(sssr_to_ring_feature(sssr, 11))
