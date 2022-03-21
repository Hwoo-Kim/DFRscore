from rdkit import Chem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from rdkit.Chem.rdmolops import GetFormalCharge
from multiprocessing import Queue, Process,current_process
from scripts.utils import logger
from sklearn.model_selection import train_test_split
import numpy as np
import queue
import torch
import pickle
import os
import multiprocessing
import random
import time
from datetime import datetime

SPLIT_SEED = 1024
# 1. Data splitting functions
# 1-1. For training data
def processing_data(data_dir,max_step,logger,num_data):
    # splitting
    smis_w_label = []
    for i in range(max_step+1):
        label = float(i)
        if i == 0.0:
            with open(f'{data_dir}/neg{max_step}.smi', 'r') as fr:
               smis = fr.read().splitlines()
        else:
            with open(f'{data_dir}/pos{i}.smi', 'r') as fr:
                smis = fr.read().splitlines()
        random.shuffle(smis)
        each_class = []
        for smi in smis:
            each_class.append(f'{smi}\t{label}\n')
        smis_w_label.append(each_class)

    # handling data size
    logger(f'  Number of data for each class (Neg, pos1, pos2, ...):')
    for i in range(max_step+1):
        logger(f'  {len(smis_w_label[i])}\t', end='')
    logger(f'  Given total number of data is: {num_data}')
    each_class_size  = num_data//(max_step*2)
    class_sizes = []
    for i in range(max_step+1):
        if i ==0:
            class_sizes.append(each_class_size*max_step)
        else :
            class_sizes.append(each_class_size)
    logger(f'  To achieve that, each class should be lager than (Neg, pos1, pos2, ...):')
    for i in range(max_step+1):
        logger(f'  {class_sizes[i]}', end='\t')
    for i in range(max_step+1):
        if len(smis_w_label[i]) < class_sizes[i]:
            logger('  Fail. You can choose one of the following options:')
            logger('   1) Decrease the number of training data (args.num_data)')
            logger('   2) Generate more retro_analysis data')
            raise Exception('Error in data preproessing.')
    else:
        logger('  Fine. Data preprocessing continued.')

    labeled_data = []
    for idx, each_class in enumerate(smis_w_label):
        labeled_data += each_class[:class_sizes[idx]]

    return labeled_data

def generate_keys(processed_data_dir, preprocess_dir, ratio):
    key_dir = os.path.join(preprocess_dir, 'data_keys')
    os.mkdir(key_dir)
    train_key_dicts, val_key_dicts, test_key_dicts = [], [],[]
    data_names = os.listdir(processed_data_dir)

    all_data_dicts = {}
    for name in data_names:
        label = int(name.split('_')[0])
        try:
            all_data_dicts[label].append(name)
        except:
            all_data_dicts[label] = [name]
    ratio = np.array(ratio)
    ratio = ratio/np.sum(ratio)
    for k,v in all_data_dicts.items():
        train,val,test = v[:int(len(v)*ratio[0])], \
                v[int(len(v)*ratio[0]):int(len(v)*(ratio[0]+ratio[1]))], \
                v[int(len(v)*(ratio[0]+ratio[1])):]
        #train, val_test = train_test_split(v, train_size=float(ratio[0])) 
        #val, test= train_test_split(val_test, train_size=float(ratio[1]/np.sum(ratio[1:])))
        train_key_dicts += train
        val_key_dicts += val
        test_key_dicts += test

    with open(f'{key_dir}/train_keys.pkl','wb') as fw:
        pickle.dump(train_key_dicts, fw)
    with open(f'{key_dir}/val_keys.pkl','wb') as fw:
        pickle.dump(val_key_dicts, fw)
    with open(f'{key_dir}/test_keys.pkl','wb') as fw:
        pickle.dump(test_key_dicts, fw)

    # for splitted smiles data with label
    smi_with_label = {'train':{}, 'val':{}, 'test':{}}
    for name in train_key_dicts:
        label, smi = name.split('_')
        label = int(label)
        smi_with_label['train'][smi] = label
    for name in val_key_dicts:
        label, smi = name.split('_')
        label = int(label)
        smi_with_label['val'][smi] = label
        #val[smi] = label
    for name in test_key_dicts:
        label, smi = name.split('_')
        label = int(label)
        smi_with_label['test'][smi] = label
        #test[smi] = label
    with open(f'{preprocess_dir}/smi_with_label.pkl', 'wb') as fw:
        pickle.dump(smi_with_label, fw)

    return True

# 1-2. For evalution model
def label_data(data_dir,save_dir,each_class_size,max_step):
    results = []
    for i in range(max_step+1):
        #each_class = []
        label = float(i)
        #label = i
        if i == 0:
            with open(f'{data_dir}/neg{max_step}.smi', 'r') as fr:
               smis = fr.read().splitlines()
        else:
            with open(f'{data_dir}/pos{i}.smi', 'r') as fr:
                smis = fr.read().splitlines()
        information = [f'{smi}\t{label}\n' for smi in smis]
        each_class = information[:each_class_size]
        #for j in range(each_class_size):
        #    each_class.append(f'{smis[j]}\t{label}\n')
        #for smi in smis:
        #    each_class.append(f'{smi}\t{label}\n')
        results.append(each_class)
    print(f'Num data for each class (Neg, pos1, pos2, ...):')
    for i in range(max_step+1):
        print(len(results[i]), end='\t')
    to_write=[]
    for each_class in results:
        to_write+=each_class
    with open(f'{save_dir}/labeled_data.txt','w') as fw:
        fw.writelines(to_write)
    return results

# 1-3. For inferencing
def read_data(data_path):
    with open(data_path, 'r') as fr:
        smis = fr.read().splitlines()
    print(f'Num data for inference: {len(smis)}')
    return smis

# 2. Graph feature generating functions
def do_get_graph_feature(tasks,save_dir,max_num_atoms,len_features):
    while True:
        try:
            args = tasks.get(timeout=1)
        except queue.Empty:
            break
        else:
            data_list = args[0]
            get_graph_feature(data_list,
                    save_dir,
                    max_num_atoms,
                    len_features
                    )
    return

def get_graph_feature(data_list,
                save_dir,
                max_num_atoms,
                len_features,
                for_inference=False
                ):
    for idx, line in enumerate(data_list):
        line = line.rstrip()
        if for_inference:
            smi = line
        else:
            smi, label = line.split('\t')
            label = int(float(label))
        mol = Mol(smi)
        sssr = Chem.GetSymmSSSR(mol)
        num_atoms = mol.GetNumAtoms()
        adj = GetAdjacencyMatrix(mol) + np.eye(num_atoms)
        padded_adj = np.zeros((max_num_atoms,max_num_atoms))
        padded_adj[:num_atoms,:num_atoms] = adj
        feature = []
        atoms = mol.GetAtoms()
        for atom in atoms:
            feature.append(get_atoms_feature(sssr, atom))
        feature = np.array(feature)
        padded_feature = np.zeros((max_num_atoms, len_features))
        padded_feature[:num_atoms,:len_features] = feature
        padded_feature = torch.from_numpy(padded_feature)
        padded_adj = torch.from_numpy(padded_adj)
        if for_inference:
            with open(f'{save_dir}/{idx}.pkl','wb') as fw:
                pickle.dump({'smi':smi,
                        'feature':padded_feature,
                        'adj':padded_adj},
                        fw)
        else:
            with open(f'{save_dir}/{label}_{idx}.pkl','wb') as fw:
                pickle.dump({'smi':smi,
                        'feature':padded_feature,
                        'adj':padded_adj,
                        'label':label},
                        fw)
    return True


def one_of_k_encoding(x,allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_ring_inform(sssr, atom):
    ring_inform = [0]*6
    if not atom.IsInRing():
        return ring_inform
    atom_index = atom.GetIdx()
    for ring in sssr:
        if (atom_index in ring) and len(ring)<8:
            ring_inform[len(ring)-3]=1
        elif (atom_index in ring) and len(ring)>=8:
            ring_inform[5]=1

    return ring_inform

def get_atoms_feature(sssr, atom):
    return np.array(one_of_k_encoding(str(atom.GetSymbol()),['C','N','O','F','S','Cl','Br','I','B','P','ELSE'])+
                    one_of_k_encoding(int(atom.GetDegree()),[0,1,2,3,4,'ELSE'])+
                    one_of_k_encoding(int(atom.GetExplicitValence()),[0,1,2,3,4,'ELSE'])+
                    one_of_k_encoding(int(atom.GetTotalDegree()),[0,1,2,3,4,'ELSE'])+
                    #one_of_k_encoding(int(atom.GetFormalCharge()),[-2,-1,0,1,2,'ELSE'])+
                    get_ring_inform(sssr,atom)+
                    [atom.GetIsAromatic()])
                    # 11+6+6+6+6+1 = 36

# 3. Main functions
def train_data_preprocess(args):
    # 1. Reading data
    preprocess_dir = os.path.join(args.data_dir, args.data_preprocessing)
    since = time.time()
    now = datetime.now()
    since_inform = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    if os.path.exists(preprocess_dir):
        print('\n1. Data preprocessing phase')
        print('  Processed data already exists.')
        print('  Training data preprocessing finished.')
        return preprocess_dir
    else:
        os.mkdir(preprocess_dir)
        log = logger(os.path.join(preprocess_dir, 'preprocessing.log'))
        log('\n1. Data preprocessing phase')
        log(f'  Started at: {since_inform}')
        log(f'  Data will be generated in: {preprocess_dir}')

    ratio = [8,1,1]         # train : val : test
    log('  Spliting data into train set, val set, and test set...')
    labeled_data = processing_data(args.data_dir,args.max_step,log,args.num_data)        # lists of (smi,label)
    log('  Done.')

    # 2. Get Feature Tensors
    save_dir= os.path.join(preprocess_dir,'generated_data')
    os.mkdir(save_dir)
    log(f'  Generating features...', end='')
    batch_size = 10000
    tasks = Queue()
    procs = []
    num_batch = len(labeled_data)//batch_size
    if len(labeled_data)%batch_size !=0: num_batch+=1
    since = time.time()
    # Creating Tasks
    for batch_idx in range(num_batch):
        #indices = list(range(batch_size*batch_idx, batch_size*(batch_idx+1)))
        #task = (labeled_data[batch_size*batch_idx:batch_size*(batch_idx+1)], indices)
        task = (labeled_data[batch_size*batch_idx:batch_size*(batch_idx+1)],)
        tasks.put(task)

    # Creating Procs
    for worker in range(args.num_cores):
        p = Process(
                target=do_get_graph_feature,
                args=(tasks,save_dir,args.max_num_atoms,args.len_features)
                )
        procs.append(p)
        p.start()
        time.sleep(0.1)

    # Completing Procs
    for p in procs:
        p.join()
    log('  Done.')

    # 3. Split into train/val/test and generate keys
    log('  Generating keys...', end='')
    generate_keys(save_dir, preprocess_dir,ratio)
    log('  Done.')
    log(f'  Elapsed time: {time.time()-since}')
    return preprocess_dir

"""
def GAT_evaluation_data_generation(data_dir, save_dir, evaluation_args):
    # 1. Reading data
    each_class_size = evaluation_args.each_class_size
    max_num_atoms = evaluation_args.max_num_atoms
    len_features = evaluation_args.len_features
    max_step = evaluation_args.max_step
    working_dir = utils.working_dir_setting(save_dir,'inference_data')     # not move
    logger = utils.logger(f'{working_dir}/data_generation.log')
    datas=label_data(data_dir, working_dir,each_class_size, max_step, logger)        # lists of (smi,label)
    logger('Done.')

    # 2. Feature generation
    logger(f'making tensors ...',end='')
    for i in range(max_step+1):
        if i == 0:
            save_path = f'{working_dir}/infer_neg{max_step}'
        else:
            save_path = f'{working_dir}/infer_pos{i}'
        get_graph_feature(datas[i],save_path,max_num_atoms,len_features)
    logger('\tDone.')

    return working_dir

def GAT_inference_data_generation(data_path, save_dir, inference_args):
    # 1. Reading data
    max_num_atoms = inference_args.max_num_atoms
    len_features = inference_args.len_features
    max_step = inference_args.max_step
    working_dir = utils.working_dir_setting(save_dir,'inference_data')     # not move
    logger = utils.logger(f'{working_dir}/data_generation.log')
    data=read_data(data_path)
    logger('Done.')

    # 2. Feature generation
    logger(f'making tensors ...', end='')
    save_path = f'{working_dir}/for_inference'
    get_graph_feature(data,save_path,max_num_atoms,len_features, for_inference=True)
    logger('\tDone.')

    return working_dir
"""
