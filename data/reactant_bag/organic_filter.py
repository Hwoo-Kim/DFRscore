import sys, os
from rdkit import Chem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from tqdm import tqdm
from multiprocessing import Process, Queue, current_process
import queue
import time

def OrganicSubset(mol):
    organic_subset = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'B', 'P']
    atoms = mol.GetAtoms()
    for a in atoms:
        if not a.GetSymbol() in organic_subset:
            return False
    return True

def do_job(tasks):
    while True:
        try:
            task = tasks.get(timeout=0.1)
        except queue.Empty:
            break
        else:
            # main operation
            ps_numb, smiles, dir_name = task
            ms = []
            iterator = smiles
            if ps_numb ==0: iterator = tqdm(smiles, total = len(smiles))
            for s in iterator:
                try: mol = Mol(s)
                except: continue
                if not OrganicSubset(mol): continue
                #try: s = Smiles(mol)
                #except: continue
                ms.append(s +'\n')
            
            with open(dir_name + "filtered_%d.txt" %(ps_numb), 'w') as fw:
                fw.writelines(ms)
    return True

def unifying_files(target_dir:str):
    files = os.listdir(target_dir)
    smis = []
    with open('canonicalized.smi', 'w') as fw:
        for file_name in files:
            with open(target_dir + file_name, 'r') as fr:
                #smis += fr.read().splitlines()
                smis += fr.readlines()
        smis = list(set(smis))
        fw.writelines(smis)
    print(f'After filtering: {len(smis)}')
    return True

def main(numb_cores:int, target_smiles:list):
    number_of_processes = numb_cores
    tasks_to_accomplish = Queue()
    processes = []
    print(f'Before filtering:\n  {len(target_smiles)}')

    # generating scratch folder
    i = 1
    while True:
        dir_name = f'scratch{i}/'
        try:
            os.mkdir(dir_name)
        except FileExistsError as e:
            i+=1
            continue
        else:
            break

    # creating tasks
    for task_idx in range(number_of_processes):
        args = (task_idx, target_smiles[int(len(target_smiles)*task_idx/numb_cores):int(len(target_smiles)*(task_idx+1)/numb_cores)], dir_name)
        tasks_to_accomplish.put(args)

    # creating processes
    print("I'm creating %d processors..." %(numb_cores))
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish,))
        processes.append(p)
        p.start()
        time.sleep(0.5)
    print("All the processes have started w/o any problem.")

    # completing process
    for p in processes:
        p.join()

    # unifying files
    unifying_files(dir_name)

    return True

if __name__ == '__main__':
    target_file = sys.argv[1]
    with open(sys.argv[1], 'r') as fr:
        smis=fr.read().splitlines()
    numb_cores = 16
    main(numb_cores, smis)
