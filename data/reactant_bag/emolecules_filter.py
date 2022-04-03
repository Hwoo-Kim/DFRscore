import os, sys
import time
from multiprocessing import Process, Queue
import queue
from tqdm import tqdm
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def NoStar(s):
    return s.count('*') == 0

def OneMol(s):
    return s.count('.') == 0

def RemoveSideChemical(s):
    if s.count('.')>1: return 'more_than_two'
    s1, s2 = s.split('.')
    m1, m2 = Mol(s1), Mol(s2)
    num1, num2 = 0,0
    if not m1 is None:
        for a in m1.GetAtoms():
            if a.GetSymbol()=='C': num1+=1
    if not m2 is None:
        for a in m2.GetAtoms():
            if a.GetSymbol()=='C': num2+=1
    if num1>num2: return s1
    elif num1<num2: return s2
    else: return False

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
            if ps_numb ==0:
                iterator = tqdm(smiles, total = len(smiles))
            more_than_two = 0
            for s in iterator:
                if not NoStar(s): continue
                if not OneMol(s):
                    s = RemoveSideChemical(s)
                if not s:
                    continue
                if s == 'more_than_two':
                    more_than_two+=1
                    continue
                try: mol = Mol(s)
                except:
                    continue
                if mol is None: continue
                ms.append(Smiles(mol)+'\n')
            
            if ps_numb == 0:
                print(f'length of given smiles: {len(smiles)}')
                print(f'length of passed smiles: {len(ms)}')
            with open(dir_name + "filtered_%d.txt" %(ps_numb), 'w') as fw:
                fw.writelines(ms)
    return True

def joining_files(target_dir:str):
    files = os.listdir(target_dir)
    smis = []
    for file_name in files:
        with open(target_dir + file_name, 'r') as fr:
            smis += fr.readlines()
    with open('filtered.smi', 'w') as fw:
        fw.writelines(smis)

    import shutil
    shutil.rmtree(target_dir)

    return True

def main(target_smiles:list, numb_cores:int):
    number_of_processes = numb_cores
    tasks_to_accomplish = Queue()
    processes = []
    target_len = len(target_smiles)
    print(f'The number of target molecules:\n  {target_len}')

    # generating scratch folder
    i = 1
    while True:
        dir_name = f'tmp{i}/'
        try:
            os.mkdir(dir_name)
        except FileExistsError as e:
            i+=1
            continue
        else:
            break

    # creating tasks
    for task_idx in range(number_of_processes):
        args = (task_idx, target_smiles[int(target_len*task_idx/numb_cores):int(target_len*(task_idx+1)/numb_cores)], dir_name)
        tasks_to_accomplish.put(args)

    # creating processes
    print("I'm creating %d processors..." %(numb_cores))
    for w in range(number_of_processes):
        p = Process(target=do_job, args=(tasks_to_accomplish,))
        processes.append(p)
        p.start()
        time.sleep(0.1)
    #print("All the processes have started w/o any problem.")

    # completing process
    for p in processes:
        p.join()

    # unifying files
    joining_files(dir_name)

    return True

if __name__ == '__main__':
    target_file = sys.argv[1]
    try:
        numb_cores = int(sys.argv[2])
    except:
        numb_cores = 1
    with open(target_file, 'r') as fr:
        smis = fr.read().splitlines()
    main(smis, numb_cores)
