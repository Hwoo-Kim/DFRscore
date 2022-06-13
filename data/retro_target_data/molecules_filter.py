import os, sys
from multiprocessing import Pool
import queue
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import random, time

class RetroFilter:
    def __init__(self, target_file, num_cores, num_molecules):
        self.target_file = target_file
        self.save_name = target_file.split('/')[-1]
        self.num_cores = num_cores
        self.num_molecules = num_molecules

    @staticmethod
    def MolWt(mol):
        return ExactMolWt(mol) < 600
    
    @staticmethod
    def NoStar(s):
        return s.count('*') == 0
    
    @staticmethod
    def OneChemical(s):
        return s.count('.') == 0
    
    @staticmethod
    def Sanitize(mol):
        return int(Chem.SanitizeMol(mol, catchErrors=True))==0
    
    @staticmethod
    def OrganicSubset(mol):
        organic_subset = ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br', 'I', 'B', 'P']
        atoms = mol.GetAtoms()
        for a in atoms:
            if not a.GetSymbol() in organic_subset:
                return False
        return True

    def FilterMolecule(self, smi):
        if not all([self.OneChemical(smi), self.NoStar(smi)]):
            return None
        try: mol= Mol(smi)
        except: return None
        if mol == None: return None
        if not all([self.Sanitize(mol), self.OrganicSubset(mol), self.MolWt(mol)]):
            return None
        return Smiles(mol)

    def main(self):
        target_smis = []
        with open(self.target_file, 'r') as fr:
            if self.num_molecules == 'All':
                target_smis = fr.read().splitlines()
            else:
                for _ in range(self.num_molecules):
                    target_smis.append(fr.readline().rstrip())

        filtered_smis = []
        with Pool(self.num_cores) as p:
            result = p.map(self.FilterMolecule, target_smis)
        for smi in result:
            if smi:
                filtered_smis.append(smi+'\n')

        print(f' after filtering: {len(filtered_smis)}')
        print('-'*20)
        with open(f'filtered_{self.save_name}', 'w') as fw:
            fw.writelines(filtered_smis)

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
                if not OneChemical(s): continue
                if not NoStar(s): continue
                try: mol = Mol(s)
                except: continue
                if mol == None: continue
                if not Sanitize(mol): continue
                if not OrganicSubset(mol): continue
                if not MolWt(mol): continue
                ms.append(Smiles(mol)+'\n')
            
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
    tasks_to_accomplish = Queue()
    processes = []
    random.shuffle(target_smiles)
    target_smiles = target_smiles[:2000000]
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
    batch_size = 10000
    num_tasks= target_len//batch_size
    if num_tasks*batch_size != target_len:
        num_tasks += 1
    for task_idx in range(num_tasks):
        args = (task_idx, target_smiles[int(batch_size*task_idx):int(batch_size*(task_idx+1))], dir_name)
        tasks_to_accomplish.put(args)

    # creating processes
    print("Number of tasks (batch_size=10000): %d" %(num_tasks))
    print("I'm creating %d processors..." %(numb_cores))
    for w in range(numb_cores):
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
    #target_file = sys.argv[1]
    #try:
    #    numb_cores = int(sys.argv[2])
    #except:
    #    numb_cores = 1
    #with open(target_file, 'r') as fr:
    #    smis = fr.read().splitlines()
    #main(smis, numb_cores)
    try: 
        target_file, num_cores, num_molecules = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    except:
        target_file, num_cores, num_molecules = sys.argv[1], int(sys.argv[2]), 'All'

    print(f'target_file: {target_file}')
    print(f'num_cores: {num_cores}')
    print(f'num_molecules: {num_molecules}')
    if input(str('  Proceed? [Y,n] ')) == 'Y':
        save_name = target_file.split('/')[-1]
        print(f'\nsave file name: filtered_{save_name}')
        print(f'In process with multiprocessing (procs = {num_cores})...')
        retro_filter = RetroFilter(target_file, num_cores, num_molecules)
        retro_filter.main()
