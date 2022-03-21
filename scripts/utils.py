from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.AllChem import ReactionFromSmarts as Rxn
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.FragmentMatcher import FragmentMatcher
import queue
import pickle
import sys, time, shutil
import subprocess
import random
import os
from functools import wraps
import errno
import signal
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

"""
def reactant_frags_generator(templates):
    '''
    Function: get only reactant parts
      Args:
        templates: target template list.

      Returns:
        reactant_fragments: extracted reactant fragment.
    '''
    reactant_fragments = []
    for temp in templates:
        idx = temp.index('>>')
        reactant_fragments.append(temp[:idx])
    return reactant_fragments

def reactant_frag_generator(template):
    '''
    Function: get only reactant part
      Args:
        template: target template.

      Returns:
        reactant_fragments: extracted reactant fragment.
    '''
    idx = template.index('>>')
    return template[:idx]

def target_enumerator(targets, reactant_frag, option:str):
    '''
    Function: check which targets the given template is applicable to using FragmentMatcher.
      Args:
        targets: target chemicals. list of SMILES.
        reactant_frag: fragment of only reactant part.
        start: the start index to be used to count molecule index.

      Returns:
        applicable_target_smiles: applicable target smiles about the given reactant_frag. (list)
    '''
    mols = [Chem.MolFromSmiles(smiles) for smiles in targets]
    if '.' in reactant_frag:
        p1=FragmentMatcher()
        p1.Init(reactant_frag[:reactant_frag.index('.')])
        p2=FragmentMatcher()
        p2.Init(reactant_frag[reactant_frag.index('.')+1:])
        Matchers = [p1,p2]
    else:
        p = FragmentMatcher()
        p.Init(reactant_frag)
        Matchers = [p]

    if option == 'classify':
        applicable_target_smiles= []
        for idx, mol in enumerate(mols):
            if mol == None: continue
            for matcher in Matchers:
                if matcher.HasMatch(mol):
                    applicable_target_smiles.append(targets[idx])
                    break
            continue
        return applicable_target_smiles

    elif option == 'extract':
        count_applicable_targets = 0
        for mol in mols:
            if mol == None: continue
            if p.HasMatch(mol):
                count_applicable_targets +=1
                continue
        return count_applicable_targets

    elif option == 'pos_neg_gen':
        pos_set, neg_set = [], []
        for idx, mol in enumerate(mols):
            pos = False
            if mol == None: continue
            for matcher in Matchers:
                if matcher.HasMatch(mol):
                    pos_set.append(targets[idx])
                    pos = True
                    break
                else: continue
            if not pos:
                neg_set.append(targets[idx])
            continue
        return pos_set, neg_set
"""

class logger():
    def __init__(self, log_file_path):
        self.log_file = log_file_path
        try:
            with open(self.log_file, 'a') as w:
                pass
        except:
            print(f"Invalid log path {log_file_path}")
            exit()

    def __call__(self, *log, save_log = True, end='\n'):
        if len(log)==0:
            log = ('',)
        log = [str(i) for i in log]
        log = '\n'.join(log)
        print(log, end=end)
        if save_log:
            self.save(log, end=end)

    def save(self, log, end):
        with open(self.log_file, 'a') as w :
            w.write(log+end)

    def log_arguments(self, args):
        d = vars(args)
        for v in d:
            if v=='logger' or v=='root' or v=='save_dir' or v=='save_name': continue
            self(f'  {v}: {d[v]}')

def retro_save_dir_setting(root, args):
    target_data_name = args.retro_target.split('/')[-1].split('.smi')[0]
    if not os.path.exists(os.path.join(root,args.save_name,target_data_name)):
        os.mkdir(os.path.join(root,args.save_name,target_data_name))

    save_dir = os.path.join(root,args.save_name,target_data_name,'retro_result')
    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(save_dir+str(i)):
            i += 1
        save_dir = save_dir+str(i)
    os.mkdir(save_dir)
    return save_dir

def train_save_dir_setting(args):
    data_dir = os.path.normpath(args.data_dir)
    retro_data_dir = '/'.join(data_dir.split('/')[:-1])
    #data_dir = args.data_dir
    save_dir = os.path.normpath(os.path.join(retro_data_dir,args.save_name))
    
    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(f'{save_dir}{i}'):
            i+=1
        save_dir = f'{save_dir}{i}'
    os.mkdir(save_dir)
    return data_dir, save_dir

class TimeoutError(Exception):
    pass

def timeout(seconds, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # cancel the alarm
            return result

        return wraps(func)(wrapper)

    return decorator

def get_cuda_visible_devices(num_gpus: int) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    max_num_gpus = 4
    idle_gpus = []

    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]

            import sys
            major, minor = sys.version_info[0], sys.version_info[1]
            if major == 3 and minor > 6:
                proc = subprocess.run(cmd, capture_output=True, text=True)  # after python 3.7
            if major == 3 and minor <= 6:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, universal_newlines=True)  # for python 3.6

            if "No devices were found" in proc.stdout:
                break

            if "No running" in proc.stdout:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus
