from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.AllChem import ReactionFromSmarts as Rxn
from rdkit.Chem.FragmentMatcher import FragmentMatcher
from scripts.utils import reactant_frags_generator
from scripts.utils import working_dir_setting
from scripts.utils import timeout, TimeoutError
from multiprocessing import Lock, Process, Queue, current_process 
from datetime import datetime
import queue # imported for using queue.Empty exception
import pickle
import shutil
import json
import sys, time
import os
from copy import copy, deepcopy
from bisect import bisect_left
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


onestep_by_reactions_cnt = 0
class SynthesisTree:
    '''
    Tree structure to represent a synthesis tree to the given target smiles.
    '''
    def __init__(self, target_smi):
        self.target = target_smi
        self.tree = [[target_smi]]
        self.notFinished = [target_smi]
        self.lastRxnIdx = None
        self.lastPrecursors = None
        self.lastRxnLoc = None

    def getCopiedTree(self):
        #new_tree = SynthesisTree(self.target)
        #new_tree.tree = [copy(pair) for pair in self.tree]
        #new_tree.notFinished = copy(self.notFinished)
        #new_tree.lastRxnIdx = copy(self.lastRxnIdx)
        #new_tree.lastPrecursors = copy(self.lastPrecursors)
        #new_tree.lastRxnLoc = copy(self.lastRxnLoc)
        #return new_tree
        return deepcopy(self)
        #return copy.copy(self)

    def getNumNotFin(self):
        return len(self.notFinished)

    def getNumRxn(self):
        return len(self.tree) -1

    def getTarget(self):
        return self.target

    def getTree(self):
        return self.tree

    def getNotFin(self):
        return self.notFinished

    def getLastRxnInform(self):
        return self.lastRxnIdx, self.lastPrecursors

    def getLastRxnLoc(self):
        return self.lastRxnLoc

    def getNumOthers(self):
        return self.numOthers

    def setLastRxnInform(self, idx:int, result:list, rxn_position:int):
        self.lastRxnIdx = idx
        self.lastPrecursors = result
        self.lastRxnLoc = rxn_position

    def removeNotFinLoc(self, loc_removed:int):
        del self.notFinished[loc_removed]

    def removeNotFinElem(self, elem_removed):
        self.notFinished.remove(elem_removed)

    def insertList(self, loc, l):
        #self.tree.append(copy.deepcopy(self.tree[-1]))
        self.tree.append(copy(self.tree[-1]))
        last = self.tree[-1]
        if len(last) > 1:
            del last[-1]
        del last[loc]
        for idx, elem in enumerate(l):
            last.insert(loc+idx, elem)

    def insertToNotFinished(self, loc, l):
        del self.notFinished[loc]
        self.numOthers = len(self.notFinished)
        for idx, elem in enumerate(l):
            self.notFinished.insert(loc+idx, elem)

    def getExpandedTrees(self, rxn_position, rxn_results):
        '''
        rxn_position is an index for notFinished list!!
        '''
        expanded_trees =[]
        elem = self.notFinished[rxn_position]
        loc = self.tree[-1].index(elem)
        for rxn_idx, result in rxn_results:
            copied_tree = self.getCopiedTree()
            copied_tree.insertList(loc, result)
            copied_tree.tree[-1].append([loc, rxn_idx])
            copied_tree.insertToNotFinished(rxn_position, result)
            copied_tree.setLastRxnInform(rxn_idx, result, rxn_position)
            expanded_trees.append(copied_tree)
        return expanded_trees

    def updateRbagInform(self, reactant_bag:set, diff:int):
        '''
        This method updates the R_bag inform of the synthesis_tree, and returns boolean for the test to yield True or False.
        If the test is not determined in the current level, this methods returns None obj.
        If the result for the test is False, the tree might be not updated properly. (Bcs not necessary)
        '''
        exit = False
        precursor_Rbag_check = []
        num_others = self.getNumOthers()
        limit_numb = diff - num_others
        if limit_numb < 0:
            return False
        rxn_idx, last_precursors = self.getLastRxnInform()
        for prec in last_precursors:
            check = prec in reactant_bag
            precursor_Rbag_check.append(check)
            if precursor_Rbag_check.count(False) > limit_numb:
                exit = True
                break
            if check: 
                self.removeNotFinElem(prec)

        if exit:
            return False
        elif precursor_Rbag_check.count(True) == len(last_precursors) and num_others == 0:
            return True
        else:
            return None

def list_duplicates_of(seq,item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item,start_at+1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs

def duplicate_remove(list_of_list):
    if list_of_list == []:
        return []
    result = []
    for s in list_of_list:
        if not s[1] in result and not s[1][::-1] in result:
            result.append(s)
    return result
    
def onestep_by_reactions(target_in_mol, rxn_objs):
    global onestep_by_reactions_cnt
    onestep_by_reactions_cnt += 0
    result = []
    for rxn_idx, rxn in enumerate(rxn_objs):
        if rxn == None:
            continue
        try:
            rxn_results = rxn.RunReactants([target_in_mol])
        except:
            print(Smiles(target_in_mol))
            continue
        for pair in rxn_results:
            products_in_smi = [Smiles(mol) for mol in pair]
            if None in products_in_smi:
                continue
            to_add = [rxn_idx, products_in_smi]
            if to_add in result:
                continue
            else:
                result.append(to_add)
    return duplicate_remove(result)

def first_reaction(synthesis_tree, rxn_objs):
    target_in_mol = Mol(synthesis_tree.getTarget())
    if target_in_mol == None:
        return False
    reaction_result = onestep_by_reactions(target_in_mol, rxn_objs)
    if reaction_result == []:
        return False
    syn_trees = synthesis_tree.getExpandedTrees(0, reaction_result)
    return syn_trees
        
def further_reaction(syn_trees, rxn_objs):
    new_syn_trees, to_del = [], []
    for tree in syn_trees:
        success = False
        not_finished = tree.getNotFin()
        last_rxn_loc = tree.getLastRxnLoc()
        for prec_idx, prec in enumerate(not_finished):
            if prec_idx < last_rxn_loc:
                continue
            mol_prec = Mol(prec)
            if mol_prec == None:
                continue
            reaction_result = onestep_by_reactions(mol_prec, rxn_objs)
            if reaction_result == []:
                continue
            else:
                #copied_tree = tree.getCopiedTree()
                #new_syn_trees += copied_tree.getExpandedTrees(prec_idx, reaction_result)
                new_syn_trees += tree.getExpandedTrees(prec_idx, reaction_result)
                success = True
        if success == False:
            to_del.append(tree)

    for tree in to_del:
        to_del.remove(tree)
    return new_syn_trees

def initial_R_bag_check(target_in_smiles:list, reactant_bag:set):
    canonicalized_smi= Smiles(Mol(target_in_smiles))
    return canonicalized_smi in reactant_bag
    """
    if canonicalized_smi in reactant_bag:
        in_R_bag_smi.append(smi)
        in_R_bag_mol.append(targets_in_mol[idx])
    else:
        otherwise_smi.append(smi)
        otherwise_mol.append(targets_in_mol[idx])
    return in_R_bag_smi, in_R_bag_mol, otherwise_smi, otherwise_mol
    """


def R_bag_check(synthesis_trees:list, reactant_bag:set, diff:int):
    '''
    This has two functions.
      1) Termination condition. Check if all the precursors are in reactant_bag or not.
      2) Filtering purpose. Check the retro_result exceeded the limit or not.
    Returns:
      result: True for pos, False for neg, an None for not determined yet.
      synthesis_trees: The new R_bag information updated synthesis_trees. 
    '''
    to_del = []
    for tree in synthesis_trees:
        result = tree.updateRbagInform(reactant_bag, diff)
        if result == True:
            return True, tree
        elif result == False:
            to_del.append(tree)
            continue
        elif result == None:
            continue

    if diff==0:
        return False, None
    for tree_removed in to_del:
        synthesis_trees.remove(tree_removed)
    if len(synthesis_trees) == 0:
        return False, None
    return synthesis_trees


def batch_retro_analysis(targets:list, reactant_bag:set, depth:int, rxn_templates:list, task_idx:int, max_time:int, exclude_in_R_bag:bool):
    '''
    This code conducts retro-analysis for a molecule by molecule recursively.
      e.g., depth=1 reult --> depth=2 result --> depth=3 result --> ...
    Args:
      targets_in_smiles: target SMILES. given in str.
      reactant_bag: set of reactants. Already augmented.
      rxn_templates: list of reaction smarts. 'RETR_smarts' given in str.
      depth: the depth determining how many steps to use in retro-analysis in maximum.
      task_idx: task idx in the multiprocessing.
      max_time: timeout criteria for a single molecule retro_analysis.
    Returns:
      None
    Saves the result file.
    '''

    @timeout(max_time)
    def retro_analysis(target_in_smiles:str, reactant_bag:set, depth:int, rxn_templates:list):
        rxn_objects = []
        for rxn in rxn_templates:
            try:
                rxn_objects.append(Rxn(rxn))
            except:
                rxn_objects.append(None)
        synthesis_tree = SynthesisTree(target_in_smiles)
        positives_dict= dict()

        # 0. In R_bag check
        in_R_bag = initial_R_bag_check(target_in_smiles, reactant_bag)
        if exclude_in_R_bag and in_R_bag:
            return None,None,True

        # 1. depth = 1 operation
        syn_trees = first_reaction(synthesis_tree, rxn_objects)
        if syn_trees == False:
            return False, None, in_R_bag
        result = R_bag_check(syn_trees, reactant_bag, depth-1)
        if type(result[0]) == bool:
            return result[0], result[1], in_R_bag
    
        # 2. depth more than 1 operation
        for current_depth in range(depth-1):
            current_depth += 2
            syn_trees = further_reaction(syn_trees, rxn_objects)
            result = R_bag_check(syn_trees, reactant_bag, depth-current_depth)
            if type(result[0]) == bool and result[0] == True:
                return result[0], result[1], in_R_bag
        return result[0], result[1], in_R_bag

    ## Main operation
    Retroanalysis_only_smiles, Retroanalysis_with_tree, Neg, in_R_bag, Failed = [], [], [], [], []
    #pos_cnt_list, Neg_cnt_list= [], []
    for i in range(depth):
        Retroanalysis_only_smiles.append([])
        Retroanalysis_with_tree.append([]) 

    for smi in targets:
        try:
            test_result, tree, _in_R_bag = retro_analysis(smi, reactant_bag, depth, rxn_templates)
        except TimeoutError as e:
            Failed.append(smi)
            continue
        if _in_R_bag ==True:
            in_R_bag.append(smi)
            if exclude_in_R_bag:
                continue
        if test_result == False:
            Neg.append(smi)
        elif test_result == True:
            label = int(tree.getNumRxn())
            Retroanalysis_only_smiles[label-1].append(smi)
            Retroanalysis_with_tree[label-1].append(tree.tree)

    for current_depth in range(depth):
        current_depth += 1
        with open(f'positive_set_depth_{current_depth}_{task_idx}.smi', 'w') as fw:
            to_write = [smi + '\n' for smi in Retroanalysis_only_smiles[current_depth-1]]
            fw.writelines(to_write)
        with open(f'positive_set_depth_{current_depth}_{task_idx}_with_tree.json', 'w') as fw:
            json.dump(Retroanalysis_with_tree[current_depth-1], fw)
    with open(f'negative_set_depth_{depth}_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in Neg]
        fw.writelines(to_write)
    with open(f'negative_set_depth_{depth}_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in Neg]
        fw.writelines(to_write)
    with open(f'in_reactant_bag_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in in_R_bag]
        fw.writelines(to_write)
    return True

def do_retro_analysis(tasks, reactant_bag, exclude_in_R_bag):
    while True:
        try:
            args = tasks.get(timeout=1)
        except queue.Empty:
            break
        else:
            #targets, depth, uni_templates, bi_templates, task_idx = args
            targets, depth, rxn_templates, task_idx, max_time = args
            since=time.time()
            print(f'  task started: {task_idx}')
            batch_retro_analysis(targets, reactant_bag, depth, rxn_templates, task_idx, max_time, exclude_in_R_bag)
            print(f'    {task_idx}th task time: {(time.time()-since):.2f}')
    return True


def retrosyntheticAnalyzer(root, common_config, retrosynthetic_analysis_config):
    '''
    Main function. This conducts multiprocessing of 'retrosynthetic_analysis_single_batch'.
    '''
    print('2. Retrosynthetic Analysis Phase.')
    now = datetime.now()
    since_inform = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    print('  Retro analysis step started successfully.')
    print(f'  retro analysis started at: {since_inform}')
    #1. reading data from the input config.
    dir_name =  str(common_config['dir_name'])
    augmented_reactant_data = f'{dir_name}/reactant_augmentation/augmented_reactants.pkl'     # set
    reactant_data = common_config['reactant_data']
    templates_path = str(common_config['template_data'])
    with open(f'{root}/{templates_path}', 'rb')  as fr:
        templates = pickle.load(fr)
    rxn_templates = []
    for temp in templates:
        if temp == None:
            rxn_templates.append(None)
            continue
        rxn_templates.append(temp['retro_smarts'])

    depth = int(retrosynthetic_analysis_config['depth'])
    numb_molecules = int(retrosynthetic_analysis_config['numb_molecules'])
    start_index = int(retrosynthetic_analysis_config['start_index'])
    target_data_name = str(retrosynthetic_analysis_config["target_data_name"])
    with_path_search = True
    max_time = int(retrosynthetic_analysis_config['max_time'])
    with open(str(retrosynthetic_analysis_config['retro_analysis_target']), 'r') as fr:
        targets = []
        for i in range(start_index):
            fr.readline()
        for i in range(numb_molecules):
            targets.append(fr.readline().rstrip())
        #targets = fr.read().splitlines()[start_index:start_index+numb_molecules]
    batch_size = int(retrosynthetic_analysis_config['batch_size'])
    numb_cores = int(common_config['numb_cores'])
    dir_name =  str(common_config['dir_name'])
    exclude_in_R_bag = str(retrosynthetic_analysis_config['exclude_in_R_bag']) == 'True'

    print('  Config information:')
    print(f'    Target data path: {retrosynthetic_analysis_config["retro_analysis_target"]}')
    print(f'    Target data name: {target_data_name}')
    print(f'    Template data: {templates_path}')
    print(f'    Start index: {start_index}')
    print(f'    Reactant_data: {reactant_data}')
    print(f'    Augmented reactant bag path: {augmented_reactant_data}')
    print(f'    Depth: {depth}')
    print(f'    Number of target molecules: {numb_molecules}')
    print(f'    Number of cores: {numb_cores}')
    print(f'    Batch size: {batch_size}')
    print(f'    With path search: {with_path_search}')
    print(f'    Max time: {max_time}')
    print(f'    Exclude_in_R_bag: {exclude_in_R_bag}')

    # 2. make a new working directory and change directory. config file report.
    working_dir = working_dir_setting(dir_name, target_data_name)
    #working_dir = f'{os.getcwd()}/retrosynthetic_analysis/scratch1'
    #os.chdir(working_dir)
    print(f'  Current working directory is:\n    {working_dir}')

    with open(f'{working_dir}/Input_config_file.txt', 'w') as fw:
        fw.writelines([f'----- Config information -----\n', \
            f'  Retro analysis started at: {since_inform}\n', \
            f'  Target data path: {retrosynthetic_analysis_config["retro_analysis_target"]}\n', \
            f'  Target data name: {target_data_name}\n', \
            f'  Template data: {templates_path}\n', \
            f'  Start index: {start_index}\n',f'  Reactant_data: {reactant_data}\n',\
            f'  Augmented reactant bag path: {augmented_reactant_data}\n', f'  Depth: {depth}\n', \
            f'  Number of target molecules: {numb_molecules}\n', f'  Number of cores: {numb_cores}\n', f'  Batch size: {batch_size}\n', \
            f'  With path search: {with_path_search}\n', f'  Max time: {max_time}\n', f'  Exclude_in_R_bag: {exclude_in_R_bag}'])

    # 3. multiprocessing of do_retro_analysis
    with open(f'{root}/{augmented_reactant_data}', 'rb') as fr:
        reactant_bag = pickle.load(fr)
    numb_of_tasks = len(targets)//batch_size
    if len(targets) % batch_size != 0:
        numb_of_tasks +=1
    numb_of_procs = int(numb_cores)

    tasks = Queue()
    procs = []

    since = time.time()
    # creating tasks
    for task_idx in range(numb_of_tasks):
        batch_targets = targets[batch_size*task_idx:batch_size*(task_idx+1)]
        #args = (batch_targets, depth, uni_templates, bi_templates, task_idx)
        args = (batch_targets, depth, rxn_templates, task_idx, max_time)
        tasks.put(args)

    # creating processes
    for worker in range(numb_of_procs):
        p = Process(target = do_retro_analysis, args = (tasks, reactant_bag, exclude_in_R_bag))
        procs.append(p)
        p.start()
        time.sleep(0.5)

    # completing processes
    for p in procs:
        p.join()

    # 4. join the results
    print('-----'*4)
    print('  Retro analysis step finished.\n  Joining the results...')

    file0 = f'in_reactant_bag.smi'
    files1 = [f'positive_set_depth_{current_depth+1}.smi' for current_depth in range(depth)]
    files2 = [f'positive_set_depth_{current_depth+1}_with_tree.json' for current_depth in range(depth)]
    file3= f'negative_set_depth_{depth}.smi'

    each_Rbag_result = []
    for task_idx in range(numb_of_tasks):
        with open(f'in_reactant_bag_{task_idx}.smi', 'r') as fr:
            each_Rbag_result.append(fr.read())
    Rbag_result = ''.join(each_Rbag_result)
    numb_of_mols_in_Rbag = Rbag_result.count('\n')
    with open(file0, 'w') as fw:
        fw.write(Rbag_result)
    for task_idx in range(numb_of_tasks):
        os.remove(f'in_reactant_bag_{task_idx}.smi')

    numb_of_mols_in_each_pos = []
    for current_depth in range(depth):
        each_pos_result = []
        each_pos_tree = []
        current_depth +=1
        for task_idx in range(numb_of_tasks):
            with open(f'positive_set_depth_{current_depth}_{task_idx}.smi', 'r') as fr:
                each_pos_result.append(fr.read())
            with open(f'positive_set_depth_{current_depth}_{task_idx}_with_tree.json', 'r') as fr:
                each_pos_tree += json.load(fr)

        pos_result = ''.join(each_pos_result)
        numb_of_mols_in_each_pos.append(pos_result.count('\n'))
        with open(files1[current_depth-1], 'w') as fw:
            fw.write(pos_result)
        with open(files2[current_depth-1], 'w') as fw:
            json.dump(each_pos_tree, fw)
        for task_idx in range(numb_of_tasks):
            os.remove(f'positive_set_depth_{current_depth}_{task_idx}.smi')
            os.remove(f'positive_set_depth_{current_depth}_{task_idx}_with_tree.json')

    each_neg_result = []
    for task_idx in range(numb_of_tasks):
        with open(f'negative_set_depth_{depth}_{task_idx}.smi', 'r') as fr:
            each_neg_result.append(fr.read())
    neg_result = ''.join(each_neg_result)
    numb_of_mols_in_neg = neg_result.count('\n')
    with open(file3, 'w') as fw:
        fw.write(neg_result)
    for task_idx in range(numb_of_tasks):
        os.remove(f'negative_set_depth_{depth}_{task_idx}.smi')

    # save the result
    time_passed = int(time.time()-since)
    now = datetime.now()
    finished_at = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    
    result_report = [f'----- Config information -----\n',
                f'  Retro analysis started at: {since_inform}\n', \
                f'  Target data path: {retrosynthetic_analysis_config["retro_analysis_target"]}\n',\
                f'  Target data name: {target_data_name}\n', \
                f'  Template data: {templates_path}\n', \
                f'  Start index: {start_index}\n', f'  Reactant_data: {reactant_data}\n',\
                f'  Augmented reactant bag path: {augmented_reactant_data}\n', f'  Depth: {depth}\n', \
                f'  Number of target molecules: {numb_molecules}\n', f'  Number of cores: {numb_cores}\n', \
                f'  Batch size: {batch_size}\n', f'  With path search: {with_path_search}\n', \
                f'  Max time: {max_time}\n',  f'  Exclude_in_R_bag: {exclude_in_R_bag}\n\n', '----- Generation result -----\n']
    result_report += [f'  In reactang bag:: {numb_of_mols_in_Rbag}\n']
    result_report += [f'  Positive set depth_{i+1}: {numb_of_mols_in_each_pos[i]}\n' for i in range(depth)]
    result_report += [f'  Negative set depth_{depth}: {numb_of_mols_in_neg}\n',\
            f'\n  finished_at: {finished_at}', \
            '\n   time passed: [%dh:%dm:%ds]' %(time_passed//3600, (time_passed%3600)//60, time_passed%60)]
    with open('generation_result.txt', 'w') as fw:
        fw.writelines(result_report)
    print('-----'*4)
    print(f'  Retro analysis finished at:\n  {finished_at}')
    print('  time passed: [%dh:%dm:%ds]' %(time_passed//3600, (time_passed%3600)//60, time_passed%60))
