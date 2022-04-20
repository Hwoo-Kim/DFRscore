from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.AllChem import ReactionFromSmarts as Rxn
from rdkit.Chem.FragmentMatcher import FragmentMatcher
from multiprocessing import Process, Queue
from datetime import datetime
import timeout_decorator
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
        global RXN_NAMES
        expanded_trees =[]
        elem = self.notFinished[rxn_position]
        loc = self.tree[-1].index(elem)
        for rxn_idx, result in rxn_results:
            copied_tree = self.getCopiedTree()
            copied_tree.insertList(loc, result)
            copied_tree.tree[-1].append([loc, RXN_NAMES[rxn_idx]])
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

def cure_rxn_result(pair):
    """
    Cure the failed results which have a problem only in stereochemistry.
    Especially for reduction reaction.
    Ex) C[C@@H]-C(CC)C >> C[C@@H]=C(CC)C (false) , CC=C(CC)C (correct)
    """
    for mol in pair:
        if int(Chem.SanitizeMol(mol, catchErrors=True)) == 0:
            continue
        #if int(Chem.SanitizeMol(mol, catchErrors=True)) != 2:   # SANITIZE_PROPERTIES
        #    return None
        atoms = mol.GetAtoms()
        explicitH_counts = dict()
        for idx, atom in enumerate(atoms):
            if atom.GetHybridization()!=Chem.rdchem.HybridizationType(0):
                continue
            explicitH_counts[idx] = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(0)
        if int(Chem.SanitizeMol(mol, catchErrors=True)) != 0:
            return None
        for idx, atom in enumerate(atoms):
            if not idx in explicitH_counts: continue
            atom.SetNumExplicitHs(explicitH_counts[idx])
            if atom.GetHybridization()!=Chem.rdchem.HybridizationType(0):
                atom.SetNumExplicitHs(0)

    return pair

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
    '''
    Inner lists are sorted ones.
    '''
    if list_of_list == []:
        return []
    result = []
    for s in list_of_list:
        if not s in result:
            result.append(s)
    return result
    
def onestep_by_reactions(target_in_mol, rxn_objs):
    result = []
    for rxn_idx, rxn in enumerate(rxn_objs):
        if rxn == None:
            continue
        try:
            rxn_results = rxn.RunReactants([target_in_mol])
        except:
            #print(Smiles(target_in_mol))
            continue
        for pair in rxn_results:
            sanitize_check = [int(Chem.SanitizeMol(mol, catchErrors=True)) == 0 for mol in pair]
            if False in sanitize_check:
                pair = cure_rxn_result(pair)
                if pair is None: continue
            products_in_smi = [Smiles(mol) for mol in pair]
            if None in products_in_smi:
                continue
            to_add = [rxn_idx, sorted(products_in_smi)]
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
    #target_in_smiles = Smiles(Mol(target_in_smiles))
    return target_in_smiles in reactant_bag

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


def batch_retro_analysis(
        targets_in_smiles:list,
        reactant_bag:set,
        depth:int,
        rxn_templates:list,
        task_idx:int,
        exclude_in_R_bag:bool,
        max_time:int
        ):
    '''
    This code conducts retro-analysis for a molecule by molecule recursively.
      e.g., depth=1 reult --> depth=2 result --> depth=3 result --> ...
    Args:
      targets_in_smiles: list of target SMILES.
      reactant_bag: set of reactants in SMILES. inchikey version deprecated.
      depth: the depth determining how many steps to use in retro-analysis in maximum.
      rxn_templates: list of reaction SMARTS.
      task_idx: task idx in the multiprocessing.
      exclude_in_R_bag: whether excluding target molecule if the molecule is in reactant bag.
      max_time: timeout criteria for a single molecule retro_analysis.
    Returns:
      True
    Saves the result file.
    '''

    @timeout_decorator.timeout(max_time, timeout_exception=TimeoutError, use_signals=False)
    def retro_analysis(target_in_smiles:str, reactant_bag:set, depth:int, rxn_objects:list):
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
    rxn_objects = []
    for rxn in rxn_templates:
        try:
            rxn_objects.append(Rxn(rxn))
        except:
            rxn_objects.append(None)

    Retroanalysis_only_smiles, Retroanalysis_with_tree, Neg, in_R_bag, Failed = [], [], [], [], []
    #pos_cnt_list, Neg_cnt_list= [], []
    for i in range(depth):
        Retroanalysis_only_smiles.append([])
        Retroanalysis_with_tree.append([]) 

    for smi in targets_in_smiles:
        try:
            test_result, tree, _in_R_bag = retro_analysis(smi, reactant_bag, depth, rxn_objects)
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
        with open(f'tmp/pos{current_depth}_{task_idx}.smi', 'w') as fw:
            to_write = [smi + '\n' for smi in Retroanalysis_only_smiles[current_depth-1]]
            fw.writelines(to_write)
        with open(f'tmp/pos{current_depth}_{task_idx}_with_tree.json', 'w') as fw:
            json.dump(Retroanalysis_with_tree[current_depth-1], fw)
    with open(f'tmp/neg{depth}_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in Neg]
        fw.writelines(to_write)
    with open(f'tmp/timed_out_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in Failed]
        fw.writelines(to_write)
    with open(f'tmp/in_reactant_bag_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in in_R_bag]
        fw.writelines(to_write)
    return True

def do_retro_analysis(tasks, reactant_bag, exclude_in_R_bag, num_tasks, max_time, log):
    while True:
        try:
            args = tasks.get(timeout=1)
        except queue.Empty:
            break
        else:
            targets, depth, rxn_templates, task_idx = args
            batch_retro_analysis(targets, reactant_bag, depth, rxn_templates, task_idx, exclude_in_R_bag, max_time)
            if num_tasks <= 5:
                log(f'  task finished: [{task_idx}/{num_tasks}]')
            elif task_idx%(num_tasks//5)==0:
                log(f'  task finished: [{task_idx}/{num_tasks}]')
    return True


RXN_NAMES = None
def retrosyntheticAnalyzer(args):
    '''
    Main function. This conducts multiprocessing of 'retrosynthetic_analysis_single_batch'.
    '''
    #1. reading data from the input config.
    log = args.logger
    log()
    log('2. Retrosynthetic Analysis Phase.')

    reactant_path = args.reactant
    reactant_set_path = os.path.join(args.root, 'data/reactant_bag/R_set.pkl')
    template_path = args.template
    retro_target_path = args.retro_target
    os.chdir(args.save_dir)
    os.mkdir('tmp')

    now = datetime.now()
    since_inform = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    log(f'  Started at: {since_inform}')

    with open(os.path.join(args.root, template_path), 'rb')  as fr:
        templates = pickle.load(fr)
    if args.using_augmented_template:
        templates = templates['augmented']
    else:
        templates = templates['original']

    global RXN_NAMES
    rxn_short_names = []
    rxn_templates = []
    for short_name, temp in templates.items():
        if temp == None:
            rxn_short_names.append(None)
            rxn_templates.append(None)
            continue
        rxn_short_names.append(short_name)
        rxn_templates.append(temp['retro_smarts'])
    RXN_NAMES = rxn_short_names

    with open(os.path.join(args.root, retro_target_path), 'r') as fr:
        for i in range(args.start_index):
            fr.readline()
        targets = [fr.readline().rstrip() for i in range(args.num_molecules)]
    batch_size = min(args.batch_size, args.num_molecules//args.num_cores)
    if args.num_molecules % batch_size != 0:
        #num_of_tasks +=1
        batch_size +=1

    log('  ----- Config information -----',
        f'  Target data path: {retro_target_path}',
        f'  Template data path: {template_path}',
        f'  Using augmented template: {args.using_augmented_template}',
        f'  Reactant data path: {reactant_path}',
        f'  Depth: {args.depth}',
        f'  Start index: {args.start_index}',
        f'  Number of target molecules: {args.num_molecules}',
        f'  Number of cores: {args.num_cores}',
        f'  Batch size: {batch_size}',
        f'  Exclude_in_R_bag: {args.exclude_in_R_bag}',
        f'  With path search: {args.path}',
        f'  Max time: {args.max_time}')

    # 2. multiprocessing of do_retro_analysis
    log('  ----- Multiprocessing -----')
    with open(reactant_set_path, 'rb') as fr:
        reactant_bag = pickle.load(fr)
    #data_format = reactant_data['format']
    reactant_bag = reactant_bag['data']
    num_of_tasks = args.num_molecules // batch_size
    if args.num_molecules % batch_size != 0:
        num_of_tasks += 1
    num_of_procs = args.num_cores
    log(f'  Number of tasks: {num_of_tasks}')

    tasks = Queue()
    procs = []

    since = time.time()
    # creating tasks
    for task_idx in range(num_of_tasks):
        batch_targets = targets[batch_size*task_idx:batch_size*(task_idx+1)]
        retro_args = (batch_targets, args.depth, rxn_templates, task_idx)
        tasks.put(retro_args)

    # creating processes
    for worker in range(num_of_procs):
        p = Process(target = do_retro_analysis, args = (tasks, reactant_bag, args.exclude_in_R_bag, num_of_tasks, args.max_time, log))
        procs.append(p)
        p.start()
        time.sleep(0.1)

    # completing processes
    for p in procs:
        p.join()

    # 3. join the results
    log(' ---------------------')
    log('  Retro analysis step finished.', '  Joining the results...')

    file0 = f'in_reactant_bag.smi'
    files1 = [f'pos{current_depth+1}.smi' for current_depth in range(args.depth)]
    files2 = [f'pos{current_depth+1}_with_tree.json' for current_depth in range(args.depth)]
    file3 = f'neg{args.depth}.smi'
    file4 = 'timed_out.smi'

    each_Rbag_result = []
    for task_idx in range(num_of_tasks):
        with open(f'tmp/in_reactant_bag_{task_idx}.smi', 'r') as fr:
            each_Rbag_result.append(fr.read())
    Rbag_result = ''.join(each_Rbag_result)
    numb_of_mols_in_Rbag = Rbag_result.count('\n')
    with open(file0, 'w') as fw:
        fw.write(Rbag_result)

    numb_of_mols_in_each_pos = []
    for current_depth in range(args.depth):
        each_pos_result = []
        each_pos_tree = []
        current_depth +=1
        for task_idx in range(num_of_tasks):
            with open(f'tmp/pos{current_depth}_{task_idx}.smi', 'r') as fr:
                each_pos_result.append(fr.read())
            with open(f'tmp/pos{current_depth}_{task_idx}_with_tree.json', 'r') as fr:
                each_pos_tree += json.load(fr)

        pos_result = ''.join(each_pos_result)
        numb_of_mols_in_each_pos.append(pos_result.count('\n'))
        with open(files1[current_depth-1], 'w') as fw:
            fw.write(pos_result)
        with open(files2[current_depth-1], 'w') as fw:
            json.dump(each_pos_tree, fw)

    each_neg_result = []
    for task_idx in range(num_of_tasks):
        with open(f'tmp/neg{args.depth}_{task_idx}.smi', 'r') as fr:
            each_neg_result.append(fr.read())
    neg_result = ''.join(each_neg_result)
    numb_of_mols_in_neg = neg_result.count('\n')
    with open(file3, 'w') as fw:
        fw.write(neg_result)

    each_fail_result = []
    for task_idx in range(num_of_tasks):
        with open(f'tmp/timed_out_{task_idx}.smi', 'r') as fr:
            each_fail_result.append(fr.read())
    fail_result = ''.join(each_fail_result)
    numb_of_mols_in_fail = fail_result.count('\n')
    with open(file4, 'w') as fw:
        fw.write(fail_result)

    shutil.rmtree('tmp')

    # save the result
    time_passed = int(time.time()-since)
    now = datetime.now()
    finished_at = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    
    log()
    log('  ----- Generation result -----',
        f'  In reactang bag: {numb_of_mols_in_Rbag}')
    for i in range(args.depth):
        log(f'  Positive set depth_{i+1}: {numb_of_mols_in_each_pos[i]}')
    log(f'  Negative set depth_{args.depth}: {numb_of_mols_in_neg}')
    log(f'  Timed out: {numb_of_mols_in_fail}')
    log(f'\n  finished_at: {finished_at}')
    log('  time passed: [%dh:%dm:%ds]' %(time_passed//3600, (time_passed%3600)//60, time_passed%60))
    return True
