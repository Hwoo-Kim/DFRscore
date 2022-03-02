from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.AllChem import ReactionFromSmarts as Rxn
from multiprocessing import Lock, Process, Queue, current_process 
from datetime import datetime
import queue # imported for using queue.Empty exception
import pickle
import json
import sys, time
import os
import shutil
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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
        if not s in result and not s[-1] in result:
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
            print(Smiles(target_in_mol))
            continue
        for pair in rxn_results:
            #sanitize_check = [int(Chem.SanitizeMol(mol, catchErrors=True)) == 0 for mol in pair]
            #if False in sanitize_check:
            #    continue
            products_in_smi = [Smiles(mol) for mol in pair]
            if None in products_in_smi:
                continue
            to_add = [rxn_idx, sorted(products_in_smi)]
            if to_add in result:
                continue
            else:
                result.append(to_add)
    return duplicate_remove(result)

def first_batch_reaction(targets_in_smiles, targets_in_mol, rxn_objs):
    dict_of_results = dict()
    for idx, target_in_mol in enumerate(targets_in_mol):
        if target_in_mol == None:
            continue
        target_result = onestep_by_reactions(target_in_mol, rxn_objs)
        if target_result != []:
            dict_of_results[targets_in_smiles[idx]] = target_result
    return dict_of_results
        
def further_batch_reaction(reaction_result, rxn_objs):
    to_del = []
    for target, retro_lists in reaction_result.items():
        new_retro_results = []
        tmp_del = []
        for retro_list in retro_lists:
            for intermediate in retro_list:
                try:
                    intermediate_in_mol = Mol(intermediate)
                except:
                    tmp_del.append(retro_list)
                    break
                if intermediate_in_mol == None:
                    break
                results = onestep_by_reactions(intermediate_in_mol, rxn_objs)
                if results == []:
                    continue
                else:
                    idx = retro_list.index(intermediate)
                    others = retro_list[:idx] + retro_list[idx+1:]
                    for pair in results:
                        new_retro_results.append(pair+others)
        for l in tmp_del:
            retro_lists.remove(l)
        if new_retro_results==[]:
            to_del.append(target)
            continue
        reaction_result[target] = new_retro_results
    for smi in to_del:
        del reaction_result[smi]
    return reaction_result

def initial_R_bag_check(targets_in_mol:list, reactant_bag:set):
    canonicalized_smiles = [Smiles(mol) for mol in targets_in_mol]
    in_R_bag_smi,in_R_bag_mol = [], []
    otherwise_smi, otherwise_mol = [], []
    for idx, smi in enumerate(canonicalized_smiles):
        if smi in reactant_bag:
            in_R_bag_smi.append(smi)
            in_R_bag_mol.append(targets_in_mol[idx])
        else:
            otherwise_smi.append(smi)
            otherwise_mol.append(targets_in_mol[idx])
    return in_R_bag_smi, in_R_bag_mol, otherwise_smi, otherwise_mol

def R_bag_check(batch_reaction_result:list, reactant_bag:set, current_depth:int, target_depth:int):
    '''
    This has two functions.
      1) Termination condition. Check if all the precursors are in reactant_bag or not.
      2) Filtering purpose. Check the retro_result exceeded the limit or not.
    Returns:
      positives: 
      batch_reaction_result: 
    '''
    positives = []
    diff = target_depth-current_depth
    if diff >0:
        for target, precursor_pairs in batch_reaction_result.items():
            to_del = []
            for pair_idx, pair in enumerate(precursor_pairs):
                exit = False
                s = time.time()
                temp_idx, to_be_checked, others = pair[0], pair[1], pair[2:]
                limit_numb = diff - len(others)
                precursor_R_bag_check = []

                for smi in to_be_checked:
                    check = smi in reactant_bag
                    precursor_R_bag_check.append(check)
                    if precursor_R_bag_check.count(False) > limit_numb:
                        to_del.append(pair)
                        exit = True
                        break
                    else:
                        continue
                if exit:
                    continue
                if precursor_R_bag_check.count(True) == len(to_be_checked) and len(others) == 0:
                    positives.append(target)
                    break
                indices = list_duplicates_of(precursor_R_bag_check, False)
                precursor_pairs[pair_idx] = [to_be_checked[idx] for idx in indices]+others
                continue
            for pair in to_del:
                precursor_pairs.remove(pair)

        for smi in positives:
            del batch_reaction_result[smi]
        return positives, batch_reaction_result

    elif diff ==0:
        for target, precursor_pairs in batch_reaction_result.items():
            for pair_idx, pair in enumerate(precursor_pairs):
                temp_idx, to_be_checked, others = pair[0], pair[1], pair[2:]
                if len(others) != 0:
                    continue
                precursor_R_bag_check = []

                for smi in to_be_checked:
                    check = smi in reactant_bag
                    if not check:
                        break
                    else:
                        precursor_R_bag_check.append(check)
                        continue

                if precursor_R_bag_check.count(True) == len(to_be_checked):
                    positives.append(target)
                    break
                    
                else:
                    continue

        return positives, None

def retrosynthetic_analysis_single_batch(targets_in_smiles:list, reactant_bag:set, depth:int, reactions:list, task_idx:int, exclude_in_R_bag:bool):
    '''
    This code conducts retro-analysis recursively.
      e.g., depth=1 reult --> depth=2 result --> depth=3 result --> ...
    Args:
      targets: list of target mol objects.
      reactant_bag: set of reactants. Already augmented.
      depth: the depth determining how many steps to use in retro-analysis in maximum.
      reactions: list of reaction smarts. 'RETRO_smarts' given in str.
    Returns:
      True 
    '''
    s = time.time()
    rxn_objects = []
    for rxn in reactions:
        try:
            rxn_objects.append(Rxn(rxn))
        except:
            rxn_objects.append(None)
    targets_in_mol = [Mol(target) for target in targets_in_smiles]
    positives_dict= dict()

    # 0. In R_bag check
    in_R_bag_smi, in_R_bag_mol, otherwise_smi, otherwise_mol = initial_R_bag_check(targets_in_mol, reactant_bag)
    with open(f'tmp/in_reactant_bag_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in in_R_bag_smi]
        fw.writelines(to_write)
    if exclude_in_R_bag:
        targets_in_smiles = otherwise_smi
        targets_in_mol = otherwise_mol

    # 1. depth = 1 operation
    reaction_result = first_batch_reaction(targets_in_smiles, targets_in_mol, rxn_objects)
    positive_each_depth, curated_reaction_result= R_bag_check(reaction_result, reactant_bag, 1, depth)
    positives_dict[1] = positive_each_depth
    if depth == 1:
        negative_set = list(set(targets_in_smiles)-set(positives_dict[1]))
        with open(f'tmp/positive_set_depth_1_{task_idx}.smi', 'w') as fw:
            to_write = [smi + '\n' for smi in positives_dict[1]]
            fw.writelines(to_write)
        with open(f'tmp/negative_set_depth_1_{task_idx}.smi', 'w') as fw:
            to_write = [smi + '\n' for smi in negative_set]
            fw.writelines(to_write)
        return True

    # 2. depth more than 1 operation
    for current_depth in range(depth-1):
        current_depth += 2
        if current_depth != depth:
            reaction_result = further_batch_reaction(curated_reaction_result, rxn_objects)
            positive_each_depth, curated_reaction_result= R_bag_check(reaction_result, reactant_bag, current_depth, depth)
            positives_dict[current_depth] = positive_each_depth
            continue
        else:
            reaction_result = further_batch_reaction(curated_reaction_result, rxn_objects)
            positive_each_depth, _ = R_bag_check(reaction_result, reactant_bag, current_depth, depth)
            positives_dict[current_depth] = positive_each_depth
            break
    # result report
    negative_set = set(targets_in_smiles)
    for current_depth in range(depth):
        current_depth += 1
        negative_set = negative_set-set(positives_dict[current_depth])
        with open(f'tmp/positive_set_depth_{current_depth}_{task_idx}.smi', 'w') as fw:
            to_write = [smi + '\n' for smi in positives_dict[current_depth]]
            fw.writelines(to_write)
    with open(f'tmp/negative_set_depth_{depth}_{task_idx}.smi', 'w') as fw:
        to_write = [smi + '\n' for smi in negative_set]
        fw.writelines(to_write)
    return True

def do_retro_analysis(tasks, reactant_bag, exclude_in_R_bag, num_tasks, log):
    while True:
        try:
            args = tasks.get(timeout=1)
        except queue.Empty:
            break
        else:
            targets, depth, templates, task_idx = args
            if task_idx%(num_tasks//4)==0:
                log(f'  task started: [{task_idx}/{num_tasks}]')
            retrosynthetic_analysis_single_batch(targets, reactant_bag, depth, templates, task_idx, exclude_in_R_bag)
    return True


def retrosyntheticAnalyzer(root, args):
    '''
    Main function. This conducts multiprocessing of 'retrosynthetic_analysis_single_batch'.
    '''
    #1. reading data from the input config.
    log = args.logger
    log()
    log('2. Retrosynthetic Analysis Phase.')

    reactant_path = os.path.join(root,args.reactant)
    reactant_set_path = os.path.join(root, 'data/reactant_bag/R_set.pkl')
    template_path = os.path.join(root, args.template)
    retro_target_path = os.path.join(root, args.retro_target)
    os.chdir(args.save_dir)
    os.mkdir('tmp')
    log = args.logger

    now = datetime.now()
    since_inform = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    log(f'  Started at: {since_inform}')

    with open(template_path, 'rb')  as fr:
        templates = pickle.load(fr)
    if args.using_augmented_template:
        templates = templates['augmented']
    else:
        templates = templates['original']

    rxn_templates = []
    for short_name, temp in templates.items():
        if temp == None:
            rxn_templates.append(None)
            continue
        rxn_templates.append(temp['retro_smarts'])


    with open(retro_target_path, 'r') as fr:
        targets = [fr.readline().rstrip() for i in range(args.num_molecules)]
    batch_size = min(args.batch_size, args.num_molecules//args.num_cores)
    max_time = 60

    log('  ----- Config information -----',
        f'  Target data path: {retro_target_path}',
        f'  Template data path: {template_path}',
        f'  Reactant data path: {reactant_path}',
        f'  Depth: {args.depth}',
        f'  Number of target molecules: {args.num_molecules}',
        f'  Number of cores: {args.num_cores}',
        f'  Batch size: {batch_size}',
        f'  Exclude_in_R_bag: {args.exclude_in_R_bag}',
        f'  With path search: {args.path}')
    if args.path:
        log(f'  Max time: {max_time}')

    # 2. multiprocessing of do_retro_analysis
    log('  ----- Multiprocessing -----')
    with open(reactant_set_path, 'rb') as fr:
        reactant_bag = pickle.load(fr)
    num_of_tasks = len(targets)//batch_size
    if len(targets) % batch_size != 0:
        num_of_tasks +=1
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
        p = Process(target = do_retro_analysis, args = (tasks, reactant_bag, args.exclude_in_R_bag, num_of_tasks, log))
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
    file2 = f'neg{args.depth}.smi'

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
        current_depth +=1
        for task_idx in range(num_of_tasks):
            with open(f'tmp/positive_set_depth_{current_depth}_{task_idx}.smi', 'r') as fr:
                each_pos_result.append(fr.read())
        pos_result = ''.join(each_pos_result)
        numb_of_mols_in_each_pos.append(pos_result.count('\n'))
        with open(files1[current_depth-1], 'w') as fw:
            fw.write(pos_result)

    each_neg_result = []
    for task_idx in range(num_of_tasks):
        with open(f'tmp/negative_set_depth_{args.depth}_{task_idx}.smi', 'r') as fr:
            each_neg_result.append(fr.read())
    neg_result = ''.join(each_neg_result)
    numb_of_mols_in_neg = neg_result.count('\n')
    with open(file2, 'w') as fw:
        fw.write(neg_result)

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
    log(f'  Negative set depth_{args.depth}: {numb_of_mols_in_neg}',
            f'\n  finished_at: {finished_at}',
            '    time passed: [%dh:%dm:%ds]' %(time_passed//3600, (time_passed%3600)//60, time_passed%60))
    return True

