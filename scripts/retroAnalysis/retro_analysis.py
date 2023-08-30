import os
import pickle
import queue  # imported for using queue.Empty exception
import shutil
import time
from datetime import datetime
from multiprocessing import Process, Queue

import timeout_decorator
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.AllChem import ReactionFromSmarts as Rxn

RDLogger.DisableLog("rdApp.*")


def cure_rxn_result(pair):
    """
    Cure the failed results which have a problem only in stereochemistry.
    Especially for reduction reaction.
    Ex) C[C@@H]-C(CC)C >> C[C@@H]=C(CC)C (false) , CC=C(CC)C (correct)
    """
    for mol in pair:
        if int(Chem.SanitizeMol(mol, catchErrors=True)) == 0:
            continue
        # if int(Chem.SanitizeMol(mol, catchErrors=True)) != 2:   # SANITIZE_PROPERTIES
        #    return None
        atoms = mol.GetAtoms()
        explicitH_counts = dict()
        for idx, atom in enumerate(atoms):
            if atom.GetHybridization() != Chem.rdchem.HybridizationType(0):
                continue
            explicitH_counts[idx] = atom.GetNumExplicitHs()
            atom.SetNumExplicitHs(0)
        if int(Chem.SanitizeMol(mol, catchErrors=True)) != 0:
            return None
        for idx, atom in enumerate(atoms):
            if idx not in explicitH_counts:
                continue
            atom.SetNumExplicitHs(explicitH_counts[idx])
            if atom.GetHybridization() != Chem.rdchem.HybridizationType(0):
                atom.SetNumExplicitHs(0)

    return pair


def list_duplicates_of(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def duplicate_remove(list_of_list):
    """
    Inner lists are sorted ones.
    """
    if list_of_list == []:
        return []
    result = []
    for s in list_of_list:
        if s not in result:
            result.append(s)
    return result


def onestep_by_reactions(target_in_mol, rxn_objs):
    result = []
    for rxn_idx, rxn in enumerate(rxn_objs):
        if rxn is None:
            continue
        try:
            rxn_results = rxn.RunReactants([target_in_mol])
        except:
            # print(Smiles(target_in_mol))
            continue
        for pair in rxn_results:
            sanitize_check = [
                int(Chem.SanitizeMol(mol, catchErrors=True)) == 0 for mol in pair
            ]
            if False in sanitize_check:
                pair = cure_rxn_result(pair)
                if pair is None:
                    continue
            products_in_smi = [Smiles(mol) for mol in pair]
            if None in products_in_smi:
                continue
            to_add = [rxn_idx, sorted(products_in_smi)]
            if to_add in result:
                continue
            else:
                result.append(to_add)
    return duplicate_remove(result)


def first_reaction(target_in_mol, rxn_objs):
    if target_in_mol is None:
        return []
    rxn_result = onestep_by_reactions(target_in_mol, rxn_objs)
    return rxn_result


def further_reaction(retro_result, rxn_objs):
    new_retro_results = []
    for pair in retro_result:
        for idx, intermediate in enumerate(pair):
            try:
                intermediate_in_mol = Mol(intermediate)
            except:
                continue
            if intermediate_in_mol is None:
                continue
            results = onestep_by_reactions(intermediate_in_mol, rxn_objs)
            if results == []:
                continue
            else:
                others = pair[:idx] + pair[idx + 1 :]
                for pair in results:
                    new_retro_results.append(pair + others)
    return new_retro_results


def initial_R_bag_check(targets_in_mol: list, reactant_bag: set):
    mol_in_text = [Smiles(mol) for mol in targets_in_mol]
    in_R_bag_smi, in_R_bag_mol = [], []
    otherwise_smi, otherwise_mol = [], []
    for idx, text in enumerate(mol_in_text):
        if text in reactant_bag:
            in_R_bag_smi.append(Smiles(targets_in_mol[idx]))
            in_R_bag_mol.append(targets_in_mol[idx])
        else:
            otherwise_smi.append(Smiles(targets_in_mol[idx]))
            otherwise_mol.append(targets_in_mol[idx])
    return in_R_bag_smi, in_R_bag_mol, otherwise_smi, otherwise_mol


def R_bag_check(
    reaction_result: list, reactant_bag: set, current_depth: int, target_depth: int
):
    """
    This has two functions.
      1) Termination condition. Check if all the precursors are in reactant_bag or not.
      2) Filtering purpose. Check the retro_result exceeded the limit or not.
    Returns:
      positives:
      batch_reaction_result:
    """
    diff = target_depth - current_depth
    if diff > 0:
        to_del = []
        for pair_idx, pair in enumerate(reaction_result):
            exit = False
            _, to_be_checked, others = pair[0], pair[1], pair[2:]
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
            if (
                precursor_R_bag_check.count(True) == len(to_be_checked)
                and len(others) == 0
            ):
                return int(current_depth), None
            indices = list_duplicates_of(precursor_R_bag_check, False)
            reaction_result[pair_idx] = [to_be_checked[idx] for idx in indices] + others
            continue
        for pair in to_del:
            reaction_result.remove(pair)

        return None, reaction_result

    elif diff == 0:
        for pair_idx, pair in enumerate(reaction_result):
            _, to_be_checked, others = pair[0], pair[1], pair[2:]
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
                return int(current_depth), None
            else:
                continue

        return None, None


def retro_analysis_single_batch(
    targets_in_smiles: list,
    reactant_bag: set,
    depth: int,
    rxn_templates: list,
    task_idx: int,
    exclude_in_R_bag: bool,
    max_time: int,
):
    """
    This code conducts retro-analysis recursively.
      e.g., depth=1 reult --> depth=2 result --> depth=3 result --> ...
    Args:
      targets_in_smiles: list of target SMILES.
      reactant_bag: set of reactants in SMILES. inchikey version deprecated.
      depth: the depth determining how many steps to use in retro-analysis in maximum.
      rxn_templates: list of reaction SMARTS.
      task_idx: task idx in the multiprocessing.
      exclude_in_R_bag: whether excluding target molecule if the molecule is in reactant bag.
    Returns:
      True
    """

    @timeout_decorator.timeout(
        max_time, timeout_exception=TimeoutError, use_signals=False
    )
    def retro_analysis(target_in_smiles, reactant_bag, depth, rxn_objects):
        # 1. depth = 1 operation
        target_in_mol = Mol(target_in_smiles)
        reaction_result = first_reaction(target_in_mol, rxn_objects)
        if reaction_result == []:
            return "Neg"
        label, filtered_reaction_result = R_bag_check(
            reaction_result, reactant_bag, 1, depth
        )
        if label is not None:
            return label

        # 2. depth more than 1 operation
        for current_depth in range(2, depth + 1):
            reaction_result = further_reaction(filtered_reaction_result, rxn_objects)
            if reaction_result == []:
                return "Neg"
            label, filtered_reaction_result = R_bag_check(
                reaction_result, reactant_bag, current_depth, depth
            )
            if label is not None:
                return label
        return "Neg"

    rxn_objects = []
    for rxn in rxn_templates:
        try:
            rxn_objects.append(Rxn(rxn))
        except:
            rxn_objects.append(None)
    targets_in_mol = [Mol(target) for target in targets_in_smiles]
    batch_result_dict = {"Neg": [], "TimeOut": []}
    for i in range(1, depth + 1):
        batch_result_dict[i] = []

    # in R_bag check
    in_R_bag_smi, in_R_bag_mol, otherwise_smi, otherwise_mol = initial_R_bag_check(
        targets_in_mol, reactant_bag
    )
    with open(f"tmp/in_reactant_bag_{task_idx}.smi", "w") as fw:
        to_write = [smi + "\n" for smi in in_R_bag_smi]
        fw.writelines(to_write)
    if exclude_in_R_bag:
        targets_in_smiles = otherwise_smi
        targets_in_mol = otherwise_mol

    # retro_analysis
    for target in targets_in_smiles:
        try:
            label = retro_analysis(target, reactant_bag, depth, rxn_objects)
        except TimeoutError:
            batch_result_dict["TimeOut"].append(target)
            continue
        else:
            batch_result_dict[label].append(target)

    # result report
    for current_depth in range(1, depth + 1):
        with open(f"tmp/pos{current_depth}_{task_idx}.smi", "w") as fw:
            to_write = [smi + "\n" for smi in batch_result_dict[current_depth]]
            fw.writelines(to_write)
    with open(f"tmp/neg{depth}_{task_idx}.smi", "w") as fw:
        to_write = [smi + "\n" for smi in batch_result_dict["Neg"]]
        fw.writelines(to_write)
    with open(f"tmp/timed_out_{task_idx}.smi", "w") as fw:
        to_write = [smi + "\n" for smi in batch_result_dict["TimeOut"]]
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
            retro_analysis_single_batch(
                targets,
                reactant_bag,
                depth,
                rxn_templates,
                task_idx,
                exclude_in_R_bag,
                max_time,
            )
            if num_tasks <= 5:
                log(f"  task finished: [{task_idx}/{num_tasks}]")
            elif task_idx % (num_tasks // 5) == 0:
                log(f"  task finished: [{task_idx}/{num_tasks}]")
    return True


def retrosyntheticAnalyzer(args):
    """
    Implementation of FRA (Full-RetroSynthetic-Analysis)
    Main function. This conducts multiprocessing of 'retrosynthetic_analysis_single_batch'.
    """
    # 1. reading data from the input config.
    log = args.logger
    log()
    log("2. Retrosynthetic Analysis Phase.")

    reactant_path = args.reactant
    reactant_set_path = os.path.join(args.root, "data/reactant_bag/R_set.pkl")
    template_path = args.template
    retro_target_path = args.retro_target
    os.chdir(args.save_dir)
    os.mkdir("tmp")

    now = datetime.now()
    since_inform = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    log(f"  Started at: {since_inform}")

    with open(os.path.join(args.root, template_path), "rb") as fr:
        templates = pickle.load(fr)
    if args.using_augmented_template:
        templates = templates["augmented"]
    else:
        templates = templates["original"]

    rxn_templates = []
    for short_name, temp in templates.items():
        if temp is None:
            rxn_templates.append(None)
            continue
        rxn_templates.append(temp["retro_smarts"])

    targets = []
    with open(os.path.join(args.root, retro_target_path), "r") as fr:
        for _ in range(args.start_index):
            fr.readline()
        for _ in range(args.num_molecules):
            line = fr.readline().rstrip()
            if line:
                targets.append(line)
            else:
                break
    args.num_molecules = len(targets)
    batch_size = min(args.batch_size, args.num_molecules // args.num_cores)
    if batch_size == 0:
        batch_size = 1
    elif args.num_molecules % batch_size != 0:
        # num_of_tasks +=1
        batch_size += 1

    log(
        "  ----- Config information -----",
        f"  Target data path: {retro_target_path}",
        f"  Template data path: {template_path}",
        f"  Using augmented template: {args.using_augmented_template}",
        f"  Reactant data path: {reactant_path}",
        f"  Depth: {args.depth}",
        f"  Start index: {args.start_index}",
        f"  Number of target molecules: {args.num_molecules}",
        f"  Number of cores: {args.num_cores}",
        f"  Batch size: {batch_size}",
        f"  Exclude_in_R_bag: {args.exclude_in_R_bag}",
        f"  With path search: {args.path}",
        f"  Max time: {args.max_time}",
    )
    # f'  Max time: -')

    # 2. multiprocessing of do_retro_analysis
    log("  ----- Multiprocessing -----")
    with open(reactant_set_path, "rb") as fr:
        reactant_data = pickle.load(fr)
    # data_format = reactant_data['format']
    reactant_bag = reactant_data["data"]
    num_of_tasks = args.num_molecules // batch_size
    if args.num_molecules % batch_size != 0:
        num_of_tasks += 1
    num_of_procs = args.num_cores
    log(f"  Number of tasks: {num_of_tasks}")

    tasks = Queue()
    procs = []

    since = time.time()
    # creating tasks
    for task_idx in range(num_of_tasks):
        batch_targets = targets[batch_size * task_idx : batch_size * (task_idx + 1)]
        retro_args = (batch_targets, args.depth, rxn_templates, task_idx)
        tasks.put(retro_args)

    # creating processes
    for worker in range(num_of_procs):
        p = Process(
            target=do_retro_analysis,
            args=(
                tasks,
                reactant_bag,
                args.exclude_in_R_bag,
                num_of_tasks,
                args.max_time,
                log,
            ),
        )
        procs.append(p)
        p.start()
        time.sleep(0.1)

    # completing processes
    for p in procs:
        p.join()

    # 3. join the results
    log(" ---------------------")
    log("  Retro analysis step finished.", "  Joining the results...")

    file0 = "in_reactant_bag.smi"
    files1 = [f"pos{current_depth+1}.smi" for current_depth in range(args.depth)]
    file2 = f"neg{args.depth}.smi"
    file3 = "timed_out.smi"

    each_Rbag_result = []
    for task_idx in range(num_of_tasks):
        with open(f"tmp/in_reactant_bag_{task_idx}.smi", "r") as fr:
            each_Rbag_result.append(fr.read())
    Rbag_result = "".join(each_Rbag_result)
    numb_of_mols_in_Rbag = Rbag_result.count("\n")
    with open(file0, "w") as fw:
        fw.write(Rbag_result)

    numb_of_mols_in_each_pos = []
    for current_depth in range(args.depth):
        each_pos_result = []
        current_depth += 1
        for task_idx in range(num_of_tasks):
            with open(f"tmp/pos{current_depth}_{task_idx}.smi", "r") as fr:
                each_pos_result.append(fr.read())
        pos_result = "".join(each_pos_result)
        numb_of_mols_in_each_pos.append(pos_result.count("\n"))
        with open(files1[current_depth - 1], "w") as fw:
            fw.write(pos_result)

    each_neg_result = []
    for task_idx in range(num_of_tasks):
        with open(f"tmp/neg{args.depth}_{task_idx}.smi", "r") as fr:
            each_neg_result.append(fr.read())
    neg_result = "".join(each_neg_result)
    numb_of_mols_in_neg = neg_result.count("\n")
    with open(file2, "w") as fw:
        fw.write(neg_result)

    each_fail_result = []
    for task_idx in range(num_of_tasks):
        with open(f"tmp/timed_out_{task_idx}.smi", "r") as fr:
            each_fail_result.append(fr.read())
    fail_result = "".join(each_fail_result)
    numb_of_mols_in_fail = fail_result.count("\n")
    with open(file3, "w") as fw:
        fw.write(fail_result)

    shutil.rmtree("tmp")

    # save the result
    time_passed = int(time.time() - since)
    now = datetime.now()
    finished_at = now.strftime("%Y. %m. %d (%a) %H:%M:%S")

    log()
    log("  ----- Generation result -----", f"  In reactant bag: {numb_of_mols_in_Rbag}")
    for i in range(args.depth):
        log(f"  Positive set depth_{i+1}: {numb_of_mols_in_each_pos[i]}")
    log(f"  Negative set depth_{args.depth}: {numb_of_mols_in_neg}")
    log(f"  Timed out: {numb_of_mols_in_fail}")
    log(f"\n  finished_at: {finished_at}")
    log(
        "  time passed: [%dh:%dm:%ds]"
        % (time_passed // 3600, (time_passed % 3600) // 60, time_passed % 60)
    )
    return True
