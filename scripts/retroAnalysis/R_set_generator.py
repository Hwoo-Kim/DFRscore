from rdkit.Chem import inchi
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smi
from multiprocessing import Process, Manager
from datetime import datetime
import os
import pickle
import time

def smis_to_inchikey(smis, l):
    inchikeys = []
    for smi in smis:
        inchikeys.append(inchi.MolToInchiKey(Mol(smi)))
    l.append(inchikeys)
    return True

def canonicalize_smiles(smis, l):
    can_smis = []
    for smi in smis:
        can_smis.append(Smi(Mol(smi)))
    l.append(can_smis)
    return True

def R_set_generator(args, use_inchikey=False, canonicalize=False):
    """
    Molecules in args.reactant file are saved in SMILES format.
    1) if 'use_inchikey' is True, all the chemicals will be saved in InchiKey format.
      In this case, 'canonicalize' option has no meaning. (not used)
    2) if 'use_inchikey' is False, all the chemicals will be saved in SMILES format.
      In this case, if 'canonicalize' is True, canonicalization process will be carried out. (slow down)
    """
    log = args.logger
    log()
    log('1. Reactant set generation Phase.')
    now = datetime.now()
    since_inform = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    log(f'  Started at: {since_inform}')
    # check the file already exists or not.
    since = time.time()
    reactant_bag_path = os.path.join(args.root, 'data/reactant_bag/R_set.pkl')     # set
    if os.path.isfile(reactant_bag_path):
        log('  The file already exists.')
        log('  Reactant set generation finished.')
        return True

    # reading data from the input config
    with open(args.reactant, 'r') as fr:
        reactants = fr.read().splitlines()

    # generate reactant set.
    to_save = {'format': 'smiles'}
    since = time.time()
    if use_inchikey == True:
        to_save['format'] = 'inchikey'
        log(f'  Size of reactant bag is: {len(reactants)}')
        log(f'  Converting the input smiles into inchi_key by multiprocessing with {args.num_cores} cores...')
        procs = []
        with Manager() as manager:
            list_of_inchikey = manager.list()
            batch_size = len(reactants)//args.num_cores+1
            for i in range(args.num_cores):
                smis= manager.list(reactants[batch_size*i:batch_size*(i+1)])
                p=Process(target=smis_to_inchikey, args=(smis,list_of_inchikey))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

            reactants = []
            for smis in list_of_inchikey:
                reactants += smis

    elif canonicalize == True:
        log(f'  Size of reactant bag is: {len(reactants)}')
        log(f'  Generating canonicalized smiles by multiprocessing with {args.num_cores} cores...')
        procs = []
        with Manager() as manager:
            list_of_can_smis= manager.list()
            batch_size = len(reactants)//args.num_cores+1
            for i in range(args.num_cores):
                smis= manager.list(reactants[batch_size*i:batch_size*(i+1)])
                p=Process(target=canonicalize_smiles, args=(smis,list_of_can_smis))
                p.start()
                procs.append(p)
            for p in procs:
                p.join()

            reactants = []
            for smis in list_of_can_smis:
                reactants += smis

    R_set = set(reactants)
    to_save.update({'data':R_set})

    # save the result
    with open(reactant_bag_path, 'wb') as fw:
        pickle.dump(to_save, fw)
    log(f'  number of reactants: {len(R_set)}')
    log('  Reactant set generation finished.')
    return True
