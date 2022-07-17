import os
import sys
from multiprocessing import Pool

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MolFromSmiles as Mol
from rdkit.Chem import MolToSmiles as Smiles
from rdkit.Chem.Descriptors import ExactMolWt

RDLogger.DisableLog("rdApp.*")


class RetroFilter:
    def __init__(self, target_file, num_cores, num_molecules):
        self.target_file = target_file
        self.save_name = target_file.split("/")[-1]
        self.num_cores = num_cores
        self.num_molecules = num_molecules

    @staticmethod
    def NoStar(s):
        return s.count("*") == 0

    @staticmethod
    def OneChemical(s):
        return s.count(".") == 0

    @staticmethod
    def GetRandomNumbers(maximum, num_to_sample):
        assert num_to_sample <= maximum, "num_to_sample must not be large than maximum."
        seed = 12345
        rand_generator = np.random.default_rng(seed)
        rand_ints = set()

        more = num_to_sample
        while True:
            rand_ints.update(rand_generator.integers(low=0, high=maximum, size=more))
            if len(rand_ints) == num_to_sample:
                break
            more = num_to_sample - len(rand_ints)
        return list(rand_ints)

    @classmethod
    def GetRandomSamples(cls, data_list, num_to_sample):
        maximum = len(data_list)
        rand_ints = cls.GetRandomNumbers(maximum, num_to_sample)
        return [data_list[idx] for idx in rand_ints]

    @classmethod
    def FilterMolecule(cls, smi):
        if not all([cls.OneChemical(smi), cls.NoStar(smi)]):
            return None
        try:
            mol = Mol(smi)
        except:
            return None
        if mol == None:
            return None
        return Smiles(mol)

    def main(self):
        with open(self.target_file, "r") as fr:
            target_smis = fr.read().splitlines()
            if not self.num_molecules == "All":
                print(f"  sampling {self.num_molecules} molecules...", end=" ")
                target_smis = self.GetRandomSamples(target_smis, self.num_molecules)
                print("done.")
            target_smis = list(set(target_smis))

        filtered_smis = []
        with Pool(self.num_cores) as p:
            result = p.map(self.FilterMolecule, target_smis)
        for smi in result:
            if smi:
                filtered_smis.append(smi + "\n")

        print(f" after filtering: {len(filtered_smis)}")
        print("-" * 20)
        with open(f"filtered_{self.save_name}", "w") as fw:
            fw.writelines(filtered_smis)

        return True


if __name__ == "__main__":
    try:
        target_file, num_cores, num_molecules = (
            sys.argv[1],
            int(sys.argv[2]),
            int(sys.argv[3]),
        )
    except:
        target_file, num_cores, num_molecules = sys.argv[1], int(sys.argv[2]), "All"

    print(f"target_file: {target_file}")
    print(f"num_cores: {num_cores}")
    print(f"num_molecules: {num_molecules}")

    while True:
        inp = input(str("  Proceed? [Y,n] "))
        if inp == "Y":
            save_name = target_file.split("/")[-1]
            print(f"\nsave file name: filtered_{save_name}")
            print(f"In process with multiprocessing (procs = {num_cores})...")
            retro_filter = RetroFilter(target_file, num_cores, num_molecules)
            retro_filter.main()
            break
        elif inp == "n":
            print("Exit.")
            break
        else:
            print("The input was neither [Y,n].")
            print("Please type again.")
            continue
