import os
import sys
from typing import Union

from rdkit import Chem
from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import sascorer

from scripts.getScores import getSCScore
from scripts.modelScripts.model import DFRscore


def get_SAScore(mol: Union[str, Chem.rdchem.Mol]):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return sascorer.calculateScore(mol)


def get_SCScore(smi: str):
    return getSCScore([smi])[0]


def get_DFRScore(mol: Union[str, Chem.rdchem.Mol], dfr_scorer):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return dfr_scorer.molToScore(mol)


def printScores(smi: str):
    mol = Chem.MolFromSmiles(smi)
    sa_score = get_SAScore(mol)
    sc_score = get_SCScore(smi)
    dfr_score = get_DFRScore(mol)
    print(f"query molecule: {smi}")
    print("SA : ", sa_score)
    print("SC : ", sc_score[0])
    print("DFR: ", dfr_score, end="\n\n")


if __name__ == "__main__":
    dfr_scorer = DFRscore.from_trained_model(
        "./save/PubChem/DFRscore/Best_model_163.pt"
    )

    smi = "C1CNCCC1C(=O)O"
    printScores(smi)
    smi = "N#Cc1c(F)ccc(c1)S(=O)(=O)Cl"
    printScores(smi)
    smi = "n1[nH]ccc1C(=O)O"
    printScores(smi)
    smi = "N1CCC[C@H]1CN"
    printScores(smi)
