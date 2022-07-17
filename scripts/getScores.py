import math
import os
import os.path as op
import pickle

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


# 0. Common functions
def rescale_score(score_list: list, m, M, reverse=False) -> list:
    "Rescale the given score_list into [0,1]"
    scores = np.array(score_list, dtype="float")
    rescaled_score = (scores - m) / (M - m)
    if reverse:
        rescaled_score = rescaled_score * (-1) + 1
    return rescaled_score


# 1. SA Score
# calculation of synthetic accessibility score as described in:
#
# Estimation of Synthetic Accessibility Score of Drug-like Molecules based on Molecular Complexity and Fragment Contributions
# Peter Ertl and Ansgar Schuffenhauer
# Journal of Cheminformatics 1:8 (2009)
# http://www.jcheminf.com/content/1/1/8
#
# several small modifications to the original paper are included
# particularly slightly different formula for marocyclic penalty
# and taking into account also molecule symmetry (fingerprint density)
#
# for a set of 10k diverse molecules the agreement between the original method
# as implemented in PipelinePilot and this implementation is r2 = 0.97
#
# peter ertl & greg landrum, september 2013
#


def readFragmentScores(name="fpscores"):
    import gzip

    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open("%s.pkl.gz" % name))
    outDict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict
    return _fscores


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def calculateSAScore(m):
    _fscores = readFragmentScores("fpscores")

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(
        m, 2
    )  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11.0 - (sascore - min + 1) / (max - min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore


def getSAScore(smi: str):
    """
    SA score scale is [1,10]. Larger the score is, harder synthesis of corresponding molecule is.
    """
    mol = Chem.MolFromSmiles(smi)
    score = calculateSAScore(mol)
    return score


# 2. SC Score
# MIT License
#
# Copyright (c) 2017 Connor Coley
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


def getSCScore(smis: list):
    """
    SC score scale is [1,5]. Larger the score is, harder synthesis of corresponding molecule is.
    """
    import sys

    import_path = op.join(op.dirname(op.abspath(__file__)), "scscore/scscore/")
    if not op.exists(import_path):
        cwd = os.getcwd()
        os.chdir(op.dirname(op.abspath(__file__)))
        os.system("git clone https://github.com/connorcoley/scscore.git")
        os.chdir(cwd)
    if not import_path in sys.path:
        sys.path.append(import_path)
    from standalone_model_numpy import SCScorer

    model = SCScorer()
    model_root = op.join(op.dirname(op.abspath(__file__)), "scscore/")
    model.restore(
        op.join(
            model_root,
            "models",
            "full_reaxys_model_1024bool",
            "model.ckpt-10654.as_numpy.json.gz",
        )
    )

    raw_scores = []
    for smi in smis:
        (smi, raw_score) = model.get_score_from_smi(smi)
        raw_scores.append(raw_score)
    return raw_scores


if __name__ == "__main__":
    smis = ["CC", "CCCC", "CCC(=O)O", "C1CCCCC1", "c1ccccc1", "c1ccccc1O"]
    print("List of SMILES:\n ", smis)
    print("SA Score:\n ", getSAScore(smis))
    print("SC Score:\n ", getSCScore(smis))
