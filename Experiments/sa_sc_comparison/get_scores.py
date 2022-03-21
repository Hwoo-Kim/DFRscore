from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import math
import pickle
import os
import os.path as op
import numpy as np

# 0. Common functions
def rescale_score(score_list:list, m, M, inverse=None) -> list:
    scores = np.array(score_list, dtype='float')
    rescaled_score = (scores-m)/(M-m)
    if inverse:
        rescaled_score = rescaled_score*(-1)+1
    return rescaled_score


# 1. SA Score 
def readFragmentScores(name='fpscores'):
    import gzip
    # generate the full path filename:
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open('%s.pkl.gz' % name))
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
    _fscores = readFragmentScores('fpscores')

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)  # <- 2 is the *radius* of the circular fingerprint
    fps = fp.GetNonzeroElements()
    score1 = 0.
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
    macrocyclePenalty = 0.
    # ---------------------------------------
    # This differs from the paper, which defines:
    #  macrocyclePenalty = math.log10(nMacrocycles+1)
    # This form generates better results when 2 or more macrocycles are present
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * .5

    sascore = score1 + score2 + score3

    # need to transform "raw" value into scale between 1 and 10
    min = -4.0
    max = 2.5
    sascore = 11. - (sascore - min + 1) / (max - min) * 9.
    # smooth the 10-end
    if sascore > 8.:
        sascore = 8. + math.log(sascore + 1. - 9.)
    if sascore > 10.:
        sascore = 10.0
    elif sascore < 1.:
        sascore = 1.0

    return sascore

def getSAScore(smis:list):
    '''
    Original SA score scale is [1,10]. Larger the score is, harder synthesis of corresponding molecule is.
    Rescaled score scale is [0,1]. Larger the score is, easier synthesis of corresponding molecule is.
    '''
    m, M = 1.0, 10.0
    raw_scores = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        raw_scores.append(calculateSAScore(mol))
    rescaled_score = rescale_score(raw_scores, m=m, M=M, inverse=True)    # rescaled into [0,1]
    return rescaled_score


# 2. SC Score
def getSCScore(smis:list):
    '''
    Original SC score scale is [1,5]. Larger the score is, harder synthesis of corresponding molecule is.
    Rescaled score scale is [0,1]. Larger the score is, easier synthesis of corresponding molecule is.
    '''
    import sys 
    import_path = op.join(op.dirname(op.abspath(__file__)), 'scscore/scscore/')
    if not op.exists(import_path):
        cwd = os.getcwd()
        os.chdir(op.dirname(op.abspath(__file__)))
        os.system('git clone https://github.com/connorcoley/scscore.git')
        os.chdir(cwd)
    if not import_path in sys.path:
        sys.path.append(import_path)
    from standalone_model_numpy import SCScorer

    model = SCScorer()
    model_root = op.join(op.dirname(op.abspath(__file__)), 'scscore/')
    model.restore(op.join(model_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))

    m, M = 1.0, 5.0
    raw_scores = []
    for smi in smis:
        (smi, raw_score) = model.get_score_from_smi(smi)
        raw_scores.append(raw_score)
    rescaled_score = rescale_score(raw_scores, m=m, M=M, inverse=True)    # rescaled into [0,1]
    return rescaled_score

if __name__=='__main__':
    smis = ['CC', 'CCCC', 'CCC(=O)O', 'C1CCCCC1', 'c1ccccc1', 'c1ccccc1O']
    print('List of SMILES:\n ', smis)
    print('SA Score (rescaled in [0,1], larger -> easier):\n ', getSAScore(smis))
    print('SC Score (rescaled in [0,1], larger -> easier):\n ', getSCScore(smis))

