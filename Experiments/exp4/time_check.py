import os,sys,pickle,time
from rdkit.Chem import MolFromSmiles as Mol
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

from scripts.modelScripts.model import DFRscore
from scripts.utils import logger

def time_check(model, file_path, num_to_test):
    smis = []
    with open(file_path, 'r') as fr:
        for _ in range(num_to_test):
            smis.append(fr.readline().rstrip())

    since = time.time()
    scores = model.smiListToScores(smis)
    elapsed_time = time.time()-since
    return elapsed_time

# 1. Experiments setting
num_cores = int(sys.argv[1])
assert sys.argv[2] in ['True', 'False'], 'Neither True nor False!'
use_cuda = sys.argv[2] == 'True'
model_path = '/home/hwkim/DFRscore/save/PubChem/DFR_nConv5_dimFC128_dimConv256/Best_model_159.pt'
#model_path = '/home/hwkim/DFRscore/save/PubChem/DFR_nConv5_dimFC256_dimConv512/Best_model_118.pt'
test_path = f'/home/hwkim/DFRscore/data/retro_target_data/'

num_to_test = 10000
max_step = 4

result_log_path = 'result.log'
log = logger(result_log_path)

# 2. Model load
log('----- Input config information -----')
log(f'num to test: {num_to_test}')
predictor = DFRscore.from_trained_model(
        num_cores=num_cores, path_to_model=model_path,
        fc_dim=128, conv_dim=256)
if use_cuda: predictor = predictor.cuda()
log(predictor)

# 3. Get Evaluation Metrics
log('----- Time check started -----')
for data_set in ['ZINC.smi', 'ChEMBL.smi', 'MOSES.smi']:
    data_path = os.path.join(test_path, data_set)
    elapsed_time = time_check(predictor, data_path, num_to_test)
    log(data_path)
    log(f'elapsed_time: {round(elapsed_time, 3)}\n')

