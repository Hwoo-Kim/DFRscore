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
model_path = '/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_94.pt'
test_path = f'/home/hwkim/DFRscore/data/retro_target_data/'

num_to_test = 10
max_step = 4

result_log_path = 'result.log'
log = logger(result_log_path)

log('----- Input config information -----')
log(f'  model_path: {model_path}')
log(f'  num_cores: {num_cores}')
log(f'  max_step: {max_step}\n')

# 2. Model load
log('----- Time check started -----')
predictor = DFRscore()
predictor.restore(model_path)
if use_cuda: predictor = predictor.cuda()

# 3. Get Evaluation Metrics
for data_set in ['ZINC.smi', 'ChEMBL.smi', 'MOSES.smi']:
    data_path = os.path.join(test_path, data_set)
    elapsed_time = time_check(predictor, data_path, num_to_test)
    log(data_path)
    log(f'elapsed_time: {round(elapsed_time, 3)}\n')

