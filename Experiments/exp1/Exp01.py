import os,sys,pickle
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from scripts.modelScripts.model import DFRscore
from scripts.modelScripts.experiments import runExp01
from scripts.utils import logger
import sys


# 1. Experiments setting
save_dir = str(sys.argv[1])       # DONOT USE '/' in front and back!
model_path = '/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_94.pt'
test_file_path = f'/home/hwkim/DFRscore/save/{save_dir}/pubchem_removed'

num_cores = int(sys.argv[2])
each_class_sizes = [int(sys.argv[3])] * 5

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
result_log_path= os.path.join(os.path.abspath(dirname(__file__)), os.path.join(save_dir, 'model_eval_result.log'))
log = logger(result_log_path)

log('----- Input Config Information -----')
log(f'  save_dir: {save_dir}')
log(f'  test_file_path: {test_file_path}')
predictor = DFRscore.from_trained_model(model_path, num_cores=num_cores)
predictor = predictor.cuda()
log(predictor)

# 2. Get Evaluation Metrics
log('----- Model evaluation started -----')
result = runExp01(predictor, 
            save_dir=save_dir,
            test_file_path=os.path.normpath(test_file_path),
            logger=log,
            each_class_sizes=each_class_sizes,
            )

log('\nExp01 finished.\n')
