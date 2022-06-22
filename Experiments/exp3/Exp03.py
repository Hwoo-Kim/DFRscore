import os,sys,pickle
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from scripts.modelScripts.model import DFRscore
from scripts.modelScripts.experiments import runExp03
from scripts.utils import logger


# 1. Experiments setting
save_dir = str(sys.argv[1])       # DONOT USE '/' in front and back!
model_path = '/home/hwkim/DFRscore/save/PubChem/DFRscore/Best_model_94.pt'
test_file_path = f'/home/hwkim/DFRscore/save/{save_dir}/exp03'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
result_log_path= os.path.join(os.path.abspath(dirname(__file__)), os.path.join(save_dir, 'model_eval_result.log'))
log = logger(result_log_path)

log('----- Input Config Information -----')
log(f'  save_dir: {save_dir}')
log(f'  test_file_path: {test_file_path}')
predictor = DFRscore.from_trained_model(model_path, num_cores=4)
#predictor = predictor.cuda()
log(predictor)

# 2. Get Evaluation Metrics
log('----- Model evaluation started -----')
result = runExp03(predictor, 
            save_dir=save_dir,
            test_file_path=os.path.normpath(test_file_path),
            logger=log
            )

log('\nExp03 finished.\n')
