import os,sys,pickle
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

from scripts.modelScripts.model import SVS
from scripts.modelScripts.experiments import runExp03
from scripts.utils import logger


# 1. Experiments setting
save_dir = str(sys.argv[1])       # DONOT USE '/' in front and back!
model_path = '/home/hwkim/SVS/save/PubChem_eMolecules/trained_model/GAT_best_model_162.pt'
test_file_path = f'/home/hwkim/SVS/save/{save_dir}/retro_result'

max_step=4

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
result_log_path= os.path.join(os.path.abspath(dirname(__file__)), os.path.join(save_dir, 'model_eval_result.log'))
log = logger( result_log_path)

log('----- Input Config Information -----')
log(f'  save_dir: {save_dir}')
log(f'  model_path: {model_path}')
log(f'  test_file_path: {test_file_path}')
log(f'  max_step: {max_step}\n')

# 2. Model load
log('----- Model evaluation started -----')
predictor = SVS()
predictor.restore(model_path)
predictor = predictor.cuda()

# 3. Get Evaluation Metrics
result = runExp03(predictor, 
            max_step=max_step,
            save_dir=save_dir,
            test_file_path=os.path.normpath(test_file_path),
            logger=log
            )

log('\nExp03 finished.\n')
