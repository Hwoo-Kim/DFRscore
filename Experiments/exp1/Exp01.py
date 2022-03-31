import os,sys,pickle
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

from scripts.modelScripts.model import SVS
from scripts.modelScripts.experiments import runExp01
from scripts.utils import logger


# 1. Experiments setting
save_dir = 'PubChem'       # DONOT USE '/' in front and back!
test_file_path = '/home/hwkim/SVS/save/PubChem4M/retro_result/DATA_Regression/smi_split_result.pkl'
#test_file_path = '/Users/hwkim/works/SVS/Experiments/exp1/smi_split_result.pkl'
regression_model_path = '/home/hwkim/SVS/save/PubChem4M/Regression_nConv6_nfc4_convd256_fcd128_drop0.2_totdeg/GAT_model_160.pt'
#classification_model_path = '/home/hwkim/SVS/save/PubChem4M/nConv6_nfc4_convd256_fcd128_drop0.2_totdeg/GAT_best_model_155.pt'
classification_model_path = None
each_class_sizes = [20, 20, 20, 20, 20]
#each_class_sizes = [2000, 2000, 2000, 2000, 2000]
num_cores = 4
max_step=4

result_log_path= os.path.join(os.path.abspath(dirname(__file__)), 'model_eval_result.txt')
log = logger(result_log_path)

log('----- Input Config Information -----')
log(f'  model_path: {regression_model_path}\n  test_file_path: {test_file_path}\n  save_dir: {save_dir}')
log(f'  max_step: {max_step}')

# 2-1. Regression
if regression_model_path:
    log(f'\n1. Regression model phase.')
    predictor = SVS(
                conv_dim=256,
                fc_dim=128,
                n_GAT_layer=6,
                n_fc_layer=4,
                num_heads=8,
                len_features=36,
                max_num_atoms=64,
                problem='regression'
                )
    predictor.restore(regression_model_path)
    #predictor = predictor.cuda()
    
    # Get Evaluation Metrics
    result = runExp01(predictor, 
                problem='regression',
                max_step=max_step,
                save_dir=save_dir,
                test_file_path=os.path.normpath(test_file_path),
                logger=log,
                each_class_sizes=each_class_sizes,
                num_cores=num_cores
                )

# 2-2. Classification
if classification_model_path:
    log(f'\n2. Classification model phase.')
    predictor = SVS(
                conv_dim=256,
                fc_dim=128,
                n_GAT_layer=6,
                n_fc_layer=4,
                num_heads=8,
                len_features=36,
                max_num_atoms=64,
                problem='classification',
                out_dim=max_step+1
                )
    predictor.restore(classification_model_path)
    #predictor = predictor.cuda()
    
    # Get Evaluation Metrics
    result = runExp01(predictor, 
                problem='classification',
                max_step=max_step,
                save_dir=save_dir,
                test_file_path=os.path.normpath(test_file_path),
                logger=log,
                each_class_sizes=each_class_sizes,
                num_cores=num_cores
                )


log('\nExp01 finished.')
log('-'*20)
