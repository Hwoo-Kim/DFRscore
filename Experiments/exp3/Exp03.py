import os,sys,pickle
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

from scripts.modelScripts.model import SVS
from scripts.modelScripts.experiments import runExp03
from scripts.utils import logger


# 1. Experiments setting
save_dir = os.path.abspath(dirname(__file__))
result_log_name = os.path.join(save_dir, 'model_eval_result.txt')
log = logger(os.path.join(save_dir, result_log_name))
max_step=4
#each_class_sizes = [2000, 2000, 2000, 2000, 2000]
each_class_sizes = [20, 20, 20, 20, 20]

# model loading
model_path = '/home/hwkim/SVS/save/PubChem4M/nConv6_nfc4_convd256_fcd128_drop0.2_totdeg/GAT_best_model_155.pt'
#model_path = '/Users/hwkim/works/SVS/Experiments/exp1/GAT_best_model_149.pt'
predictor = SVS(
            conv_dim=256,
            fc_dim=128,
            n_GAT_layer=6,
            n_fc_layer=4,
            num_heads=8,
            len_features=36,
            max_num_atoms=64,
            num_class=max_step+1
            )
predictor.restore(model_path)
#predictor = predictor.cuda()

# Test data loading
#test_file_path = '/Users/hwkim/works/SVS/Experiments/exp1/smi_split_result.pkl'
test_file_path = '/home/hwkim/SVS/save/PubChem4M/retro_result/totdegree_seed1024/smi_split_result.pkl'


# 2. Get Evaluation Metrics
log('----- Input Config Information -----')
log(f'  model_path: {model_path}\n  test_file_path: {test_file_path}\n  save_dir: {save_dir}')
log(f'  max_step: {max_step}')
result = runExp03(predictor, 
            max_step=max_step,
            save_dir=save_dir,
            test_file_path=os.path.normpath(test_file_path),
            logger=log,
            each_class_sizes=each_class_sizes
            )

log('\nExp03 finished.')
log('-'*20)
