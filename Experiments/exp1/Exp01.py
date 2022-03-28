import os,sys,pickle
from os.path import dirname
sys.path.append(f'{dirname(dirname(os.path.abspath(dirname(__file__))))}')

from scripts.modelScripts.model import SVS
from scripts.modelScripts.experiments import runExp01
from scripts.utils import logger


# 1. Experiments setting
save_dir = os.path.abspath(dirname(__file__))
result_log_name = os.path.join(save_dir, 'model_eval_result.txt')
log = logger(os.path.join(save_dir, result_log_name))
max_step=4
test_file_path = '/home/hwkim/SVS/save/PubChem4M/retro_result/totdegree_seed1024/smi_split_result.pkl'
each_class_sizes = [2000, 2000, 2000, 2000, 2000]
#each_class_sizes = [20, 20, 20, 20, 20]

# 2. model load
model_path = '/home/hwkim/SVS/save/PubChem4M/SVS_nConv6_lr0_0004_conv256_fc128_8HEADS/GAT_best_model_174.pt'
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
predictor = predictor.cuda()

# 3. Get Evaluation Metrics
log('----- Input Config Information -----')
log(f'  model_path: {model_path}\n  test_file_path: {test_file_path}\n  save_dir: {save_dir}')
log(f'  max_step: {max_step}')
result = runExp01(predictor, 
            max_step=max_step,
            save_dir=save_dir,
            test_file_path=test_file_path,
            logger=log,
            each_class_sizes=each_class_sizes
            )

exit()
# 4. Plot the results
import matplotlib.pyplot as plt
import numpy as np
import pickle


with open('benchmark_result.pkl', 'rb') as fr:
    bench_result = pickle.load(fr)

idx = 0
sim, prec, recall = [], [], []
for k, v in bench_result.items():
    sim.append(float(v['sim']))
    prec.append(float(v['prec']))
    recall.append(float(v['recall']))
    if float(v['prec']) < 0.3:
        print(f'{idx}th temp prec: {float(v["prec"])}')
    idx +=1

sim = np.array(sim)
prec = np.array(prec)
recall = np.array(recall)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
xdata = np.arange(len(sim))
axs[0].plot(xdata, sim)
axs[0].set_title('Similarity')

axs[1].plot(xdata, prec);
axs[1].set_title('Precision')

axs[2].plot(xdata, recall);
axs[2].set_title('Recall')
p
