import matplotlib.pyplot as plt
import numpy as np
import sys, os
import pickle

# 1. Reading result file
target_data = sys.argv[1]
if target_data[-1] == '/':
    target_data = target_data[:-1]
path_to_experiments = '/'.join(os.path.realpath(__file__).split('/')[:-2])
test_file_path = f'{path_to_experiments}/exp1/{target_data}/scores.pkl'
with open(test_file_path, 'rb') as f:
    data_dict = pickle.load(f)

# 2. Plot setting
plot_list = ['1','2','3','4']
color_list = [np.array([50,50,50]),np.array([100,100,233]),np.array([200,128,50]),np.array([230,30,30])]
for i in range(len(color_list)):
    color_list[i] = color_list[i]/255

metrics = ['svs', 'sa', 'sc']
metric_name_dict = {'svs': 'ENSS', 'sa': 'SAscore', 'sc': 'SCScore'}
#plt.figure(figsize=[16,5])
plt.figure(figsize=[6.4,4.8])
for metric_idx, metric in enumerate(['svs', 'sa', 'sc']):
    data_plot = []
    data = data_dict[metric]
    ax=plt.subplot(1, 1, metric_idx+1)
    ax.set_xlabel('True label', fontsize=15)
    ax.set_ylabel('Pred label', fontsize=15)
    #ax.set_title(metric_name_dict[metric])
    ax.set_title(target_data, fontsize=25)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    for label in plot_list:
        data_plot.append(data[label])
    
    positions = list(range(1,len(plot_list)+1))
    figure_plot = plt.violinplot(data_plot,positions = positions,showextrema = True,points=50)
    for idx in range(len(color_list)):
        violin = figure_plot['bodies'][idx]
        violin.set_facecolor(color_list[idx])
        violin.set_edgecolor('black')
        violin.set_alpha(1)
        bar = figure_plot['cbars']
        bar.set_color([0,0,0])
    break

#plt.savefig(f'{target_data}.png', format='png')
plt.savefig(f'{target_data}_only_ENSS.png', format='png')
