from get_scores import getSCScore, getSAScore

test_file = None
with open(test_file, 'r') as fr:
    true_smis, false_smis = [], []
    pos_smis, neg_smis = [], []

# 1. 

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
plt.savefig('bench_result.png', format='png')
