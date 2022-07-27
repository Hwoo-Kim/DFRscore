import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

# 1. Reading result file
target_data = sys.argv[1]
test_file_path = f"./{target_data}/scores.pkl"
with open(test_file_path, "rb") as f:
    data_dict = pickle.load(f)

dfrs = data_dict['dfr']
sas = data_dict['sa']
scs = data_dict['sc']

# 2. Data plot
plt.subplot(313)
ax1 = plt.subplot(311)
ax1.hist(dfrs, cumulative=False, bins=50, label='cumulative=False')
ax2 = plt.subplot(312)
ax2.hist(sas, cumulative=False, bins=50, label='cumulative=False')
ax3 = plt.subplot(313)
ax3.hist(scs, cumulative=False, bins=50, label='cumulative=False')

plt.savefig(f"{target_data}/{target_data}.png", format="png")
exit()

plot_list = ["1", "2", "3", "4"]
color_list = [
    np.array([50, 50, 50]),
    np.array([100, 100, 233]),
    np.array([200, 128, 50]),
    np.array([230, 30, 30]),
]
for i in range(len(color_list)):
    color_list[i] = color_list[i] / 255

# print (data_dict.keys())

metric = "dfr"
plt.figure(figsize=[6.4, 4.8])
data_plot = []
data = data_dict[metric]
ax = plt.subplot(1, 1, 1)


ax.set_xlabel("True label", fontsize=15)
ax.set_ylabel("Pred label", fontsize=15)
# ax.set_title(metric_name_dict[metric])
ax.set_title(target_data, fontsize=25)
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)
for label in plot_list:
    data_plot.append(data[label])
    for d in data[label]:
        if d < 1:
            print(d)

positions = list(range(1, len(plot_list) + 1))
figure_plot = plt.violinplot(
    data_plot, positions=positions, showextrema=True, points=50
)
for idx in range(len(color_list)):
    violin = figure_plot["bodies"][idx]
    violin.set_facecolor(color_list[idx])
    violin.set_edgecolor("black")
    violin.set_alpha(1)
    bar = figure_plot["cbars"]
    bar.set_color([0, 0, 0])

plt.savefig(f"{target_data}/{target_data}.png", format="png")
