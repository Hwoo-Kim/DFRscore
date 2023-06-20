import os
import sys
import pickle
import pandas as pd
from os.path import dirname

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

proj_dir = dirname(dirname(os.path.abspath(dirname(__file__))))
sys.path.append(proj_dir)

from scripts.modelScripts.model import DFRscore

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)             # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE-1)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIG_SIZE)      # fontsize of the tick labels
plt.rc('ytick', labelsize=BIG_SIZE)      # fontsize of the tick labels
plt.rc('legend', title_fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('legend', fontsize=MEDIUM_SIZE)      # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)        # fontsize of the figure title


def make_dist(model_path:str, target_dir:str, num_cores:int=4, use_cuda:bool=False):
    print(f"\nmodel_path: {model_path}")
    print(f"target_dir: {target_dir}")
    print(f"num_cores: {num_cores}")
    print(f"use_cuda: {use_cuda}")
    # load model
    model = DFRscore.from_trained_model(model_path, num_cores = num_cores)
    if use_cuda:
        model.cuda()

    dataset_name = target_dir.split('/')[-2]
    result = {"DFRscore":[], "FRA label":[]}
    data_list = ["pos1.smi", "pos2.smi", "pos3.smi", "pos4.smi", "neg4.smi"]
    for idx, file in enumerate(data_list):
        if idx == 4: idx = "unsolved"
        else: idx = str(idx+1)

        # read and evaluate result file
        file_path = os.path.join(target_dir, file)
        with open(file_path, 'r') as fr:
            data=fr.read().splitlines()
        dfrscores = model.smiListToScores(data)
        result["DFRscore"] += list(dfrscores)
        result["FRA label"] += [idx] * len(data)

    # make dataframe
    df = pd.DataFrame.from_dict(result)

    # plot figure
    plt.figure()
    #ax = sns.kdeplot(data=df, x="DFRscore", hue="FRA label",  multiple="layer", common_norm=False, fill=False, palette=COLOR_LIST)
    ax = sns.kdeplot(data=df, x="DFRscore", hue="FRA label",  multiple="layer", common_norm=False, fill=False)
    #ax.set_title(dataset_name, pad=20, fontsize=20)
    for i in range(1,5):
        ax.axvline(x=i+0.5, color="#D0D3D4", ymin=0, ymax=1, linestyle="--")
        plt.text(i+0.2,1.53, f"c={i+0.5}", fontsize=MEDIUM_SIZE+1)
    ax.set_xticks(list(range(0,9)))
    ax.set_yticks(np.arange(0,7)/4)

    # save the figure
    plt.savefig(f"{dataset_name}.pdf", format="pdf")
    print("Done.")
    return True


if __name__ == "__main__":
    model_path, target_dir = sys.argv[1], sys.argv[2]
    make_dist(model_path, target_dir)

