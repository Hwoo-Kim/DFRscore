import argparse
import os
import pickle
import sys
from os.path import dirname
import matplotlib.pyplot as plt
import numpy as np

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

proj_dir = dirname(dirname(os.path.abspath(dirname(__file__))))
sys.path.append(proj_dir)

from scripts.modelScripts.experiments import runExp01
from scripts.modelScripts.model import DFRscore
from scripts.utils import logger

def make_violin(target_name):
    # Read result file
    test_file_path = f"{target_name}/scores.pkl"
    with open(test_file_path, "rb") as f:
        data_dict = pickle.load(f)
    save_fig_path = f"{target_name}.png"
    
    # Plot setting
    plot_list = ["1", "2", "3", "4", "0"]   # "0" means "unsolved"
    color_list = [
        [4, 96, 217],       # 1
        [4, 227, 186],      # 2
        [200, 159, 5],      # 3
        [242, 92, 5],       # 4
        [153, 0, 0],        # unsolved
    ]
    color_list = np.array(color_list)/256
    #for i in range(len(color_list)):
    #    color_list[i] = color_list[i] / 255
    
    # Figure configuration
    metric = "dfr"
    plt.figure(figsize=[6.4, 4.8])
    data_plot = []
    data = data_dict[metric]
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel("$y_{true}$", fontsize=15)
    ax.set_xticks(ticks=np.arange(5)+1, labels=["1", "2", "3", "4", "unsolved"])
    ax.set_ylabel("$y_{pred}$", fontsize=15)
    # ax.set_title(metric_name_dict[metric])
    ax.set_title(target_name, fontsize=25)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    for label in plot_list:
        data_plot.append(data[label])
        #for d in data[label]:
        #    if d < 0.5:
        #        print(d)
    
    # Plot the distribution
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
    
    plt.savefig(save_fig_path, format="png")
    return save_fig_path


if __name__ == "__main__":
    # experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--num_cores", type=int)
    parser.add_argument("--class_size", type=int)
    parser.add_argument("--model_path", type=str)
    #parser.add_argument("--only_DFR", action="store_true")
    args = parser.parse_args()
    test_file_to_score = os.path.join(proj_dir, f"save/{args.test_data}/pubchem_removed")
    result_dir = os.path.join(dirname(__file__), args.test_data)

    if os.path.exists(result_dir):  # do not need to calculate scores.
        print(f"The calculated scores for {args.test_data} already exists.")

        # plot the results
        fig_path = make_violin(args.test_data)
        print(f"Figure is in: {os.path.abspath(fig_path)}\n")
    
    else:                           # calculate all the scores.
        os.mkdir(args.test_data)

        result_log_path = os.path.join(result_dir, "exp_result.log")
        log = logger(result_log_path)
        
        log("===== Calculate Scores ======")
        log(f"num_cores: {args.num_cores}")
        log(f"save_dir: {args.test_data}")
        log(f"test file path: {test_file_to_score}")
        predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)
        log(predictor)
        
        # calculate evaluation metrics
        result = runExp01(
            predictor,
            save_dir=args.test_data,
            num_cores=args.num_cores,
            test_file_path=test_file_to_score,
            logger=log,
            class_size=args.class_size,
            only_DFR=args.only_DFR,
        )
        log("\n** Exp01 finished **")

        # plot the results
        fig_path = make_violin(args.test_data)
        log(f"Figure is in: {os.path.abspath(fig_path)}\n")
