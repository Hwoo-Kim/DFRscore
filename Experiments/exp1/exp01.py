import argparse
import os
import pickle
import sys
from os.path import dirname

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

proj_dir = dirname(dirname(os.path.abspath(dirname(__file__))))
sys.path.append(proj_dir)

from scripts.modelScripts.experiments import runExp01
from scripts.modelScripts.model import DFRscore
from scripts.utils import logger

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIG_SIZE = 12
BIGGER_SIZE = 14


def make_violin(target_name):
    # Read result file
    test_file_path = f"{target_name}/scores.pkl"
    with open(test_file_path, "rb") as f:
        data_dict = pickle.load(f)
    save_fig_path = f"{target_name}.pdf"

    # Plot setting
    plot_order = ["1", "2", "3", "4", "unsolved"]
    color_list = np.array(
        [
            [4, 96, 217],  # 1
            [4, 227, 186],  # 2
            [200, 159, 5],  # 3
            [242, 92, 5],  # 4
            [153, 0, 0],  # unsolved
        ]
    )
    color_list = color_list / 256
    data = data_dict["dfr"]
    data_plot = []
    for label in plot_order:
        if label == "unsolved":
            data_plot.append(data["0"])
        else:
            data_plot.append(data[label])

    # Figure configuration
    # ax = plt.subplot(1, 1, 1)
    fig = plt.figure(figsize=[6.4, 5.2])
    plt.xticks(
        ticks=np.arange(5) + 1, labels=["1", "2", "3", "4", "unsolved"], fontsize=20
    )
    plt.yticks(fontsize=20)
    fig.suptitle(target_name, fontsize=25)

    # Plot the distribution
    positions = list(range(1, len(plot_order) + 1))
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

    plt.savefig(save_fig_path, format="pdf")
    return save_fig_path


if __name__ == "__main__":
    # experiment configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--num_cores", type=int)
    parser.add_argument("--class_size", type=int)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()
    test_file_to_score = os.path.join(
        proj_dir, f"save/{args.test_data}/pubchem_removed"
    )
    result_dir = os.path.join(dirname(__file__), args.test_data)
    predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)

    if os.path.exists(result_dir):  # do not need to calculate scores.
        print(f"The calculated scores for {args.test_data} already exists.")
        log = logger(os.path.join(dirname(__file__), ".tmp_log.txt"))

    else:  # calculate all the scores.
        os.mkdir(args.test_data)
        result_log_path = os.path.join(result_dir, "exp_result.log")
        log = logger(result_log_path)

        log("===== Calculate Scores ======")
        log(f"num_cores: {args.num_cores}")
        log(f"save_dir: {args.test_data}")
        log(f"test file path: {test_file_to_score}")
        log(predictor)

    # 1. obtain the calculated scores
    data_df = runExp01(
        predictor,
        save_dir=args.test_data,
        num_cores=args.num_cores,
        test_file_path=test_file_to_score,
        logger=log,
        class_size=args.class_size,
    )

    # 2. plot viloin plots
    fig_path = make_violin(args.test_data)
    log(f"Figure with violin plots is in: {os.path.abspath(fig_path)}")

    # 3. plot SA vs DFR / SC vs DFR scatter plots
    # 3-0. Re-label the unsolved ones
    true_label = data_df["True Label"].to_numpy()
    data_df["True Label"] = np.where(true_label == "5", "U", true_label)
    columns = list(data_df.columns)
    columns[2] = "FRA Label"
    data_df.columns = columns

    # 3-1. clipping for sascores
    sas = data_df["SA score"].to_numpy()
    data_df["SA score"] = np.where(sas > 6, 6, sas)

    # 3-2. plot theme setting
    plot_context = sns.plotting_context()
    plot_context["grid.linewidth"] = 0.5
    plot_context["grid.linewidth"] = 0.5
    plot_context["font.size"] = 24
    plot_context["legend.title_fontsize"] = 14
    plot_context["legend.fontsize"] = 12
    sns.set_theme(context=plot_context, style="whitegrid", palette="pastel")
    dot_size = 2.4

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 12.0))
    plt.subplots_adjust(hspace=0.3)
    # sns.despine(left=True, bottom=True)

    # SA vs DFR scatter plot
    g = sns.scatterplot(
        data=data_df,
        x="SA score",
        y="DFRscore",
        hue="FRA Label",
        hue_order=["1", "2", "3", "4", "U"],
        s=dot_size,  # dot size
        palette="mako",
        # palette="crest_r",
        # palette="ch:r=-.2,d=.3_r",
        linewidth=0,
        ax=axes[0],
    )
    g.set_xticks([2, 3, 4, 5, 6], fontsize=22)
    g.set_yticks([1, 2, 3, 4, 5, 6], fontsize=22)

    # SC vs DFR scatter plot
    print
    g = sns.scatterplot(
        data=data_df,
        x="SC score",
        y="DFRscore",
        hue="FRA Label",
        hue_order=["1", "2", "3", "4", "U"],
        s=dot_size,  # dot size
        palette="mako",
        # palette="crest_r",
        # palette="ch:r=-.2,d=.3_r",
        linewidth=0,
        ax=axes[1],
    )
    print
    g.set_xticks([2, 3, 4, 5, 6], fontsize=22)
    g.set_yticks([1, 2, 3, 4, 5, 6], fontsize=22)

    # save the figure
    fig_path = f"{result_dir}/{args.test_data}_SA_SC_DFR.pdf"
    plt.savefig(fig_path, format="pdf")
    log(
        f"Figure for SAscore vs DFRscore & SCscore vs DFRscore is in: {os.path.abspath(fig_path)}\n"
    )
