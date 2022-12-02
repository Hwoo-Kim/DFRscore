import multiprocessing as mp
import os
import pickle
import sys

from .metrics import BinaryConfusionMatrix as BinCM
from .metrics import get_AUROC

sys.path.append(f"{os.path.dirname(os.path.abspath(os.path.dirname(__file__)))}")
import csv

import numpy as np
from getScores import getSAScore, getSCScore, rescale_score


def runExp01(
    predictor, save_dir, num_cores, test_file_path, logger, class_size, only_DFR: bool
):
    """
    Arguments:
      predictor: DFRscore object already restored by trained model.
      save_dir: The directory where Experiment result will be saved.
      num_cores: The number of cpu cores to use for multiprocessing.
      test_file_path: Path to the exp01 test file.
      logger: utils.Logger obj.
      class_size: The number of molecules for each class (pos1, pos2, ...).
      only_DFR: (bool) Whether calculating SA and SC scores or not.
    """
    max_step = predictor.max_step
    class_sizes = [class_size] * (max_step + 1)

    # 1. reading test files
    test_smi_list = []

    for i in range(max_step + 1):
        if i == 0:
            with open(os.path.join(test_file_path, f"neg{max_step}.smi"), "r") as fr:
                data = fr.read().splitlines()[: class_sizes[i]]
        else:
            with open(os.path.join(test_file_path, f"pos{i}.smi"), "r") as fr:
                data = fr.read().splitlines()[: class_sizes[i]]
        test_smi_list += data

    true_label_list = []
    for idx, l in enumerate(class_sizes):
        if idx == 0:
            true_label_list += [0 for i in range(l)]
        else:
            true_label_list += [1 for i in range(l)]
    true_label_list = np.array(true_label_list)

    # 2. AUROC - comparison with SA and SC
    logger("Number of each data:")
    for step in range(max_step + 1):
        if step == 0:
            logger(f" Neg: {class_sizes[step]}")
        else:
            logger(f" Pos{step}: {class_sizes[step]}")

    logger("\n===== Calculating Scores =====")
    logger("Calculating DFRscore...", end="\t")
    DFRscores = predictor.smiListToScores(test_smi_list)
    logger("Done.")

    if not only_DFR:
        logger("Calculating SA score...", end="\t")
        with mp.Pool(num_cores) as p:
            SAScores = p.map(getSAScore, test_smi_list)
        #SAScores = rescale_score(SAScores, 1, 10, reverse=True)
        logger("Done.")

        logger("Calculating SC score...", end="\t")
        SCScores = getSCScore(test_smi_list)
        #SAScores = rescale_score(SCScores, 1, 5, reverse=True)
        logger("Done.")
    else:
        logger("Only_DFR option is True, so others are not calculated.")

    logger("\n===== Calculating AUROCs =====")
    logger("1. DFRscore")
    dfr_auroc = get_AUROC(true_label_list, -1 * np.array(DFRscores))
    logger(f"auroc = {dfr_auroc}, auroc was calculated by reversed score.")

    if not only_DFR:
        logger("2. SA score")
        sas_auroc = get_AUROC(true_label_list, np.array(SAScores))
        logger(f"auroc = {sas_auroc}, auroc was calculated by not reversed score.")

        logger("3. SC score")
        scs_auroc = get_AUROC(true_label_list, np.array(SCScores))
        logger(f"auroc = {scs_auroc}, auroc was calculated by not reversed score.")

    # 3. Save the calculated scores
    logger("\n===== Saving the Calculated Scores =====")
    DFRs_dict = dict()
    SAScores_dict = dict()
    SCScores_dict = dict()
    index = 0
    for i in range(len(class_sizes)):
        n = class_sizes[i]
        key = str(i)
        DFRs_dict[key] = DFRscores[index : index + n]
        if not only_DFR:
            SAScores_dict[key] = SAScores[index : index + n]
            SCScores_dict[key] = SCScores[index : index + n]
        index += n

    result_datas = dict()
    result_datas["dfr"] = DFRs_dict
    if not only_DFR:
        result_datas["sa"] = SAScores_dict
        result_datas["sc"] = SCScores_dict

    # Save pickle files
    with open(os.path.join(save_dir, "scores.pkl"), "wb") as f:
        pickle.dump(result_datas, f)

    return True


def runExp03(predictor, test_file_path, logger):
    """
    Arguments:
      predictor: DFRscore object already restored by trained model.
      test_file_path: Path to the exp01 test file.
      logger: utils.Logger obj.
    """
    max_step = predictor.max_step

    # 1. reading test files
    each_class_sizes = []
    test_smi_list = []

    test_data = dict()
    for i in range(max_step + 1):
        if i == 0:
            with open(os.path.join(test_file_path, f"neg{max_step}.smi"), "r") as fr:
                test_data[0] = fr.read().splitlines()
        else:
            with open(os.path.join(test_file_path, f"pos{i}.smi"), "r") as fr:
                test_data[i] = fr.read().splitlines()

    for i in range(max_step + 1):
        data = test_data[i]
        each_class_sizes.append(len(data))
        test_smi_list += data

    # 2. Evaluate the refining ability
    logger("Number of each data:")
    for step in range(max_step + 1):
        if step == 0:
            logger(f" Neg: {each_class_sizes[step]}")
        else:
            logger(f" Pos{step}: {each_class_sizes[step]}")

    # Get DFR scores
    logger("\n===== Calculating Scores =====")
    logger("Calculating DFRscore...", end="\t")
    DFRscores = predictor.smiListToScores(test_smi_list)
    logger("Done.")

    M = max_step
    for max_step in range(1, M + 1):
        logger(f"\n** Current max step: {max_step} **")
        true_label_list = []
        for idx, l in enumerate(each_class_sizes):
            if idx == 0 or idx > max_step:
                true_label_list += [0 for i in range(l)]
            else:
                true_label_list += [1 for i in range(l)]
        true_label_list = np.array(true_label_list)

        threshold = max_step + 0.5
        pred_label_list = (DFRscores < threshold).astype(int)

        # Confusion matrix and evaluate metrics
        conf_matrix = BinCM(
            true_list=true_label_list, pred_list=pred_label_list, neg_label=0
        )
        acc, prec, recall, critical_err = conf_matrix.get_main_results()
        init_pos_ratio, ratio_change, filtering_ratio, = (
            conf_matrix.get_initial_pos_ratio(),
            conf_matrix.get_ratio_change(),
            conf_matrix.get_filtering_ratio(),
        )
        logger("Accuracy, Precision, Recall, Critical Error:")
        logger(f" {acc}, {prec}, {recall}, {critical_err}")
        logger("Initial pos ratio, Pos/Neg Ratio (before:after) =")
        logger(f" {init_pos_ratio}, {ratio_change}")
        logger("Filtering ratio =")
        logger(
            f" Pos: {filtering_ratio['pos']}, Neg: {filtering_ratio['neg']}, Total: {filtering_ratio['tot']}\n"
        )

    return True

def runExp04(
    predictor, save_dir, num_cores, test_file_path, logger, only_DFR: bool
):
    """
    Arguments:
      predictor: DFRscore object already restored by trained model.
      save_dir: The directory where Experiment result will be saved.
      num_cores: The number of cpu cores to use for multiprocessing.
      test_file_path: Path to the exp04 test file.
      logger: utils.Logger obj.
      class_size: The number of molecules for each class (pos1, pos2, ...).
      only_DFR: (bool) Whether calculating SA and SC scores or not.
    """
    max_step = predictor.max_step

    # 1. reading test files
    with open(f"{test_file_path}.smi", 'r') as fr:
        test_smi_list = fr.read().splitlines()

    # 2. Calculate scoring metrics
    logger(f"Number of each data: {len(test_smi_list)}")

    logger("\n===== Calculating Scores =====")
    logger("Calculating DFRscore...", end="\t")
    DFRscores = predictor.smiListToScores(test_smi_list)
    logger("Done.")

    if not only_DFR:
        logger("Calculating SA score...", end="\t")
        with mp.Pool(num_cores) as p:
            SAScores = p.map(getSAScore, test_smi_list)
        logger("Done.")

        logger("Calculating SC score...", end="\t")
        SCScores = getSCScore(test_smi_list)
        logger("Done.")
    else:
        logger("Only_DFR option is True, so others are not calculated.")

    logger("\n===== Saving the Calculated Scores =====")
    results_dict = dict()
    results_dict["smilies"] = test_smi_list
    results_dict["dfr"] = DFRscores
    if not only_DFR:
        results_dict["sa"] = SAScores
        results_dict["sc"] = SCScores

    # Save pickle files
    with open(os.path.join(save_dir, "scores.pkl"), "wb") as f:
        pickle.dump(results_dict, f)
    logger("Done.")

    return True
