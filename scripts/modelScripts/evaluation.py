import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.modelScripts.data import GraphDataset
from scripts.modelScripts.model import SVS
from scripts.modelScripts.utils import working_dir_setting
from torch.utils.data import DataLoader


def model_evaluate(data_dir, save_dir, evaluation_args):
    # 0. inintial setting
    now = datetime.now()
    since_from = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    since = time.time()
    save_dir = working_dir_setting(save_dir, "model_inference")
    # with open(f'{data_dir}/infer_tensor.pt', 'rb') as fr:
    #    infer_tensor = torch.load(fr)
    # 1. Set inference parameters
    n_GAT_layer = 5
    n_fc_layer = 5
    max_num_atoms = evaluation_args.max_num_atoms
    len_features = evaluation_args.len_features
    n_dim = evaluation_args.n_dim
    max_step = evaluation_args.max_step
    cuda_device = evaluation_args.cuda_device
    model_path = evaluation_args.model_path
    torch.set_num_threads(int(evaluation_args.num_threads))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(cuda_device))
    # os.environ['ONP_NUM_THREADS'] = '1'

    predictor = SVS(
        n_dim=n_dim,
        n_GAT_layer=n_GAT_layer,
        n_fc_layer=n_fc_layer,
        len_features=len_features,
        num_class=max_step + 1,
        dropout=0,
    )
    # print(os.path.isfile(model_path))
    predictor.load_state_dict(torch.load(model_path))
    predictor.cuda()
    predictor.eval()
    inference_setting_inform = [
        "----- Input Config Information -----\nGAT_model\n",
        f"Since from: {since_from}\nCUDA_DEVICE: {cuda_device}\nmax_step: {max_step}\n",
        f"max_num_atoms: {max_num_atoms}\nlen_features: {len_features}\nn_dim: {n_dim}\n",
        f"n_GAT_layer: {n_GAT_layer}\nn_fc_layer: {n_fc_layer}\nmodel_path: {model_path}\n",
        f"data_dir: {data_dir}\nsave_dir: {save_dir}\n",
        "\n----- Inference Result -----\n",
    ]
    for line in inference_setting_inform:
        print(line, end="")

    with open(f"{save_dir}/inference_result.txt", "w") as fw:
        fw.writelines(inference_setting_inform)

    # 2. Inference phase
    total_prob_list = []
    for step in range(max_step + 1):
        if step == 0:
            data_path = f"{data_dir}/infer_neg{max_step}.pt"
        else:
            data_path = f"{data_dir}/infer_pos{step}.pt"
        data_list = torch.load(data_path)
        smi_list, feature_list, adj_list, label_list = data_list
        test_dataset = GraphDataset([feature_list, adj_list, label_list])
        test_data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        raw_outputs = []
        start = time.time()
        with torch.no_grad():
            for batch in test_data_loader:
                x = batch["feature"].float().cuda()
                A = batch["adj"].float().cuda()
                y = batch["label"].long().cuda()
                y_pred = predictor(x, A)
                raw_outputs += y_pred.cpu().tolist()
        end = time.time()
        with open(f"{save_dir}/inference_result.txt", "a") as fw:
            fw.write(f"Inference time: {end-start:.2f}\n")
        print(f"Inference time: {end-start:.2f}")

        # expected_value: averaged value. sum_i(prob_i*i)
        raw_outputs = torch.Tensor(raw_outputs)
        pred_probs = F.softmax(raw_outputs, dim=1)
        probs_only_positives = F.softmax(
            raw_outputs[:, 1:], dim=1
        )  # negative not included
        neg_prob = pred_probs[:, 0]
        other_prob = 1 - neg_prob
        # binary_prob = torch.hstack((other_prob,neg_prob))

        data_size, num_pos_class = probs_only_positives.shape
        labels = np.ones(num_pos_class) + np.array(range(num_pos_class))  # [1,2,3,4]
        expected_value = torch.sum(
            probs_only_positives * labels, dim=1
        )  # negative not included
        pred_label = np.array(torch.argmax(pred_probs, dim=1))

        if step == 0:
            save_path = f"{save_dir}/neg{max_step}_pred_label.txt"
        else:
            save_path = f"{save_dir}/pos{step}_pred_label.txt"

        with open(save_path, "w") as fw:
            fw.write("SMILES\tTrue_label\tPredicted_label\tExpected_value\n")
            fw.write("-------------------------\n")
            for data_idx in range(data_size):
                exp = f"{expected_value[data_idx]:.2f}"
                true_label = int(label_list[data_idx])
                pred = int(pred_label[data_idx])
                if pred == 0:
                    exp = None
                fw.write(f"{smi_list[data_idx]}\t{true_label}\t{pred}\t{exp}\n")
                # smi, true_label, pred_label, pred_probabilities
                total_prob_list.append(
                    [smi_list[data_idx], true_label, pred, pred_probs[data_idx]]
                )

    # 3. Finish and save the result
    now = datetime.now()
    finished_at = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    time_passed = int(time.time() - since)
    torch.save(total_prob_list, f"{save_dir}/total_prob_list.pt")

    with open(f"{save_dir}/total_pred_label.txt", "w") as fw:
        for step in range(max_step + 1):
            if step == 0:
                with open(f"{save_dir}/neg{max_step}_pred_label.txt", "r") as fr:
                    lines = fr.readlines()
            else:
                with open(f"{save_dir}/pos{step}_pred_label.txt", "r") as fr:
                    lines = fr.readlines()
            fw.writelines(lines)

    with open(f"{save_dir}/evaluation_result.txt", "a") as fw:
        fw.write(f"\n----- Evaluation Finished -----\n")
        fw.write(f"finished at : {finished_at}\n")
        fw.write(
            "time passed: [%dh:%dm:%ds]\n"
            % (time_passed // 3600, (time_passed % 3600) // 60, time_passed % 60)
        )
        fw.write(f"You can find the result files at:\n")
        fw.write(f"\tSave Dir: {save_dir}/total_prob_list.pt\n")
        fw.write(f"\tTotal Prob List: {save_dir}/total_prob_list.pt\n")
        fw.write(f"\tTotal Pred Label: {save_dir}/total_pred_label.pt\n")
    print(f"\n----- Evaluation Finished -----")
    print(f"finished at : {finished_at}")
    print(
        "time passed: [%dh:%dm:%ds]"
        % (time_passed // 3600, (time_passed % 3600) // 60, time_passed % 60)
    )
    print(f"You can find the result files at:")
    print(f"\tSave Dir: {save_dir}/total_prob_list.pt")
    print(f"\tTotal Prob List: {save_dir}/total_prob_list.pt")
    print(f"\tTotal Pred Label: {save_dir}/total_pred_label.pt")

    return save_dir
