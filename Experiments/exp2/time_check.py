import argparse
import os
import sys
import time
from os.path import dirname
from typing import Tuple

sys.path.append(f"{dirname(dirname(os.path.abspath(dirname(__file__))))}")

import numpy as np
import torch
import torch.nn as nn

from scripts.modelScripts.model import DFRscore
from scripts.utils import logger


class WarmingupModel(nn.Module):
    def __init__(self, num_layers: int, hid_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(hid_dim, hid_dim) for _ in range(num_layers)]
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.act(x)
        return x


def warmup(num_layers, hid_dim):
    model = WarmingupModel(num_layers, hid_dim).cuda()
    tensor = torch.empty([64, 128, 128]).float().cuda()

    for _ in range(100000):
        x = model(tensor)

    return


def cpu_time_check(model: DFRscore, file_path, num_to_test) -> np.array:
    smis = []
    with open(file_path, "r") as fr:
        for _ in range(num_to_test):
            smis.append(fr.readline().rstrip())

    with torch.no_grad():
        since = time.time()
        scores = model.smiListToScores(smis)
        cpu_time = time.time() - since
    print(scores)
    print(len(scores))

    return cpu_time


def gpu_time_check(model: DFRscore, file_path, num_to_test) -> np.array:
    smis = []
    with open(file_path, "r") as fr:
        for _ in range(num_to_test):
            smis.append(fr.readline().rstrip())

    model = model.cuda()
    with torch.no_grad():
        since = time.time()
        scores = model.smiListToScores(smis)
        gpu_time = time.time() - since

    model = model.to("cpu")
    return gpu_time


def proc_time_check(model: DFRscore, file_path, num_to_test) -> np.array:
    smis = []
    with open(file_path, "r") as fr:
        for _ in range(num_to_test):
            smis.append(fr.readline().rstrip())

    with torch.no_grad():
        since = time.time()
        _ = model._preprocessing_time_check(smis)
        preprocessing_time = time.time() - since

    return preprocessing_time


if __name__ == "__main__":
    # Experiments setting
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cores", type=int)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    num_to_test = 100000
    num_iter = 3

    result_log_path = f"benchmark_{args.seed}.log"
    log = logger(result_log_path)

    log("===== Input Config Information ======")
    log(f"num_cores: {args.num_cores}")
    log(f"test path: {args.test_path}")
    log(f"model path: {args.model_path}")
    log(f"num to test: {num_to_test}")
    log(f"data seed: {args.seed}")

    # Model load
    predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)
    log(predictor)

    # Warmup
    log("\n===== Warmup GPU ======")
    warmup(num_layers=10, hid_dim=128)
    log("Done.")

    # Get Evaluation Metrics
    log("\n===== Time check started =====")
    if args.seed == 0:
        data_list = ["ZINC.smi", "ChEMBL.smi", "MOSES.smi"]
    elif args.seed == 1:
        data_list = ["ChEMBL.smi", "MOSES.smi", "ZINC.smi"]
    elif args.seed == 2:
        data_list = ["MOSES.smi", "ZINC.smi", "ChEMBL.smi"]
    elif args.seed == 3:
        data_list = ["ChEMBL.smi"]

    for data_set in data_list:
        proc_times, cpu_times, gpu_times = [], [], []
        data_path = os.path.join(args.test_path, data_set)
        log(data_path)

        for _ in range(num_iter):
            cpu_times.append(cpu_time_check(predictor, data_path, num_to_test))
        for _ in range(num_iter):
            gpu_times.append(gpu_time_check(predictor, data_path, num_to_test))
        for _ in range(num_iter):
            proc_times.append(proc_time_check(predictor, data_path, num_to_test))

        cpu_times = np.array(cpu_times)
        gpu_times = np.array(gpu_times)
        proc_times = np.array(proc_times)

        log(f"cpu time: {cpu_times}, {cpu_times.mean()}")
        log(f"gpu time: {gpu_times}, {gpu_times.mean()}")
        log(f"processing time: {proc_times}, {proc_times.mean()}\n")
