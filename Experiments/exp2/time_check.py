import argparse
import os
import pickle
import sys
import time
from os.path import dirname

from rdkit.Chem import MolFromSmiles as Mol

sys.path.append(f"{dirname(dirname(os.path.abspath(dirname(__file__))))}")

from scripts.modelScripts.model import DFRscore
from scripts.utils import logger


def time_check(model, file_path, num_to_test):
    smis = []
    with open(file_path, "r") as fr:
        for _ in range(num_to_test):
            smis.append(fr.readline().rstrip())

    since = time.time()
    scores = model.smiListToScores(smis)
    elapsed_time = time.time() - since
    return elapsed_time


# 1. Experiments setting
parser = argparse.ArgumentParser()
parser.add_argument("--num_cores", type=int)
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--model_path", type=str)
parser.add_argument("--test_path", type=str)
args = parser.parse_args()

num_to_test = 10000

result_log_path = f"CUDA_{args.use_cuda}.log"
log = logger(result_log_path)

# 2. Model load
log("===== Input Config Information ======")
log(f"num_cores: {args.num_cores}")
log(f"test path: {args.test_path}")
log(f"model_path: {args.model_path}")
log(f"use_cuda: {args.use_cuda}")
log(f"num to test: {num_to_test}")
predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)
if args.use_cuda:
    predictor = predictor.cuda()
log(predictor)

# 3. Get Evaluation Metrics
log("\n===== Time check started =====")
for data_set in ["ZINC.smi", "ChEMBL.smi", "MOSES.smi"]:
    data_path = os.path.join(args.test_path, data_set)
    log(data_path)
    elapsed_time = time_check(predictor, data_path, num_to_test)
    log(f"elapsed_time: {round(elapsed_time, 3)}\n")
