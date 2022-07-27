import argparse
import os
import pickle
import sys
from os.path import dirname

sys.path.append(f"{dirname(dirname(os.path.abspath(dirname(__file__))))}")

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from scripts.modelScripts.experiments import runExp04
from scripts.modelScripts.model import DFRscore
from scripts.utils import logger, get_cuda_visible_devices

# 1. Experiments setting
parser = argparse.ArgumentParser()
parser.add_argument("--test_data", type=str)
parser.add_argument("--num_cores", type=int)
parser.add_argument("--model_path", type=str)
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--only_DFR", action="store_true")
args = parser.parse_args()

if not os.path.exists(args.test_data):
    os.mkdir(args.test_data)
result_log_path = os.path.join(
    os.path.abspath(dirname(__file__)), os.path.join(args.test_data, "exp4_result.log")
)
log = logger(result_log_path)

log("===== Input Config Information ======")
log(f"num_cores: {args.num_cores}")
log(f"save_dir: {args.test_data}")
log(f"test file path: {args.test_data}")
log(f"use CUDA: {args.use_cuda}")
log(f"only DFR: {args.only_DFR}")
predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)
if args.use_cuda:
    cuda_device = get_cuda_visible_devices(1)
    predictor = predictor.to(f"cuda:{cuda_device}")
log(predictor)

# 2. Test on the dataset
result = runExp04(
    predictor,
    save_dir=args.test_data,
    num_cores=args.num_cores,
    test_file_path=args.test_data,
    logger=log,
    only_DFR=args.only_DFR,
)

log("\n** Exp04 finished **")
