import argparse
import os
import pickle
import sys
from os.path import dirname

sys.path.append(f"{dirname(dirname(os.path.abspath(dirname(__file__))))}")

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")
from scripts.modelScripts.experiments import runExp01
from scripts.modelScripts.model import DFRscore
from scripts.utils import logger

# 1. Experiments setting
parser = argparse.ArgumentParser()
parser.add_argument("--test_data", type=str)
parser.add_argument("--num_cores", type=int)
parser.add_argument("--class_size", type=int)
parser.add_argument("--model_path", type=str)
parser.add_argument("--only_DFR", action="store_true")
args = parser.parse_args()
test_file_path = f"/home/hwkim/DFRscore/save/{args.test_data}/pubchem_removed"

if not os.path.exists(args.test_data):
    os.mkdir(args.test_data)
result_log_path = os.path.join(
    os.path.abspath(dirname(__file__)), os.path.join(args.test_data, "exp_result.log")
)
log = logger(result_log_path)

log("===== Input Config Information ======")
log(f"num_cores: {args.num_cores}")
log(f"save_dir: {args.test_data}")
log(f"test file path: {test_file_path}")
log(f"only DFR: {args.only_DFR}")
predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)
#predictor = predictor.cuda()
log(predictor)

# 2. Get Evaluation Metrics
result = runExp01(
    predictor,
    save_dir=args.test_data,
    num_cores=args.num_cores,
    test_file_path=test_file_path,
    logger=log,
    class_size=args.class_size,
    only_DFR=args.only_DFR,
)

log("\n** Exp01 finished **")
