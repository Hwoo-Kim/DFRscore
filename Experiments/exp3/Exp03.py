import argparse
import os
import pickle
import sys
from os.path import dirname

sys.path.append(f"{dirname(dirname(os.path.abspath(dirname(__file__))))}")

from scripts.modelScripts.experiments import runExp03
from scripts.modelScripts.model import DFRscore
from scripts.utils import logger

# 1. Experiments setting
parser = argparse.ArgumentParser()
parser.add_argument("--test_data", type=str)
parser.add_argument("--num_cores", type=int)
parser.add_argument("--model_path", type=str)
args = parser.parse_args()
test_file_path = f"/home/hwkim/DFRscore/save/{args.test_data}/exp03"

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
predictor = DFRscore.from_trained_model(args.model_path, num_cores=args.num_cores)
predictor = predictor.cuda()
log(predictor)

# 2. Get Evaluation Metrics
result = runExp03(
    predictor, save_dir=args.test_data, test_file_path=test_file_path, logger=log
)

log("\n** Exp03 finished **")
