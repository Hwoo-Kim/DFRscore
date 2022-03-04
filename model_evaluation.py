import torch
from scripts.preprocessing import GAT_evaluation_data_generation
from scripts.evaluation import GAT_model_evaluation
from scripts.metrics import  model_eval
from scripts.utils import working_dir_setting
from datetime import datetime
import argparse
import time
import os
os.environ['PYTHONPATH'] += f':{os.getcwd()}'

# 0. Read config
now = datetime.now()
since_from = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
since = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help = 'data directory', type = str)
parser.add_argument('--save_dir', help = 'save directory', type = str)
parser.add_argument('--model_path', help = 'model to use for inference', type = str)
parser.add_argument('--model_name', help = 'data or model name', type = str)
#parser.add_argument('--mode', help = "'DI' for data generation and inference, 'I' for only inference", type = str)
parser.add_argument('--max_step',help='the maximum number of reaction steps',type=int)
parser.add_argument('--shuffle_before_sampling',help='whether shuffling the dataset before sampling or not',type=str)
parser.add_argument('--max_num_atoms', help = 'maximum number of atoms', type = int)
parser.add_argument('--len_features', help = 'number of atomic features', type = int)
parser.add_argument('--n_dim', help = 'hidden dimension', type = int)
parser.add_argument('--cuda_device', help = 'cuda device', type = str)
parser.add_argument('--num_threads', help = 'the number of threads', type = int)
parser.add_argument("--each_class_size", help = "the number of samples for each class", type = int)
parser.add_argument("--exp", help = "for which experiment", type = str)
args = parser.parse_args()

# Setting cofig
data_directory = args.data_dir 
if os.path.isdir('result_evaluation')==False:
    os.mkdir('result_evaluation')
save_directory = f'result_evaluation/{args.save_dir}'
if os.path.isdir(save_directory)==False:
    os.mkdir(save_directory)
model_name= args.model_name
if data_directory[-1] =='/':
    data_directory= data_directory[:-1]
if save_directory[-1] =='/':
    save_directory= save_directory[:-1]
if model_name[-1] =='/':
    model_name = model_name[:-1]
save_directory = f'{save_directory}/{model_name}'
if os.path.isdir(save_directory)==False:
    os.mkdir(save_directory)
max_step = args.max_step
shuffle_before_sampling = args.shuffle_before_sampling=='True'

# 1. input data generation
print('\n##### Data generation Phase #####')
input_data_path = f'{save_directory}/inference_data'
if os.path.isdir(input_data_path):
    print('Data exists!')
else:
    input_data_path = GAT_evaluation_data_generation(data_directory, save_directory, args, with_smiles=True)

# 2. model inference
print('\n##### Model Inference Phase #####')
inf_data_dir = GAT_model_evaluation(data_dir=input_data_path, save_dir=save_directory, inference_args=args)

# 3. model evaluation
print('\n##### Model Evaluation Phase #####')
# data_dir = total_prob_list.pt
model_eval(data_dir=inf_data_dir, save_dir=save_directory, max_step=max_step, threshold=0.5, model_path=args.model_path, exp=args.exp)

print('\n##### Finished #####\n')
