import argparse
import os
#from scripts.modelScripts.preprocessing import train_data_preprocess
#from scripts.modelScripts.train import train_SVS
#from scripts.utils import logger, train_save_dir_setting, get_cuda_visible_devices
#from datetime import datetime
#os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_visible_devices(1)

from scripts.modelScripts.preprocessing_mpnn import train_data_preprocess
from scripts.modelScripts.train_mpnn import train_DFRscore
from scripts.utils import logger, train_save_dir_setting, get_cuda_visible_devices
from datetime import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = get_cuda_visible_devices(1)

def main_train(args):
    # 0. Reading config and directory setting
    args.root = os.path.dirname(os.path.realpath(__file__))
    args.data_dir, args.save_dir = train_save_dir_setting(args)
    args.logger = logger(os.path.join(args.save_dir, 'training.log'))
    args.logger(f'Model training save directory is:\n  {args.save_dir}')
    
    # 1. Training data preprocessing
    now = datetime.now()
    since_inform = now.strftime('%Y. %m. %d (%a) %H:%M:%S')
    preprocess_dir = os.path.join(args.data_dir, args.data_preprocessing)
    args.preprocess_dir = preprocess_dir
    if os.path.exists(preprocess_dir):
        print('1. Data preprocessing phase')
        print('  Processed data already exists.')
        print('  Training data preprocessing finished.')
        args.data_dir = preprocess_dir
    else:
        os.mkdir(preprocess_dir)
        args.preprocess_logger = logger(os.path.join(preprocess_dir, 'preprocessing.log'))
        args.preprocess_logger('1. Data preprocessing phase')
        args.preprocess_logger(f'  Started at: {since_inform}')
        args.preprocess_logger(f'  Data will be generated in: {preprocess_dir}')
        args.data_dir = train_data_preprocess(args=args)

    # 2. model train
    #train_SVS(args=args)
    train_DFRscore(args=args)

# main operation:
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'path to retro analysis result directory')
    parser.add_argument('--save_name', type = str, help = "model's name to be saved as")
    parser.add_argument('--num_data', type = int, help = 'number of data used in train/val/test.')
    # Default setting
    parser.add_argument("--data_preprocessing", type = str, default='processed_data', help = "name of preprocessed data.")
    parser.add_argument('--num_cores', type = str, default=4, help = 'number of cores')
    parser.add_argument('--max_step',type=int, default=4, help='the maximum number of reaction steps')
    parser.add_argument('--max_num_atoms', type = int, default=64, help = 'maximum number of atoms')
    parser.add_argument('--batch_size', type = int, default=32, help = 'batch size')
    parser.add_argument('--num_threads', type = int, default=2, help = 'the number of threads')
    parser.add_argument('--node_dim', type = int, default=30, help = 'dim of node feature')
    parser.add_argument('--edge_dim', type = int, default=5, help = 'dim of edge feature')
    parser.add_argument('--num_layers', type = int, default=3, help = 'number of message passing layers')
    parser.add_argument('--hidden_dim', type = int, default=80, help = 'hidden feature dimension')
    parser.add_argument('--message_dim', type = int, default=80, help = 'message passing dimension')
    parser.add_argument('--dropout', type = float, default=0.2, help = 'dropout for edge drop and normal dropout')
    parser.add_argument('--num_epoch',type = int, default=100, help = 'number of epochs')
    parser.add_argument('--lr',type = float, default=5e-4, help = 'learning rate')
    # For ReduceLROnPlateau
    parser.add_argument('--factor',type = float, default=0.5, help = 'decreasing factor')
    parser.add_argument('--patience',type = int, default=10, help = 'number of epochs with no improvement after which learning rate will be reduced')
    parser.add_argument('--threshold',type = float, default=1e-3, help = 'Threshold for measuring the new optimum, to only focus on significant changes')
    parser.add_argument('--min_lr',type = float, default=1e-7, help = ' A lower bound on the learning rate of all param groups')
    args = parser.parse_args()
    main_train(args)

