import argparse
import os
from scripts.modelScripts.preprocessing import train_data_preprocess
from scripts.modelScripts.train import train_SVS
from scripts.utils import logger
from scripts.utils import train_save_dir_setting
from datetime import datetime

def main_train(args):
    # 0. Reading config and directory setting
    args.root = os.path.dirname(os.path.realpath(__file__))
    args.data_dir, args.save_dir = train_save_dir_setting(args)
    args.logger = logger(os.path.join(args.save_dir, 'training.log'))
    args.logger(f'Model training save directory is:\n  {args.save_dir}')
    
    # 1. Training data preprocessing
    train_data_preprocess(args=args)

    # 2. model train
    train_SVS(args=args)

# main operation:
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'path to retro analysis result directory')
    parser.add_argument('--model_save_name', type = str, help = "model's name to be saved as")
    parser.add_argument("--data_preprocessing_name", type = str, default='training_data', help = "name of preprocessed data.")       #TODO: move this part below
    # Default setting
    parser.add_argument('--num_data', type = str, default=250000, help = 'number of data used in train/val/test.')
    parser.add_argument('--num_cores', type = str, default=4, help = 'number of cores')
    parser.add_argument('--max_step',type=int, default=4, help='the maximum number of reaction steps')
    parser.add_argument('--max_num_atoms', type = int, default=64, help = 'maximum number of atoms')
    parser.add_argument('--batch_size', type = int, default=512, help = 'batch size')
    parser.add_argument('--num_threads', type = int, default=2, help = 'the number of threads')
    parser.add_argument('--len_features', type = int, default=36, help = 'dim of atomic feature')
    parser.add_argument('--n_conv_layer', type = int, default=5, help = 'number of convolution layers')
    parser.add_argument('--n_fc_layer', type = int, default=4, help = 'number of fully connected layers')
    parser.add_argument('--num_heads', type = int, default=8, help = 'number of heads for multi-head attention')
    parser.add_argument('--conv_dim', type = int, default=256, help = 'graph conv layer hidden dimension')
    parser.add_argument('--fc_dim', type = int, default=128, help = 'fc layer hidden dimension')
    parser.add_argument('--dropout', type = float, default=0.2, help = 'dropout for edge drop and normal dropout')
    parser.add_argument('--num_epoch',type = float, default=300, help = 'number of epochs')
    parser.add_argument('--lr',type = float, default=2e-4, help = 'learning rate')
    parser.add_argument('--gamma',type = float, default=0.99, help = 'decaying rate')
    parser.add_argument('--decay_epoch',type = float, default=0, help = 'decaying starts epoch')
    parser.add_argument('--load_model',type = int, default=0, help = 'epoch to load')
    args = parser.parse_args()
    main_train(args)
