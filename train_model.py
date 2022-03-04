import argparse
import os
from scripts.modelScripts.preprocessing import train_data_preprocessing
from scripts.modelScripts.train import GAT_model_train
from scripts.utils import logger
from scripts.utils import train_save_dir_setting
from datetime import datetime

def main_train(args):
    # 0. Reading config and directory setting
    project_path = os.path.dirname(os.path.realpath(__file__))
    args.root = project_path
    #project_path = os.getcwd()

    assert args.data_preprocessing=='True' or args.data_preprocessing=='False', \
        "given args.data_preprocessing is neither 'True' nor 'False'."
    args.data_preprocessing = args.data_preprocessing=='True'
    args.data_dir, args.save_dir = train_save_dir_setting(args)
    args.logger = logger(os.path.join(args.save_dir, 'training.log'))
    args.logger(f'Model training save directory is:\n  {args.save_dir}')
    
    # 1. Training data preprocessing
    train_data_preprocessing(args)
    exit()

    """
    # 2. Retrosynthetic analysis
    if args.path:
        from scripts.retroAnalysis.retro_path_analysis import retrosyntheticAnalyzer
    else:
        from scripts.retroAnalysis.retro_analysis import retrosyntheticAnalyzer






    # 0. Reading config and directory setting
    data_directory = os.path.normpath(args.data_dir)
    save_directory = os.path.normpath(args.save_dir)
    save_name = os.path.normpath(args.save_name)
    save_directory = f'{save_directory}/{save_name}'
    if not os.path.isdir(save_directory):
        os.mkdir(save_directory)
    assert args.data_preprocessing=='True' or args.data_preprocessing=='False', \
            'args.data_preprocessing is neither True nor False.'
    args.data_preprocessing=args.data_preprocessing=='True'
    
    # 2. input data generation
    # raio = (train:val:test)
    
    # 3. model train
    GAT_model_train(data_dir=input_data_path, save_dir=save_directory, args=args)
    """

# main operation:
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type = str, help = 'path to retro analysis result directory')
    parser.add_argument('--model_save_name', type = str, help = "model's name to be saved as")
    parser.add_argument("--data_preprocessing_name", type = str, default='training_data', help = "name of preprocessed data.")       #TODO: move this part below
    # Default setting
    parser.add_argument('--num_data', type = str, default=250000, help = 'number of data used in train/val/test.')
    parser.add_argument('--num_cores', type = str, default=4, help = 'number of cores')
    parser.add_argument('--data_preprocessing', type = str, default='True', help = "True for data generation and training, False for only training")
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

