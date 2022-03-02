import argparse
import os
import os.path as op
import sys, shutil
from scripts.utils import logger 
from scripts.utils import save_dir_setting

def retrosynthetic_analysis(args):
    # 0. Reading config and directory setting
    if not op.isdir(args.save_name):
        os.mkdir(args.save_name)
    project_path = os.getcwd()

    args.save_dir = save_dir_setting(project_path, args)
    args.logger = logger(os.path.join(args.save_dir, 'generation_log.txt'))
    args.logger(f'Retro save directory is:\n  {args.save_dir}')
    
    # 1. Preprocessing reactant set
    from scripts.R_set_generator import R_set_generator
    R_set_generator(project_path, args)

    # 2. retrosynthetic analysis
    if args.path:
        from scripts.retro_path_analysis import retrosyntheticAnalyzer
    else:
        from scripts.retro_analysis import retrosyntheticAnalyzer
    retrosyntheticAnalyzer(project_path, args)
    

# main operation:
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", help='path to retrosynthetic template file', type=str)
    parser.add_argument("--using_augmented_template", help='whether using augmented template or not', default=True, type=bool)
    parser.add_argument("--reactant", help='path to reactant bag smiles file', type=str)
    parser.add_argument("--retro_target", help='path to retrosynthetic target smiles file', type=str)
    parser.add_argument("--depth", help='retrosynthetic analysis max depth', type=int)
    parser.add_argument("--num_molecules", help='number of molecules to be analyzed', type=int)
    parser.add_argument("--num_cores", help='number of cores used for multiprocessing', type=int)
    parser.add_argument("--save_name", help='directory to save the results', default='save', type=str)
    parser.add_argument("--path", help='whether restoring synthesis path or not. if Ture, more time required.', default=False, type=bool)
    parser.add_argument("--exclude_in_R_bag", help='whether excluding molecules in retro_target file included in R bag', default=True, type=bool)
    parser.add_argument("--batch_size", help='batch size for retrosynthetic analysis', default=10000, type=int)
    args = parser.parse_args()
    retrosynthetic_analysis(args)
