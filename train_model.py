import argparse
import os
from datetime import datetime

from scripts.modelScripts.preprocessing import train_data_preprocess
from scripts.modelScripts.train import train_DFRscore
from scripts.utils import get_cuda_visible_devices, logger, train_save_dir_setting, set_random_seed


def main_train(args):
    # 0. Reading config and directory setting
    args.root = os.path.dirname(os.path.realpath(__file__))
    args.data_dir, args.save_dir = train_save_dir_setting(args)
    args.logger = logger(os.path.join(args.save_dir, "training.log"))
    args.logger(f"Model training save directory is:\n  {args.save_dir}")

    # Set notion logging
    if args.database_id != "":
        args.logger.set_notion_logging(
            database_id=args.database_id,
            database_props={
                "Model Name": {"title": {}},
                "n_conv_layer": {"number": {}},
                "n_fc_layer": {"number": {}},
                "conv_dim": {"number": {}},
                "fc_dim": {"number": {}},
                "best_epoch": {"number": {}},
                "best_loss": {"number": {}},
            },
        )

    # Set random random seed
    if args.random_seed != -1:
        set_random_seed(seed=args.random_seed)

    # 1. Training data preprocessing
    now = datetime.now()
    since_inform = now.strftime("%Y. %m. %d (%a) %H:%M:%S")
    preprocess_dir = os.path.join(args.data_dir, args.data_preprocessing)
    args.preprocess_dir = preprocess_dir
    if os.path.exists(preprocess_dir):
        print("1. Data preprocessing phase")
        print("  Processed data already exists.")
        print("  Training data preprocessing finished.")
        args.data_dir = preprocess_dir
    else:
        os.mkdir(preprocess_dir)
        args.preprocess_logger = logger(
            os.path.join(preprocess_dir, "preprocessing.log")
        )
        args.preprocess_logger("1. Data preprocessing phase")
        args.preprocess_logger(f"  Started at: {since_inform}")
        args.preprocess_logger(f"  Data will be generated in: {preprocess_dir}")
        args.data_dir = train_data_preprocess(args=args)

    # 2. model train
    os.environ["CUDA_VISIBLE_DEVICES"] = get_cuda_visible_devices(1)
    best_epoch, best_loss = train_DFRscore(args=args)
    if args.database_id != "":
        args.logger.notion_logging(
            new_data={
                "Model Name": {"title": [{"text": {"content": args.save_dir.split("/")[-1]}}]},
                "n_conv_layer": {"number": args.n_conv_layer},
                "n_fc_layer": {"number": args.n_fc_layer},
                "conv_dim": {"number": args.conv_dim},
                "fc_dim": {"number": args.fc_dim},
                "best_epoch": {"number": best_epoch},
                "best_loss": {"number": best_loss},
            }
        )



# main operation:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="path to retro analysis result directory"
    )
    parser.add_argument("--save_name", type=str, help="model's name to be saved as")
    parser.add_argument(
        "--num_data", type=int, help="number of data used in train/val/test."
    )
    # Default setting
    parser.add_argument(
        "--data_preprocessing",
        type=str,
        default="processed_data",
        help="name of preprocessed data.",
    )
    parser.add_argument("--num_cores", type=str, default=4, help="number of cores")
    parser.add_argument(
        "--max_step", type=int, default=4, help="the maximum number of reaction steps"
    )
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--num_threads", type=int, default=4, help="the number of threads"
    )
    parser.add_argument(
        "--feature_size", type=int, default=36, help="dim of atomic feature"
    )
    parser.add_argument(
        "--n_conv_layer", type=int, default=5, help="number of convolution layers"
    )
    parser.add_argument(
        "--n_fc_layer", type=int, default=4, help="number of fully connected layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="number of heads for multi-head attention",
    )
    parser.add_argument(
        "--conv_dim", type=int, default=256, help="graph conv layer hidden dimension"
    )
    parser.add_argument(
        "--fc_dim", type=int, default=128, help="fc layer hidden dimension"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout for edge drop and normal dropout",
    )
    parser.add_argument("--num_epoch", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=4e-4, help="learning rate")
    parser.add_argument(
        "--use_scratch", action="store_true", help="use scratch data or not"
    )
    # For ReduceLROnPlateau
    parser.add_argument("--factor", type=float, default=0.5, help="decreasing factor")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="number of epochs with no improvement after which learning rate will be reduced",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Threshold for measuring the new optimum, to only focus on significant changes",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-7,
        help="A lower bound on the learning rate of all param groups",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=-1,
        help="Random seed for the model training",
    )
    parser.add_argument(
        "--database_id",
        type=str,
        default="",
        help="The notion database id to be used for loss logging",
    )
    args = parser.parse_args()
    main_train(args)
