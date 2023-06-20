import os
import subprocess


class logger:
    def __init__(self, log_file_path):
        self.log_file = log_file_path
        try:
            with open(self.log_file, "a") as w:
                pass
        except:
            print(f"Invalid log path {log_file_path}")
            exit()
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def __call__(self, *log, save_log=True, end="\n"):
        if len(log) == 0:
            log = ("",)
        log = [str(i) for i in log]
        log = "\n".join(log)
        print(log, end=end)
        if save_log:
            self.save(log, end=end)

    @classmethod
    def get_skip_args(cls):
        return [
            "logger",
            "root",
            "save_name",
            "preprocess_dir",
            "preprocess_logger",
            "data_preprocessing",
            "data_dir",
            "save_dir",
        ]

    def save(self, log, end):
        with open(self.log_file, "a") as w:
            w.write(log + end)

    def log_arguments(self, args):
        d = vars(args)
        _skip_args = self.get_skip_args()
        for v in d:
            if not v in _skip_args:
                self(f"  {v}: {d[v]}")

    def set_notion_logging(self, database_id: str, database_props: dict):
        import dotenv
        from notion_client import Client

        # from datetime import datetime

        config = dotenv.dotenv_values(".env")
        notion_secret = config.get("NOTION_TOKEN")
        self.client = Client(auth=notion_secret)
        self.database_id = database_id

        self.client.databases.update(
            database_id=database_id, properties=database_props
        )
        return

    def notion_logging(self, new_data: dict):
        # Add data to the database with Notion API
        self.client.pages.create(parent={"database_id": self.database_id}, properties=new_data)
        return

def set_random_seed(seed:int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def retro_save_dir_setting(root, args):
    target_data_name = args.retro_target.split("/")[-1].split(".smi")[0]
    if not os.path.exists(os.path.join(root, args.save_name, target_data_name)):
        os.mkdir(os.path.join(root, args.save_name, target_data_name))

    save_dir = os.path.join(root, args.save_name, target_data_name, "retro_result")
    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(save_dir + str(i)):
            i += 1
        save_dir = save_dir + str(i)
    os.mkdir(save_dir)
    return save_dir


def train_save_dir_setting(args):
    data_dir = os.path.normpath(args.data_dir)
    retro_data_dir = "/".join(data_dir.split("/")[:-1])
    save_dir = os.path.normpath(os.path.join(retro_data_dir, args.save_name))

    if "/" in args.save_name:
        out_dir, inner_dir = args.save_name.split("/")
        out_dir = os.path.join(retro_data_dir, out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    if os.path.exists(save_dir):
        i = 2
        while os.path.exists(f"{save_dir}_{i}"):
            i += 1
        save_dir = f"{save_dir}_{i}"
    os.mkdir(save_dir)
    return data_dir, save_dir


def get_cuda_visible_devices(num_gpus: int) -> str:
    """Get available GPU IDs as a str (e.g., '0,1,2')"""
    max_num_gpus = 4
    idle_gpus = []

    if num_gpus:
        for i in range(max_num_gpus):
            cmd = ["nvidia-smi", "-i", str(i)]

            import sys

            major, minor = sys.version_info[0], sys.version_info[1]
            if major == 3 and minor > 6:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True
                )  # after python 3.7
            if major == 3 and minor <= 6:
                proc = subprocess.run(
                    cmd, stdout=subprocess.PIPE, universal_newlines=True
                )  # for python 3.6

            if "No devices were found" in proc.stdout:
                break

            if "No running" in proc.stdout:
                idle_gpus.append(i)

            if len(idle_gpus) >= num_gpus:
                break

        if len(idle_gpus) < num_gpus:
            msg = "Avaliable GPUs are less than required!"
            msg += f" ({num_gpus} required, {len(idle_gpus)} available)"
            raise RuntimeError(msg)

        # Convert to a str to feed to os.environ.
        idle_gpus = ",".join(str(i) for i in idle_gpus[:num_gpus])

    else:
        idle_gpus = ""

    return idle_gpus


if __name__ == "__main__":
    pass
