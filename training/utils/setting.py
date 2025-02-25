import json
import logging
import os

import wandb
from utils.distributed import get_rank, is_main_process
import warnings


def set_system(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

    if "NinjaPath" in config:
        os.environ["PATH"] = config["NinjaPath"] + ":" + os.environ["PATH"]

    if "Environment" in config:
        for key, value in config["Environment"].items():
            os.environ[key] = value
    
    return None


def set_distributed_logging(strict: bool = False):
    ''' 
        In default, only the main process will log INFO level
        Currently this function only controls logs from logging and warnings modules
        Some others libraries implemented their own logging system:
            - Deepspeed: implemented yellow color warnings
        But still need to be used carefully since some important logs might be missed
    '''
    rank = get_rank()
    if is_main_process():
        print(f"Rank {rank}: Setting logging level to INFO")
        logging.basicConfig(level=logging.INFO)
    else:
        if strict:
            print(f"Rank {rank}: Setting logging level to ERROR")
            logging.basicConfig(level=logging.ERROR)
            warnings.filterwarnings("ignore")
        else:
            print(f"Rank {rank}: Setting logging level to WARNING")
            logging.basicConfig(level=logging.WARNING)
        

def set_args(args):
    if args.cache_dir is not None:
        # User has specified a cache directory
        pass
    else:
        # System setted cache directory
        if "HF_HUB_CACHE" in os.environ:
            args.cache_dir = os.environ["HF_HUB_CACHE"]
        # Use HF default cache directory
        else:
            args.cache_dir = None

    return None

def set_project(args):
    with open("src/configs/project_config.json", "r") as f:
        project_config = json.load(f)
    
    if "WANDB_PROJECT" in project_config:
        os.environ["WANDB_PROJECT"] = project_config["WANDB_PROJECT"]
    if "WANDB_ENTITY" in project_config:
        os.environ["WANDB_ENTITY"] = project_config["WANDB_ENTITY"]

    # Detect if file exists
    keys_file = "src/configs/keys.json"
    if os.path.exists(keys_file):
        with open(keys_file, "r") as f:
            keys = json.load(f)
        
        if "WANDB_KEY" in keys:
            wandb.login(key=keys["WANDB_KEY"])

