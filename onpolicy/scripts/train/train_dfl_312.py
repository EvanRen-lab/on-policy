import sys
import os
import torch
import numpy as np
from pathlib import Path
import wandb

# ================= Automatic path configuration =================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = str(Path(current_dir).parent.parent.parent) 

if root_dir not in sys.path:
    sys.path.append(root_dir)
# ================================================================

from onpolicy.config import get_config
from onpolicy.config_dfl import get_dfl_config
from onpolicy.envs.dfl_v2.dfl_env_312 import DFLEnv 
from onpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = DFLEnv(all_args)
            return env
        return init_env
    
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = DFLEnv(all_args)
            return env
        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])

def parse_args(args, parser):
    # Load DFL and MAPPO configuration
    parser = get_dfl_config(parser)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    
    # Force set / correct key parameters
    all_args.env_name = "DFL"
    all_args.algorithm_name = "mappo"
    
    
    # Device selection
    if all_args.cuda and torch.cuda.is_available():
        print(">>> Device: Using GPU")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        print(">>> Device: Using CPU")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Set save path
    run_dir = Path(root_dir + "/onpolicy/scripts/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    all_args.run_dir = str(run_dir)

    # Initialize parallel environments
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "num_agents": all_args.num_agents,
        "run_dir": str(run_dir)
    }

    from onpolicy.runner.shared.dfl_runner_v2 import DFLRunner as Runner
    runner = Runner(config)
    
    runner.run()
    
    envs.close()
    if all_args.use_eval:
        eval_envs.close()

if __name__ == "__main__":
    main(sys.argv[1:])