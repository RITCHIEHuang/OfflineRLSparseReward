import argparse

from loguru import logger

from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.data import d4rl

from datasets import d4rl_dataset
from datasets.traj_dataset import load_decomposed_d4rl_buffer


from config import algo_select
from utils.d4rl_tasks import task_list
from utils.io_util import proj_path


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("D4rl trainer")
    parser.add_argument(
        "--algo_name", help="algorithm", type=str, default="cql"
    )
    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument(
        "--delay_mode",
        help="delay mode",
        type=str,
        default="constant",
        choices=["constant", "random", "none"],
    )
    parser.add_argument(
        "--delay", help="constant delay steps", type=int, default=20
    )
    parser.add_argument(
        "--shaping_method",
        help="custom shaping methods, default is transformer",
        type=str,
        default="transformer",
    )
    parser.add_argument(
        "--name", help="experiment name", type=str, default="exp_name_is_none"
    )
    parser.add_argument(
        "--delay_min", help="min delay steps", type=int, default=10
    )
    parser.add_argument(
        "--delay_max", help="max delay steps", type=int, default=50
    )
    parser.add_argument(
        "--task",
        help="task name",
        type=str,
        default="walker2d-expert-v0",
        choices=task_list,
    )

    return parser.parse_args()


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)

    if algo_config["delay_mode"] == "none":
        train_buffer = d4rl.load_d4rl_buffer(algo_config["task"])
    else:
        train_buffer = load_decomposed_d4rl_buffer(algo_config)

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = OnlineCallBackFunction()
    callback.initialize(
        train_buffer=train_buffer, val_buffer=None, task=algo_config["task"]
    )
    callback_list = CallBackFunctionList(
        [callback, d4rl_eval_fn(task=algo_config["task"])]
    )

    algo_trainer.train(train_buffer, None, callback_fn=callback_list)


if __name__ == "__main__":
    # python train_d4rl.py --algo_name=cql --task=walker2d-expert-v0 --delay=20 --delay_mode=constant
    args = argsparser()
    args = vars(args)
    args["task"] = f"d4rl-{args['task']}"
    args["log_path"] = f"{proj_path}/logs"

    if args["delay_mode"] == "none":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-{args['algo_name']}-seed-{args['seed']}"
    if args["delay_mode"] == "constant":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-delay-{args['delay']}-{args['algo_name']}-seed-{args['seed']}"
    elif args["delay_mode"] == "random":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-delay_min-{args['delay_min']}-delay_max-{args['delay_max']}-{args['algo_name']}-seed-{args['seed']}"

    args["exp_name"] = f"{exp_name}-decomposed-{args['shaping_method']}"

    logger.info(
        f"Task: {args['task']}-delay-{args['delay']}, algo: {args['algo_name']}, exp_name: {args['exp_name']}"
    )
    import torch
    import numpy as np

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    run_algo(args)
