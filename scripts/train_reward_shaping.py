import argparse

from loguru import logger

from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.data import d4rl

from datasets import d4rl_dataset

from config import algo_select
from utils.d4rl_tasks import task_list
from utils.io_util import proj_path


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("D4rl trainer")
    parser.add_argument(
        "--algo_name", help="algorithm", type=str, default="reward_shaper"
    )
    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument(
        "--policy_mode",
        help="policy mode",
        type=str,
        default="random",
        choices=["bc", "random"],
    )
    parser.add_argument(
        "--shaping_version",
        help="shaping version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
    )
    parser.add_argument(
        "--delay_mode",
        help="delay mode",
        type=str,
        default="constant",
        choices=["constant", "random", "none"],
    )
    parser.add_argument(
        "--name", help="experiment name", type=str, default="exp_name_is_none"
    )
    parser.add_argument(
        "--delay", help="constant delay steps", type=int, default=20
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
        train_buffer = d4rl_dataset.load_d4rl_buffer(algo_config)

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


# python train_reward_shaping.py --policy_mode=bc --task=walker2d-medium-replay-v0 --delay_mode=constant --delay=20
if __name__ == "__main__":
    args = argsparser()
    args = vars(args)
    args["task"] = f"d4rl-{args['task']}"
    args["log_path"] = f"{proj_path}/logs"

    if args["delay_mode"] == "none":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}"
    if args["delay_mode"] == "constant":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-delay-{args['delay']}"
    elif args["delay_mode"] == "random":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-delay_min-{args['delay_min']}-delay_max-{args['delay_max']}"

    args[
        "exp_name"
    ] = f"{exp_name}--{args['algo_name']}-{args['policy_mode']}-{args['shaping_version']}-seed-{args['seed']}"

    logger.info(
        f"Task: {args['task']}, algo: {args['algo_name']}, exp_name: {args['exp_name']}"
    )
    import torch
    import numpy as np

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    run_algo(args)
