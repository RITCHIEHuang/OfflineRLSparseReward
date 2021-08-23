import argparse
import random

from loguru import logger
from ray import tune

from offlinerl.algo import algo_select
from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.data import d4rl

from datasets import d4rl_dataset

from utils.d4rl_tasks import task_list


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("D4rl Tune Trainer")
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


def training_function(config):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(config["kwargs"])
    if algo_config["delay_mode"] == "none":
        train_buffer = d4rl.load_d4rl_buffer(algo_config["task"])
    else:
        train_buffer = d4rl_dataset.load_d4rl_buffer(algo_config)

    algo_config.update(config)
    algo_config["device"] = "cuda"
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = OnlineCallBackFunction()
    callback.initialize(
        train_buffer=train_buffer, val_buffer=None, task=algo_config["task"]
    )
    callback_list = CallBackFunctionList(
        [callback, d4rl_eval_fn(task=algo_config["task"])]
    )

    score = algo_trainer.train(train_buffer, None, callback_fn=callback_list)

    return score


def run_algo(kwargs):
    config = {}
    config["kwargs"] = kwargs
    # config["kwargs"]["seed"] = random.randint(0, 1000000)
    config["kwargs"][
        "exp_name"
    ] = f"{config['kwargs']['exp_name']}-seed-{kwargs['seed']}"
    logger.info(
        f"Task: {kwargs['task']}, algo: {kwargs['algo_name']}, exp_name: {kwargs['exp_name']}"
    )
    _, _, algo_config = algo_select(kwargs)
    # Prepare Dataset
    if algo_config["delay_mode"] == "none":
        d4rl.load_d4rl_buffer(algo_config["task"])
    else:
        d4rl_dataset.load_d4rl_buffer(algo_config)

    grid_tune = algo_config["grid_tune"]
    for k, v in grid_tune.items():
        config[k] = tune.grid_search(v)

    analysis = tune.run(
        training_function,
        config=config,
        local_dir=f"../tune_results",
        resources_per_trial={"gpu": 1, "cpu": 8},
        queue_trials=True,
    )


# python train_d4rl_tune.py --algo_name=cql --task=walker2d-medium-replay-v0 -delay_mode=constant --delay=20
if __name__ == "__main__":
    args = argsparser()
    args = vars(args)
    args["task"] = f"d4rl-{args['task']}"

    if args["delay_mode"] == "none":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-{args['algo_name']}"
    if args["delay_mode"] == "constant":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-delay-{args['delay']}-{args['algo_name']}"
    elif args["delay_mode"] == "random":
        exp_name = f"{args['task']}-delay_mode-{args['delay_mode']}-delay_min-{args['delay_min']}-delay_max-{args['delay_max']}-{args['algo_name']}"

    args["exp_name"] = exp_name

    run_algo(args)
