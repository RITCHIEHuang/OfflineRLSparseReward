import argparse

from loguru import logger

from offlinerl.algo import algo_select
from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.data import d4rl

from datasets import d4rl_dataset

from utils.d4rl_tasks import task_list


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("D4rl trainer")
    parser.add_argument(
        "--algo_name", help="algorithm", type=str, default="cql"
    )
    parser.add_argument("--delay", help="delay steps", type=int, default=0)
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

    if algo_config["delay"] == 0:
        train_buffer = d4rl.load_d4rl_buffer(algo_config["task"])
    else:
        train_buffer = d4rl_dataset.load_d4rl_buffer(algo_config)

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    callback = OnlineCallBackFunction()
    callback.initialize(
        train_buffer=train_buffer, val_buffer=None, task=algo_config["task"]
    )
    callback_list = CallBackFunctionList([callback, d4rl_eval_fn(task=algo_config["task"])])

    algo_trainer.train(train_buffer, None, callback_fn=callback_list)


if __name__ == "__main__":
    # python train_d4rl.py --algo_name=cql --task=halfcheetah-medium-v0 --delay=20
    args = argsparser()
    args = vars(args)
    args["task"] = f"d4rl-{args['task']}"
    args[
        "exp_name"
    ] = f"{args['task']}-delay-{args['delay']}-{args['algo_name']}"

    logger.info(
        f"Task: {args['task']}-delay-{args['delay']}, algo: {args['algo_name']}, exp_name: {args['exp_name']}"
    )
    run_algo(args)
