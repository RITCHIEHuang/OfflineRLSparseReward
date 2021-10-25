import argparse

from loguru import logger

from utils.io_util import proj_path
from utils.d4rl_tasks import task_list


""" Experiment setting """


def argsparser():
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
    parser.add_argument("--bc_epoch", help="bc epochs", type=int, default=0)
    parser.add_argument(
        "--strategy",
        help="delay rewards strategy, can be multiple strategies seperated by  `,`",
        type=str,
        default="none",
        # choices=[
        #     "none",
        #     "scale",
        #     "scale_v2",
        #     "scale_minmax",
        #     "scale_zscore",
        #     "episodic_average",
        #     "episodic_random",
        #     "episodic_ensemble",
        #     "interval_average",
        #     "interval_random",
        #     "interval_ensemble",
        #     "minmax",
        #     "zscore",
        #     "transformer_decompose",
        #     "pg_reshaping",
        # ],
    )
    parser.add_argument(
        "--task",
        help="task name",
        type=str,
        default="walker2d-expert-v0",
        choices=task_list,
    )

    return parser.parse_args()


def setup_exp_args():
    args = argsparser()
    args = vars(args)
    args["task"] = f"d4rl-{args['task']}"
    args["log_path"] = f"{proj_path}/logs"

    delay_tag = ""
    if args["delay_mode"] == "none":
        delay_tag = f"delay_mode-{args['delay_mode']}"
    elif args["delay_mode"] == "constant":
        delay_tag = f"delay_mode-{args['delay_mode']}-delay-{args['delay']}"
    elif args["delay_mode"] == "random":
        delay_tag = f"delay_mode-{args['delay_mode']}-delay_min-{args['delay_min']}-delay_max-{args['delay_max']}"
    else:
        raise NotImplementedError()

    args["delay_tag"] = delay_tag
    args[
        "exp_name"
    ] = f"{args['task']}-{delay_tag}-{args['algo_name']}-strategy-{args['strategy']}"
    if args["bc_epoch"] != 0:
        args["exp_name"] = f"{args['exp_name']}-bc-{args['bc_epoch']}"
    args["exp_name"] = f"{args['exp_name']}-seed-{args['seed']}"

    logger.info(f"Experiment name: {args['exp_name']}")
    logger.info(
        f"Task: {args['task']}, algo: {args['algo_name']}, delay_tag: {args['delay_tag']}"
    )
    return args
