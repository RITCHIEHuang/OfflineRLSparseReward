import argparse

from datetime import datetime

from loguru import logger

from utils.io_util import proj_path
from utils.task_util import get_domain_by_task


""" Experiment setting """


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



def argsparser():
    parser = argparse.ArgumentParser("D4rl trainer")
    parser.add_argument(
        "--project", type=str, default="OfflineRL_DelayRewards"
    )
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
    parser.add_argument("--name", help="experiment name", type=str, default="")
    parser.add_argument("--model_path", help="model path", type=str, default="")
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
        "--reward_scale", help="scale for reward", type=float, default=1.0
    )
    parser.add_argument(
        "--reward_shift", help="shift for reward", type=float, default=0.0
    )
    parser.add_argument(
        "--use_noisy_linear",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Activate nice mode.",
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
        #     "transformer_decompose",
        #     "pg_reshaping",
        # ],
    )
    parser.add_argument(
        "--task",
        help="task name",
        type=str,
        default="halfcheetah-medium-expert-v0",
    )
    parser.add_argument("--log_to_wandb", type=bool, default=True)

    return parser.parse_args()


def setup_exp_args():
    args = argsparser()
    args = vars(args)

    domain = get_domain_by_task(args["task"])
    if domain == "neorl":
        split_list = args["task"].split("-")
        args["task_name"] = "-".join(split_list[:2])
        args["task_data_type"] = split_list[2]
        args["task_train_num"] = int(split_list[3])

    logger.info(f"Task: {args['task']} in Domain: {domain} !!!")

    args["task"] = f"{domain}-{args['task']}"
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
    args[
        "exp_name"
    ] = f"{args['exp_name']}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')[:-3]}"

    return args
