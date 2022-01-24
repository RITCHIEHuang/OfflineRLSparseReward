from collections import defaultdict
from copy import deepcopy

import os
import csv
import json

import wandb
import numpy as np
from tqdm import tqdm
from loguru import logger

from utils.io_util import proj_path


debug = False

result_file_path = f"{proj_path}/assets/iql/d4rl"

if not os.path.exists(result_file_path):
    os.makedirs(result_file_path)


api = wandb.Api()
runs = api.runs("ritchiehuang/OfflineRL_DelayRewards")

exp_variant_mapping = defaultdict(list)
# {"group": [{}, {}]}

# filter_group = ["d4rl-halfcheetah-medium-replay-v0"]
filter_group = [
    "d4rl-antmaze-umaze-v2",
    "d4rl-antmaze-umaze-diverse-v2",
    "d4rl-antmaze-medium-play-v2",
    "d4rl-antmaze-medium-diverse-v2",
    "d4rl-antmaze-large-play-v2",
    "d4rl-antmaze-large-diverse-v2",
]
# filter_group = None

filter_strategy = ["interval_ensemble", "interval_average", "none"]
# filter_strategy = ["interval_ensemble"]
# filter_strategy = ["none"]

filter_delaymode = ["constant"]
# filter_delaymode = ["none"]

filter_delay = [20]
filter_seed = [10, 100, 1000]

# filter_domain = ["neorl", "d4rl"]
filter_domain = ["d4rl"]

# filter_algo = ["mopo"]
filter_algo = ["iql"]

# collect
for run in tqdm(runs):
    group = run.group
    task = run.config["task"]

    split_list = task.split("-")
    domain = split_list[0]
    if domain == "neorl":
        # [neorl, HalfCheetah, v3, low, 100]
        env = split_list[1].lower()
        dataset_type = "-".join(split_list[-2:])
    elif domain in ["d4rl", "recs"]:
        # [d4rl, walker2d, medium, replay, v0]
        env = split_list[1]
        dataset_type = "-".join(split_list[2:])
    else:
        raise NotImplementedError()

    delay_mode = run.config.get("delay_mode", "constant")
    delay = run.config["delay"]
    seed = run.config.get("seed", 0)

    algo = run.config["algo_name"]
    strategy = run.config["strategy"]

    if group in filter_group:
        continue

    if not (
        (filter_domain is None or domain in filter_domain)
        and (filter_delaymode is None or delay_mode in filter_delaymode)
        # and group in filter_group
        and algo in filter_algo
        and strategy in filter_strategy
        and seed in filter_seed
        and delay in filter_delay
    ):
        continue

    variant_result_info = {
        "Dataset Type": dataset_type,
        "Environment": env,
        "Delay Mode": delay_mode,
        "Delay": delay,
        "Strategy": strategy,
        "Algo": algo,
        "Seed": seed,
    }

    history_dict = run.scan_history()
    exp_identity_tag = f"{variant_result_info['Environment']}-{variant_result_info['Dataset Type']}-{variant_result_info['Delay Mode']}-{variant_result_info['Delay']}-{variant_result_info['Algo']}-{variant_result_info['Strategy']}"

    for history_item in history_dict:
        epoch = history_item["_step"]
        if "D4rl_Score" not in history_item:
            continue
        result_info_item = deepcopy(variant_result_info)
        result_info_item["Iteration"] = epoch
        result_info_item["D4rl_Score"] = history_item["D4rl_Score"]

        exp_variant_mapping[exp_identity_tag].append(result_info_item)

if debug:
    with open("test.json", "w") as f:
        json.dump(exp_variant_mapping, f)

# aggregate
flag = True
for k, v in exp_variant_mapping.items():
    task_name = v[0]["Environment"] + "-" + v[0]["Dataset Type"] + "-strategy"
    # task_name = v[0]["Environment"] + "-" + v[0]["Dataset Type"]
    result_file_path_ = f"{result_file_path}/{task_name}.csv"
    if os.path.exists(result_file_path_):
        flag = False
    else:
        flag = True
    logger.info(f"Start writing {task_name} to {result_file_path}!")
    with open(result_file_path_, "a+") as f:
        fields = list(v[0].keys())

        writer = csv.DictWriter(f, fieldnames=fields)

        if flag:
            writer.writeheader()
            flag = False
        for row in v:
            writer.writerow(row)

    logger.info(f"Write {task_name} to {result_file_path} finished!")
