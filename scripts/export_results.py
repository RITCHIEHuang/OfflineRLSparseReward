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

result_file_path = f"{proj_path}/assets/Delay_results.csv"
agg_result_file_path = f"{proj_path}/assets/Delay_agg_results.csv"

if os.path.exists(result_file_path):
    os.remove(result_file_path)

if os.path.exists(agg_result_file_path):
    os.remove(agg_result_file_path)

api = wandb.Api()
runs = api.runs("ritchiehuang/OfflineRL_DelayRewards")

exp_variant_mapping = defaultdict(lambda: defaultdict(list))
# {"group": [iter-0: [{seed1}, {seed2}, ...{}], }, {iter-1}, ..., {iter-}]}

filter_group = ["d4rl-halfcheetah-medium-replay-v0"]
filter_strategy = ["interval_ensemble", "interval_average", "none"]
# filter_strategy = ["interval_ensemble"]
# filter_strategy = ["none"]
filter_seed = [10, 100, 1000]

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
    elif domain == "d4rl":
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

    if not (
        group in filter_group
        and strategy in filter_strategy
        and seed in filter_seed
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

        exp_variant_mapping[exp_identity_tag][epoch].append(result_info_item)

if debug:
    with open("test.json", "w") as f:
        json.dump(exp_variant_mapping, f)

# aggregate
flag = True
flag2 = True
for k, v in exp_variant_mapping.items():
    iter_scores = [
        (
            i_iter,
            np.mean([it["D4rl_Score"] for it in iter_res]),
            [it["Seed"] for it in iter_res],
            np.std([it["D4rl_Score"] for it in iter_res]),
            [it["D4rl_Score"] for it in iter_res],
        )
        for i_iter, iter_res in v.items()
        if len(iter_res) >= 3
    ]

    sorted_iter_scores = sorted(iter_scores, key=lambda v: v[1], reverse=True)
    if debug:
        logger.debug(f"{k}, {len(iter_scores)}")

    selected_item = sorted_iter_scores[0]

    normal_item = deepcopy(v[selected_item[0]][0])
    for seed, score in zip(selected_item[2], selected_item[-1]):
        # normal_item["Iteration"] = selected_item[0]
        normal_item["D4rl_Score"] = round(score, 1)
        normal_item["Seed"] = seed

        fields = list(normal_item.keys())
        with open(result_file_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            if flag:
                writer.writeheader()
                flag = False

            writer.writerow(normal_item)

    agg_item = deepcopy(v[selected_item[0]][0])
    agg_item["Iteration"] = selected_item[0]
    agg_item["D4rl_Score"] = round(selected_item[1], 1)
    agg_item["Seed"] = selected_item[2]
    agg_item["D4rl_Score_stddev"] = round(selected_item[3], 1)

    logger.info(f"{k} aggregating results finish!")

    agg_fields = list(agg_item.keys())
    with open(agg_result_file_path, "a+") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)

        if flag2:
            writer.writeheader()
            flag2 = False

        writer.writerow(agg_item)
