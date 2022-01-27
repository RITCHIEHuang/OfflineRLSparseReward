from collections import defaultdict
from copy import deepcopy

import json
import os
import csv

import wandb
import numpy as np
from tqdm import tqdm
from loguru import logger

from utils.io_util import proj_path
from utils.task_util import get_domain_by_task


debug = False

agg_result_file_path = f"{proj_path}/assets/agg_results_dt_mujoco_v0.csv"

if os.path.exists(agg_result_file_path):
    os.remove(agg_result_file_path)


filter_model_types = ["dt"]

api = wandb.Api()
runs = api.runs("ritchiehuang/decision-transformer")

exp_variant_mapping = defaultdict(lambda: defaultdict(list))
# {"group": [{iter-0: [{seed1}, {seed2}, ...{}], }, {iter-1}, ..., {iter-}]}


for run in tqdm(runs):
    group = run.group
    env = run.config["env"]
    delay_mode = run.config.get("delay_mode", "constant")
    delay = run.config["delay"]
    seed = run.config.get("seed", 0)
    dataset_type = run.config["dataset"]
    model_type = run.config["model_type"]
    # domain = get_domain_by_task(f"{env}-{dataset_type}")
    # if domain != "neorl":
    #     domain = "d4rl"

    if model_type not in filter_model_types:
        continue

    variant_result_info = {
        "Dataset Type": dataset_type,
        "Environment": env,
        "Delay Mode": delay_mode,
        "Delay": delay,
        "Strategy": model_type,
        "Algo": model_type.upper(),
        "Seed": seed,
    }

    history_dict = run.scan_history()
    for history_item in history_dict:
        epoch = history_item["_step"]
        result_info_item = deepcopy(variant_result_info)
        result_info_item["Iteration"] = epoch

        d4rl_score_mean_list = []
        for k in history_item:
            if "d4rl_score_mean" in k:
                d4rl_score_mean_list.append(history_item[k])
        result_info_item["D4rl_Score"] = max(d4rl_score_mean_list)

        exp_variant_mapping[group][epoch].append(result_info_item)

if debug:
    with open("test.json", "w") as f:
        json.dump(exp_variant_mapping, f)


# aggregate
flag = True
for k, v in exp_variant_mapping.items():
    iter_scores = [
        (
            i_iter,
            np.mean([it["D4rl_Score"] for it in iter_res]),
            [it["Seed"] for it in iter_res],
            np.std([it["D4rl_Score"] for it in iter_res]),
        )
        for i_iter, iter_res in v.items()
        if len(iter_res) >= 3
    ]

    try:
        sorted_iter_scores = sorted(
            iter_scores, key=lambda v: v[1], reverse=True
        )
        if debug:
            logger.debug(f"{k}, {len(iter_scores)}")

        selected_item = sorted_iter_scores[0]
        agg_item = deepcopy(v[selected_item[0]][0])
        agg_item["Iteration"] = selected_item[0]
        agg_item["D4rl_Score"] = round(selected_item[1], 1)
        agg_item["Seed"] = selected_item[2]
        agg_item["D4rl_Score_stddev"] = round(selected_item[3], 1)

        logger.info(f"{k} aggregating results finish!")

        agg_fields = list(agg_item.keys())
        with open(agg_result_file_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fields)

            if flag:
                writer.writeheader()
                flag = False

            writer.writerow(agg_item)

    except Exception:
        print("=" * 100)
        print(k)
        print("=" * 100)
