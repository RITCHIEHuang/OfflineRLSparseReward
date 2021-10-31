from collections import defaultdict
from copy import deepcopy
import os
import json
import re
import csv

import numpy as np
from tqdm import tqdm
from loguru import logger

from utils.io_util import proj_path

log_path = f"{proj_path}/logs"

result_file_path = f"{proj_path}/results.csv"
agg_result_file_path = f"{proj_path}/agg_results.csv"

if os.path.exists(result_file_path):
    os.remove(result_file_path)
if os.path.exists(agg_result_file_path):
    os.remove(agg_result_file_path)

# max_iteration for experiment
experiment_iter_mapping = {
    "halfcheetah-meidum-v0-delay_mode-constant-delay-50-mopo-strategy-interval_average": 1500
}


def fetch_experiment_results():
    exp_logs_list = os.listdir(log_path)
    exp_variant_mapping = defaultdict(list)
    flag = True
    flag2 = True
    for exp_log_dir in tqdm(exp_logs_list):
        if exp_log_dir == ".aim":
            continue
        exp_hparam_file = f"{log_path}/{exp_log_dir}/hparams.json"
        exp_metric_file = f"{log_path}/{exp_log_dir}/metric_logs.json"

        if not os.path.exists(exp_hparam_file) or not os.path.exists(
            exp_metric_file
        ):
            continue

        with open(exp_hparam_file, "r") as f:
            exp_hparam = json.load(f)
            logger.info(f"Load exp hparam: {exp_hparam}")
            task = exp_hparam["task"]
            if task.startswith("d4rl"):
                task = task[5:]

            exp_name = exp_hparam["exp_name"]
            strategy = "none"
            if "strategy" in exp_hparam:
                strategy = exp_hparam["strategy"]
            elif "shaping_method" in exp_hparam:
                strategy = exp_hparam["shaping_method"]

        iteration = None
        if exp_name in experiment_iter_mapping:
            iteration = experiment_iter_mapping[exp_name]

        with open(exp_metric_file, "rb") as f:
            exp_metric = json.load(f)
            d4rl_score = -float("-inf")
            if str(iteration) in exp_metric:
                exp_metric = exp_metric[str(iteration)]
            else:
                # default the last iteration
                exp_metric = exp_metric[sorted(exp_metric.keys())[-1]]
            d4rl_score = exp_metric["D4rl_Score"]

        task_split_list = task.split("-")
        print(task_split_list)
        variant_result_info = {
            # "exp_log_dir": exp_log_dir,
            # "task": task,
            # "exp_name": exp_name,
            "Dataset Type": "-".join(task_split_list[1:-1]),
            "Environment": task_split_list[0],
            "Delay Mode": exp_hparam["delay_mode"],
            "Delay": exp_hparam["delay"],
            "Algo": exp_hparam["algo_name"],
            "Strategy": strategy,
            "Seed": exp_hparam["seed"],
        }

        result_info = deepcopy(variant_result_info)
        result_info.update({"D4rl_Score": d4rl_score})

        identity_tag = f"{variant_result_info['Environment']}-{variant_result_info['Dataset Type']}-{variant_result_info['Delay Mode']}-{variant_result_info['Delay']}-{variant_result_info['Algo']}-{variant_result_info['Strategy']}"
        exp_variant_mapping[identity_tag].append(result_info)

        fields = list(result_info.keys())
        with open(result_file_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=fields)

            if flag:
                writer.writeheader()
                flag = False

            writer.writerow(result_info)

    for k, v in exp_variant_mapping.items():
        agg_item = deepcopy(v[0])
        agg_item["D4rl_Score"] = np.mean([item["D4rl_Score"] for item in v])

        del agg_item["Seed"]

        agg_fields = list(agg_item.keys())
        with open(agg_result_file_path, "a+") as f:
            writer = csv.DictWriter(f, fieldnames=agg_fields)

            if flag2:
                writer.writeheader()
                flag2 = False

            writer.writerow(agg_item)

    return exp_variant_mapping


fetch_experiment_results()
# log_path = f"{log_path}/.aim"
# for a in os.listdir(log_path):
#     if a.startswith("d4rl"):
#         seed_idx = a.index("seed")
#         stra_idx = a.index("strategy")
#         if seed_idx < stra_idx:
#             new_dir_name = (
#                 a[:seed_idx]
#                 + a[stra_idx : a.rindex("2021") - 1]
#                 + "-"
#                 + a[seed_idx : stra_idx - 1]
#                 + "_"
#                 + a[a.rindex("2021") :]
#             )
#             os.rename(f"{log_path}/{a}", f"{log_path}/{new_dir_name}")
