from collections import defaultdict
from copy import deepcopy
import os
import json
import csv

import numpy as np
from tqdm import tqdm
from loguru import logger

from utils.io_util import proj_path

debug = False
log_path = f"{proj_path}/logs"

result_file_path = f"{proj_path}/assets/results.csv"
agg_result_file_path = f"{proj_path}/assets/agg_results.csv"

if os.path.exists(result_file_path):
    os.remove(result_file_path)
if os.path.exists(agg_result_file_path):
    os.remove(agg_result_file_path)


def fetch_experiment_results():
    exp_logs_list = os.listdir(log_path)
    exp_variant_mapping = defaultdict(lambda: defaultdict(list))
    flag = True
    flag2 = True
    for exp_log_dir in tqdm(exp_logs_list):
        if exp_log_dir == ".aim":
            continue
        exp_log_path = f"{log_path}/{exp_log_dir}"
        exp_hparam_file = f"{exp_log_path}/hparams.json"
        exp_metric_file = f"{exp_log_path}/metric_logs.json"

        if not os.path.exists(exp_hparam_file) or not os.path.exists(
            exp_metric_file
        ):
            continue

        with open(exp_hparam_file, "r") as f:
            exp_hparam = json.load(f)
            if "delay_mode" not in exp_hparam:
                continue
            if debug:
                logger.debug(f"{exp_log_dir} Load exp hparam: {exp_hparam}")

            task = exp_hparam["task"]
            split_list = task.split("-")
            domain = split_list[0]
            if domain == "neorl":
                # [neorl, HalfCheetah, v3, low, 100]
                environment = split_list[1].lower()
                dataset_type = "-".join(split_list[-2:])
            elif domain == "d4rl":
                # [d4rl, walker2d, medium, replay, v0]
                environment = split_list[1]
                dataset_type = "-".join(split_list[2:-1])
            else:
                raise NotImplementedError()

            exp_name = exp_hparam["exp_name"]
            strategy = "none"
            if "strategy" in exp_hparam:
                strategy = exp_hparam["strategy"]
            elif "shaping_method" in exp_hparam:
                strategy = exp_hparam["shaping_method"]
            max_epoch = exp_hparam["max_epoch"]

        variant_result_info = {
            # "exp_log_dir": exp_log_dir,
            # "task": task,
            # "exp_name": exp_name,
            "Dataset Type": dataset_type,
            "Environment": environment,
            "Domain": domain,
            "Delay Mode": exp_hparam["delay_mode"],
            "Delay": exp_hparam["delay"],
            "Algo": exp_hparam["algo_name"],
            "Strategy": strategy,
            "Seed": exp_hparam["seed"],
        }

        # load metric json file
        with open(exp_metric_file, "rb") as f:
            exp_metrics = json.load(f)
            exp_max_epoch = sorted(list(map(int, exp_metrics.keys())))[-1]

            if exp_max_epoch + 1 != max_epoch:
                print("expected", max_epoch, "got", exp_max_epoch + 1)
                logger.error(f"Experiment {exp_log_path} not finished !!!")
                continue

            for k, v in exp_metrics.items():
                iteration = int(k)
                d4rl_score = v["D4rl_Score"]

                result_info_item = deepcopy(variant_result_info)
                result_info_item.update(
                    {"Iteration": iteration, "D4rl_Score": d4rl_score}
                )

                exp_identity_tag = f"{variant_result_info['Environment']}-{variant_result_info['Dataset Type']}-{variant_result_info['Delay Mode']}-{variant_result_info['Delay']}-{variant_result_info['Algo']}-{variant_result_info['Strategy']}"

                exp_variant_mapping[exp_identity_tag][iteration].append(
                    result_info_item
                )

                fields = list(result_info_item.keys())
                with open(result_file_path, "a+") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)

                    if flag:
                        writer.writeheader()
                        flag = False

                    writer.writerow(result_info_item)

    for k, v in exp_variant_mapping.items():
        iter_scores = [
            (
                i_iter,
                np.mean([it["D4rl_Score"] for it in iter_res]),
                [it["Seed"] for it in iter_res],
                np.std([it["D4rl_Score"] for it in iter_res]),
            )
            for i_iter, iter_res in v.items()
        ]

        try:
            agg_item = deepcopy(v[0][0])
            sorted_iter_scores = sorted(
                iter_scores, key=lambda v: v[1], reverse=True
            )
            if debug:
                logger.debug(f"{k}, {len(iter_scores)}")

            selected_item = sorted_iter_scores[0]
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

        except Exception:
            print("=" * 100)
            print(k)
            print("=" * 100)

    return exp_variant_mapping


fetch_experiment_results()
