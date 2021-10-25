import argparse
import os
from copy import deepcopy

import torch
import gym
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

from torch.utils.data.dataloader import DataLoader

import d4rl
from offlinerl.utils.data import SampleBatch
from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.utils.config import parse_config

from utils.d4rl_tasks import task_list
from utils.io_util import proj_path

from datasets.traj_dataset import TrajDataset
from datasets.qlearning_dataset import qlearning_dataset
from algos import reward_shaper, reward_decoposer, reward_giver
from config import shaping_config, decomposer_config, reward_giver_config


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("Dataset collector for d4rl")
    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument(
        "--delay_mode",
        help="delay mode",
        type=str,
        default="constant",
        choices=["constant", "random"],
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
        "--strategy",
        help="delay rewards strategy, can be multiple strategies seperated by  `,`",
        type=str,
        default="none",
        # choices=[
        #     "none",
        #     "scale",
        #     "scale_v2",
        #     "minmax",
        #     "zscore",
        #     "episodic_average",
        #     "episodic_random",
        #     "episodic_ensemble",
        #     "interval_average",
        #     "interval_random",
        #     "interval_ensemble",
        #     "transformer_decompose",
        #     "pg_reshaping",
        # ],
    )
    parser.add_argument(
        "--task",
        help="task name",
        type=str,
        default="d4rl-walker2d-expert-v0",
        choices=task_list,
    )

    return parser.parse_args()


def delay_transition_dataset(config):
    """Generate transition data for delay-reward

    Args:
        config (dict):  dataset configuration
    """
    np.random.seed(config["seed"])
    env = gym.make(
        config["task"][5:] if "d4rl" in config["task"] else config["task"]
    )
    # dataset = env.get_dataset()
    dataset = qlearning_dataset(env, terminate_on_end=True)
    raw_rewards = dataset["rewards"]
    raw_terminals = dataset["terminals"]
    raw_timeouts = dataset["timeouts"]
    episode_ends = np.argwhere(
        np.logical_or(raw_terminals == True, raw_timeouts == True)
    )
    data_size = len(raw_rewards)
    if episode_ends[-1][0] + 1 != data_size:
        episode_ends = np.append(episode_ends, data_size - 1)
    delay_rewards = np.zeros_like(raw_rewards)
    returns = np.zeros_like(raw_rewards)

    trans_idx = 0
    plot_traj_idx_list = [np.random.randint(0, len(dataset)) for _ in range(5)]

    if config["delay_mode"] == "constant":
        delay = config["delay"]
    else:
        delay = np.random.randint(config["delay_min"], config["delay_max"] + 1)

    for ep in tqdm(range(len(episode_ends))):
        last_idx = trans_idx
        episode_end_idx = episode_ends[ep]
        return_ = raw_rewards[last_idx : episode_end_idx + 1].sum()
        returns[last_idx : episode_end_idx + 1] = return_
        while trans_idx <= episode_end_idx:
            delay_idx = min(trans_idx + delay, episode_end_idx)
            delay_rewards[delay_idx] = np.sum(
                raw_rewards[trans_idx : delay_idx + 1]
            )
            returns[trans_idx : trans_idx + delay] = np.sum(
                raw_rewards[trans_idx : episode_end_idx + 1]
            )
            # print(trans_idx, delay_idx, np.sum(raw_rewards[trans_idx:delay_idx]))
            trans_idx += delay

            if config["delay_mode"] == "random":
                delay = np.random.randint(
                    config["delay_min"], config["delay_max"] + 1
                )

        if ep in plot_traj_idx_list:
            plot_ep_reward(
                [
                    raw_rewards[last_idx : episode_end_idx + 1],
                    delay_rewards[last_idx : episode_end_idx + 1],
                    # returns[last_idx : episode_end_idx + 1],
                ],
                # ["raw", "delay", "return"],
                ["raw", "delay"],
                config,
                suffix=ep,
            )

        trans_idx = episode_end_idx + 1

    dataset["rewards"] = delay_rewards
    dataset["returns"] = returns
    logger.info(
        f"[TransDataset] Task: {config['task']}, data size: {len(raw_rewards)}"
    )
    return dataset


def load_d4rl_buffer(config):
    dataset = delay_transition_dataset(config)
    buffer = SampleBatch(
        obs=dataset["observations"],
        obs_next=dataset["next_observations"],
        act=dataset["actions"],
        rew=np.expand_dims(np.squeeze(dataset["rewards"]), 1),
        ret=np.expand_dims(np.squeeze(dataset["returns"]), 1),
        done=np.expand_dims(np.squeeze(dataset["terminals"]), 1),
    )

    logger.info("obs shape: {}", buffer.obs.shape)
    logger.info("obs_next shape: {}", buffer.obs_next.shape)
    logger.info("act shape: {}", buffer.act.shape)
    logger.info("rew shape: {}", buffer.rew.shape)
    logger.info("ret shape: {}", buffer.ret.shape)
    logger.info("done shape: {}", buffer.done.shape)
    logger.info("Episode reward: {}", buffer.rew.sum() / np.sum(buffer.done))
    logger.info("Number of terminals on: {}", np.sum(buffer.done))
    return buffer


def delay_traj_dataset(config):
    """Generate trajectory data for delay-reward

    Args:
        config (dict):  dataset configuration
    """
    if "seed" not in config:
        config["seed"] = 42
    np.random.seed(config["seed"])
    env = gym.make(
        config["task"][5:] if "d4rl" in config["task"] else config["task"]
    )
    # dataset = env.get_dataset()
    dataset = qlearning_dataset(env, terminate_on_end=True)
    raw_observations = dataset["observations"]
    raw_actions = dataset["actions"]
    raw_rewards = dataset["rewards"]
    raw_terminals = dataset["terminals"]
    raw_timeouts = dataset["timeouts"]
    raw_next_obs = dataset["next_observations"]

    keys = [
        "observations",
        "actions",
        "delay_rewards",
        "rewards",
        "terminals",
        "length",
        "next_observations",
        "returns",
    ]
    traj_dataset = {k: [] for k in keys}

    episode_ends = np.argwhere(
        np.logical_or(raw_terminals == True, raw_timeouts == True)
    )
    data_size = len(raw_rewards)
    if episode_ends[-1][0] + 1 != data_size:
        episode_ends = np.append(episode_ends, data_size - 1)
    delay_rewards = np.zeros_like(raw_rewards)

    if config["delay_mode"] == "constant":
        delay = config["delay"]
    else:
        delay = np.random.randint(config["delay_min"], config["delay_max"] + 1)

    trans_idx = 0
    last_delay_idx = 0
    plot_traj_idx_list = [np.random.randint(0, len(dataset)) for _ in range(5)]
    for ep in tqdm(range(len(episode_ends))):
        traj_observations = []
        traj_actions = []
        traj_terminals = []
        traj_rewards = []
        traj_delay_rewards = []
        traj_next_obs = []

        episode_end_idx = episode_ends[ep]
        episode_idx = 1

        while trans_idx <= episode_end_idx:
            traj_observations.append(raw_observations[trans_idx])
            traj_actions.append(raw_actions[trans_idx])
            traj_terminals.append(raw_terminals[trans_idx])
            traj_rewards.append(raw_rewards[trans_idx])
            traj_next_obs.append(raw_next_obs[trans_idx])

            if episode_idx % delay == 0 or trans_idx == episode_end_idx:
                delay_rewards[trans_idx] = np.sum(
                    raw_rewards[last_delay_idx : trans_idx + 1]
                )
                last_delay_idx = trans_idx

                if config["delay_mode"] == "random":
                    delay = np.random.randint(
                        config["delay_min"], config["delay_max"] + 1
                    )

            traj_delay_rewards.append(delay_rewards[trans_idx])
            trans_idx += 1
            episode_idx += 1

        traj_returns = np.cumsum(traj_delay_rewards[::-1])[::-1]
        traj_dataset["observations"].append(np.array(traj_observations))
        traj_dataset["actions"].append(np.array(traj_actions))
        traj_dataset["terminals"].append(np.array(traj_terminals))
        traj_dataset["rewards"].append(np.array(traj_rewards))
        traj_dataset["delay_rewards"].append(np.array(traj_delay_rewards))
        traj_dataset["length"].append(len(traj_observations))
        traj_dataset["next_observations"].append(np.array(traj_next_obs))
        traj_dataset["returns"].append(traj_returns)

        if ep in plot_traj_idx_list:
            plot_ep_reward(
                # [traj_rewards, traj_delay_rewards, traj_returns],
                [traj_rewards, traj_delay_rewards],
                # ["raw", "delay", "return"],
                ["raw", "delay"],
                config,
                suffix=f"{ep}_delayed",
            )
    delay_rewards = np.squeeze(np.concatenate(traj_dataset["delay_rewards"]))
    # specific processing
    if "strategy" in config:
        # delay rewards process strategy
        strategies = config["strategy"]
        logger.info(f"Deal with delay rewards by strategies: {strategies} !!!")
        if strategies == "none":
            pass
        else:
            # for multi strategies, process by order
            strategies = strategies.split(",")
            for strategy in strategies:
                strategy = strategy.strip()
                traj_dataset = load_reward_by_strategy(
                    config, traj_dataset, plot_traj_idx_list, strategy
                )
    # plot reward distribution
    # processed_delay_rewards = np.squeeze(
    #     np.concatenate(traj_dataset["delay_rewards"])
    # )
    # raw_rewards = np.squeeze(np.concatenate(traj_dataset["rewards"]))
    # plot_reward_dist(
    #     [
    #         raw_rewards,
    #         delay_rewards[delay_rewards >= 1e-5],
    #         processed_delay_rewards,
    #     ],
    #     ["raw", "delay", "processed_delay"],
    #     config,
    #     config["strategy"],
    # )

    logger.info(
        f"Task: {config['task']}, data size: {len(raw_rewards)}, traj num: {len(traj_dataset['length'])}"
    )
    return traj_dataset


def reward_redistributed(predictions, rewards, length):
    redistributed_reward = predictions[..., 1:] - predictions[..., :-1]
    redistributed_reward = torch.cat(
        [predictions[..., :1], redistributed_reward], dim=1
    )
    returns = rewards.sum(dim=1)
    for i, l in enumerate(length):
        redistributed_reward[i, l:] = 0
    predicted_returns = redistributed_reward.sum(dim=1)
    prediction_error = returns - predicted_returns
    for i, l in enumerate(length):
        redistributed_reward[i, :l] += prediction_error[i, None] / l
    return redistributed_reward


def process_episodic_average(traj_dataset):
    traj_dataset["smooth_rewards"] = deepcopy(traj_dataset["delay_rewards"])
    for i, traj_length in enumerate(traj_dataset["length"]):
        cur_episode_delay_rewards = traj_dataset["delay_rewards"][i]
        average_reward = np.sum(cur_episode_delay_rewards) / traj_length
        traj_dataset["smooth_rewards"][i] = average_reward * np.ones_like(
            cur_episode_delay_rewards
        )

    return traj_dataset


def process_interval_average(traj_dataset):
    traj_dataset["smooth_rewards"] = deepcopy(traj_dataset["delay_rewards"])
    for i, traj_length in enumerate(traj_dataset["length"]):
        cur_episode_delay_rewards = traj_dataset["delay_rewards"][i]
        episode_idx = 0
        while episode_idx < traj_length:
            interval_start_idx = episode_idx
            while (
                episode_idx + 1 < traj_length
                and abs(cur_episode_delay_rewards[episode_idx]) <= 1e-5
            ):
                episode_idx += 1
            # interval_range [interval_start_idx, episode_idx]
            average_reward = cur_episode_delay_rewards[episode_idx] / (
                episode_idx - interval_start_idx + 1
            )
            traj_dataset["smooth_rewards"][i][
                interval_start_idx : episode_idx + 1
            ] = average_reward * np.ones_like(
                cur_episode_delay_rewards[interval_start_idx : episode_idx + 1]
            )
            episode_idx += 1
    return traj_dataset


def load_smooth_traj_buffer(traj_dataset):
    buffer = SampleBatch(
        obs=np.concatenate(traj_dataset["observations"], axis=0),
        obs_next=np.concatenate(traj_dataset["next_observations"], axis=0),
        act=np.concatenate(traj_dataset["actions"], axis=0),
        rew=np.expand_dims(
            np.squeeze(np.concatenate(traj_dataset["smooth_rewards"])), 1
        ),
        ret=np.expand_dims(
            np.squeeze(np.concatenate(traj_dataset["returns"])), 1
        ),
        done=np.expand_dims(
            np.squeeze(np.concatenate(traj_dataset["terminals"])), 1
        ),
    )

    logger.info("obs shape: {}", buffer.obs.shape)
    logger.info("obs_next shape: {}", buffer.obs_next.shape)
    logger.info("act shape: {}", buffer.act.shape)
    logger.info("rew shape: {}", buffer.rew.shape)
    logger.info("ret shape: {}", buffer.ret.shape)
    logger.info("done shape: {}", buffer.done.shape)
    logger.info("Episode reward: {}", buffer.rew.sum() / np.sum(buffer.done))
    logger.info("Number of terminals on: {}", np.sum(buffer.done))
    return buffer


def load_reward_by_strategy(
    config, traj_dataset, plot_traj_idx_list, strategy
):
    if "exp_name" not in config:
        config["exp_name"] = "example"
    tmp = np.array(
        [
            traj_dataset["returns"][i][0]
            for i in range(len(traj_dataset["returns"]))
        ]
    )
    return_max = np.max(tmp)
    return_min = np.min(tmp)
    return_mean = np.mean(tmp)
    return_std = np.std(tmp)
    # preprocessing
    # (1) for `transformer_decompose` and `pg_shaping`, train the model first;
    # (2) for `episodic_ensemble` and `interval_ensemble`, process by corresponoding `average` strategy first.

    if strategy == "transformer_decompose":
        # train decompose model
        logger.info(f"Training Transformer decompose model start...")
        dataset = TrajDataset(traj_dataset)

        algo_config = parse_config(decomposer_config)
        algo_config["task"] = config["task"]
        algo_config["log_path"] = config["log_path"]

        algo_config["exp_name"] = f"{config['exp_name']}-reward_decomposer"

        train_dataloader = DataLoader(
            dataset, batch_size=algo_config["batch_size"], shuffle=True
        )
        # val_dataloader = DataLoader(
        #     dataset, batch_size=algo_config["batch_size"] * 5, shuffle=True
        # )
        algo_init = reward_decoposer.algo_init(algo_config)
        algo_trainer = reward_decoposer.AlgoTrainer(algo_init, algo_config)
        init_decomposer_model = deepcopy(algo_trainer.get_policy())

        algo_trainer.train(train_dataloader, None, None)

        trained_decomposer_model = algo_trainer.get_policy()
        device = algo_trainer.device

        logger.info(f"Training Transformer decompose model end...")

    elif strategy == "pg_reshaping":
        # train reshaping model
        logger.info(f"Training PG reshaping model start...")
        algo_config = parse_config(shaping_config)
        algo_config.update(
            {
                "policy_mode": "random",
                "shaping_version": "v2",
            }  # proximal
        )
        algo_config["task"] = config["task"]
        algo_config["log_path"] = config["log_path"]
        algo_config[
            "exp_name"
        ] = f"{config['exp_name']}-reward_shaper-policy_mode-{algo_config['policy_mode']}-shaping_version-{algo_config['shaping_version']}"

        train_buffer = load_d4rl_buffer(config)

        algo_init = reward_shaper.algo_init(algo_config)
        algo_trainer = reward_shaper.AlgoTrainer(algo_init, algo_config)

        init_shaping_model = deepcopy(algo_trainer.get_model())
        callback = OnlineCallBackFunction()
        callback.initialize(
            train_buffer=train_buffer,
            val_buffer=None,
            task=algo_config["task"],
        )
        callback_list = CallBackFunctionList(
            [callback, d4rl_eval_fn(task=algo_config["task"])]
        )

        algo_trainer.train(train_buffer, None, callback_fn=callback_list)
        logger.info(f"Training PG reshaping model end ...")

        trained_shaping_model = algo_trainer.get_model()
        device = algo_trainer.device

    elif strategy in ["episodic_ensemble", "interval_ensemble"]:
        logger.info(f"Training reward giver model start...")
        if strategy == "episodic_ensemble":
            traj_dataset = process_episodic_average(traj_dataset)
        else:
            traj_dataset = process_interval_average(traj_dataset)

        buffer = load_smooth_traj_buffer(traj_dataset)
        algo_config = parse_config(reward_giver_config)
        algo_config["task"] = config["task"]
        algo_config["log_path"] = config["log_path"]
        algo_config["exp_name"] = f"{config['exp_name']}-reward_giver"
        algo_init = reward_giver.algo_init(algo_config)
        algo_trainer = reward_giver.AlgoTrainer(algo_init, algo_config)
        init_reward_giver_model = deepcopy(algo_trainer.get_policy())

        algo_trainer.train(buffer, None, None)

        trained_reward_giver_model = algo_trainer.get_policy()
        device = algo_trainer.device

        logger.info(f"Training reward giver model end...")

    reward_max = np.max(
        [
            np.max(traj_dataset["delay_rewards"][i])
            for i in range(len(traj_dataset["returns"]))
        ]
    )
    reward_min = np.min(
        [
            np.min(traj_dataset["delay_rewards"][i])
            for i in range(len(traj_dataset["returns"]))
        ]
    )
    # reward_mean = np.mean(tmp2)
    # reward_std = np.std(tmp2)
    logger.info(f"Delay reward min: {reward_min}, max: {reward_max}")

    def process_scale(reward):
        if abs(reward_min) < 1e-6:
            return reward / reward_max
        else:
            if reward < 0:
                return -reward / reward_min
            elif reward > 0:
                return reward / reward_max
            else:
                return reward

    def process_scale_v2(reward):
        return reward / (reward_max - reward_min)

    for i, traj_length in enumerate(traj_dataset["length"]):
        traj_delay_rewards = traj_dataset["delay_rewards"][i].copy()
        if strategy == "minmax":
            traj_delay_rewards = (
                traj_dataset["delay_rewards"][i]
                - return_min / traj_dataset["length"][i]
            ) / (return_max - return_min)
        elif strategy == "zscore":
            traj_delay_rewards = (
                traj_dataset["delay_rewards"][i]
                - return_mean / traj_dataset["length"][i]
            ) / return_std
        elif strategy == "scale":
            traj_delay_rewards = np.array(
                [
                    process_scale(traj_delay_rewards[i])
                    for i in range(traj_length)
                ]
            )
        elif strategy == "scale_v2":
            traj_delay_rewards = np.array(
                [
                    process_scale_v2(traj_delay_rewards[i])
                    for i in range(traj_length)
                ]
            )
        elif strategy == "episodic_average":
            traj_delay_rewards = np.ones_like(
                traj_dataset["delay_rewards"][i]
            ) * (np.sum(traj_dataset["delay_rewards"][i]) / traj_length)
        elif strategy == "episodic_random":
            weights = np.random.normal(
                size=traj_dataset["delay_rewards"][i].shape
            )
            normalized_weights = softmax(weights)
            traj_delay_rewards = normalized_weights * np.sum(
                traj_dataset["delay_rewards"][i]
            )
        elif strategy == "interval_average":
            episode_idx = 0
            while episode_idx < traj_length:
                interval_start_idx = episode_idx
                while (
                    episode_idx + 1 < traj_length
                    and abs(traj_delay_rewards[episode_idx]) <= 1e-5
                ):
                    episode_idx += 1
                # interval_range [interval_start_idx, episode_idx]
                average_reward = traj_delay_rewards[episode_idx] / (
                    episode_idx - interval_start_idx + 1
                )
                traj_delay_rewards[
                    interval_start_idx : episode_idx + 1
                ] = average_reward * np.ones_like(
                    traj_delay_rewards[interval_start_idx : episode_idx + 1]
                )
                episode_idx += 1

        elif strategy == "interval_random":
            episode_idx = 0
            while episode_idx < traj_length:
                interval_start_idx = episode_idx
                while (
                    episode_idx + 1 < traj_length
                    and abs(traj_delay_rewards[episode_idx]) <= 1e-5
                ):
                    episode_idx += 1
                # interval_range [interval_start_idx, episode_idx]
                weights = np.random.normal(
                    size=traj_delay_rewards[
                        interval_start_idx : episode_idx + 1
                    ].shape
                )
                normalized_weights = softmax(weights)
                traj_delay_rewards[
                    interval_start_idx : episode_idx + 1
                ] = normalized_weights * np.sum(
                    traj_delay_rewards[interval_start_idx : episode_idx + 1]
                )
                episode_idx += 1

        elif strategy == "interval_ensemble":
            smooth_delay_rewards = traj_dataset["smooth_rewards"][i]
            with torch.no_grad():
                traj_delay_rewards = torch.from_numpy(
                    traj_dataset["delay_rewards"][i]
                ).to(device)

                traj_obs = torch.from_numpy(
                    traj_dataset["observations"][i]
                ).to(device)
                traj_act = torch.from_numpy(traj_dataset["actions"][i]).to(
                    device
                )
                traj_obs_act_pair = torch.cat([traj_obs, traj_act], dim=-1)
                # init
                init_reward_pre = (
                    init_reward_giver_model(
                        traj_obs_act_pair.unsqueeze(dim=0),
                    )
                    .squeeze(dim=-1)
                    .squeeze(dim=0)
                )

                # trained
                trained_reward_pre = (
                    trained_reward_giver_model(
                        traj_obs_act_pair.unsqueeze(dim=0),
                    )
                    .squeeze(dim=-1)
                    .squeeze(dim=0)
                )

            init_reward_redistribution = torch.empty_like(traj_delay_rewards)
            trained_reward_redistribution = torch.empty_like(
                traj_delay_rewards
            )

            episode_idx = 0
            while episode_idx < traj_length:
                interval_start_idx = episode_idx
                while (
                    episode_idx + 1 < traj_length
                    and abs(traj_delay_rewards[episode_idx]) <= 1e-5
                ):
                    episode_idx += 1
                # interval_range [interval_start_idx, episode_idx]
                init_weights = (
                    traj_delay_rewards[
                        interval_start_idx : episode_idx + 1
                    ].sum()
                    / init_reward_pre[
                        interval_start_idx : episode_idx + 1
                    ].sum()
                )

                init_reward_redistribution[
                    interval_start_idx : episode_idx + 1
                ] = (
                    init_reward_pre[interval_start_idx : episode_idx + 1]
                    * init_weights
                )
                trained_weights = (
                    traj_delay_rewards[
                        interval_start_idx : episode_idx + 1
                    ].sum()
                    / trained_reward_pre[
                        interval_start_idx : episode_idx + 1
                    ].sum()
                )
                trained_reward_redistribution[
                    interval_start_idx : episode_idx + 1
                ] = (
                    trained_reward_pre[interval_start_idx : episode_idx + 1]
                    * trained_weights
                )
                episode_idx += 1

            if i in plot_traj_idx_list:
                plot_ep_reward(
                    [
                        smooth_delay_rewards,
                        init_reward_redistribution.cpu().numpy(),
                        trained_reward_redistribution.cpu().numpy(),
                    ],
                    ["smooth_reward", "init_reward", "trained_reward"],
                    config,
                    suffix=f"{i}_{strategy}_compare",
                )

                plot_ep_reward(
                    [
                        traj_dataset["rewards"][i],
                        smooth_delay_rewards,
                        trained_reward_redistribution.cpu().numpy(),
                    ],
                    ["non-delay_reward", "smooth_reward", "trained_reward"],
                    config,
                    suffix=f"{i}_{strategy}_compare_raw",
                )
            traj_delay_rewards = trained_reward_redistribution.cpu().numpy()

        elif strategy == "episodic_ensemble":
            smooth_delay_rewards = traj_dataset["smooth_rewards"][i]
            with torch.no_grad():
                traj_delay_rewards = torch.from_numpy(
                    traj_dataset["delay_rewards"][i]
                ).to(device)

                traj_obs = torch.from_numpy(
                    traj_dataset["observations"][i]
                ).to(device)
                traj_act = torch.from_numpy(traj_dataset["actions"][i]).to(
                    device
                )
                traj_obs_act_pair = torch.cat([traj_obs, traj_act], dim=-1)
                # init
                init_reward_pre = (
                    init_reward_giver_model(
                        traj_obs_act_pair.unsqueeze(dim=0),
                    )
                    .squeeze(dim=-1)
                    .squeeze(dim=0)
                )

                init_rescale_weight = (
                    traj_delay_rewards.sum() / init_reward_pre.sum()
                )
                init_reward_redistribution = (
                    init_reward_pre * init_rescale_weight
                )
                # trained
                trained_reward_pre = (
                    trained_reward_giver_model(
                        traj_obs_act_pair.unsqueeze(dim=0),
                    )
                    .squeeze(dim=-1)
                    .squeeze(dim=0)
                )

                trained_rescale_weight = (
                    traj_delay_rewards.sum() / trained_reward_pre.sum()
                )
                trained_reward_redistribution = (
                    trained_reward_pre * trained_rescale_weight
                )

            if i in plot_traj_idx_list:
                plot_ep_reward(
                    [
                        # traj_delay_rewards.cpu().numpy(),
                        smooth_delay_rewards,
                        init_reward_redistribution.cpu().numpy(),
                        trained_reward_redistribution.cpu().numpy(),
                    ],
                    ["smooth_reward", "init_reward", "trained_reward"],
                    config,
                    suffix=f"{i}_{strategy}_compare",
                )
            traj_delay_rewards = trained_reward_redistribution.cpu().numpy()

        elif strategy == "transformer_decompose":
            with torch.no_grad():
                traj_delay_rewards = torch.from_numpy(
                    traj_dataset["delay_rewards"][i]
                ).to(device)

                traj_obs = torch.from_numpy(
                    traj_dataset["observations"][i]
                ).to(device)
                traj_act = torch.from_numpy(traj_dataset["actions"][i]).to(
                    device
                )
                traj_obs_act_pair = torch.cat([traj_obs, traj_act], dim=-1)
                # init
                init_reward_pre = init_decomposer_model(
                    traj_obs_act_pair.unsqueeze(dim=0),
                ).squeeze(dim=-1)
                init_reward_redistribution = reward_redistributed(
                    init_reward_pre,
                    traj_delay_rewards.unsqueeze(dim=0),
                ).squeeze(dim=0)

                # trained
                trained_reward_pre = trained_decomposer_model(
                    traj_obs_act_pair.unsqueeze(dim=0),
                ).squeeze(dim=-1)
                trained_reward_redistribution = reward_redistributed(
                    trained_reward_pre,
                    traj_delay_rewards.unsqueeze(dim=0),
                ).squeeze(dim=0)

            if i in plot_traj_idx_list:
                plot_ep_reward(
                    [
                        traj_delay_rewards.cpu().numpy(),
                        init_reward_redistribution.cpu().numpy(),
                        trained_reward_redistribution.cpu().numpy(),
                    ],
                    ["raw", "init_decompose", "trained_decompose"],
                    config,
                    suffix=f"{i}_{strategy}_compare",
                )
            traj_delay_rewards = (
                trained_reward_redistribution.squeeze(-1).cpu().numpy()
            )

        elif strategy == "pg_reshaping":
            with torch.no_grad():
                traj_delay_rewards = torch.from_numpy(
                    traj_dataset["delay_rewards"][i]
                ).to(device)
                traj_next_obs = torch.from_numpy(
                    traj_dataset["next_observations"][i]
                ).to(device)
                traj_obs = torch.from_numpy(
                    traj_dataset["observations"][i]
                ).to(device)

                # init
                init_residual_rewards = init_shaping_model(
                    traj_next_obs.unsqueeze(dim=0)
                ) - init_shaping_model(traj_obs.unsqueeze(dim=0))
                init_residual_rewards = init_residual_rewards.squeeze(
                    dim=-1
                ).squeeze(dim=0)
                init_delay_rewards = traj_delay_rewards + init_residual_rewards

                # trained
                trained_residual_rewards = trained_shaping_model(
                    traj_next_obs.unsqueeze(dim=0)
                ) - trained_shaping_model(traj_obs.unsqueeze(dim=0))
                trained_residual_rewards = trained_residual_rewards.squeeze(
                    dim=-1
                ).squeeze(dim=0)
                trained_delay_rewards = (
                    traj_delay_rewards + trained_residual_rewards
                )

            if i in plot_traj_idx_list:
                plot_ep_reward(
                    [
                        traj_delay_rewards.cpu().numpy(),
                        init_delay_rewards.cpu().numpy(),
                        trained_delay_rewards.cpu().numpy(),
                    ],
                    ["raw", "init_shaping", "trained_shaping"],
                    config,
                    suffix=f"{i}_{strategy}_compare",
                )
            traj_delay_rewards = trained_delay_rewards.cpu().numpy()
            logger.debug(
                f"PG reshaping rewards shape: {traj_delay_rewards.shape}"
            )

        else:
            raise NotImplementedError()

        if i in plot_traj_idx_list:
            plot_ep_reward(
                [
                    # traj_dataset["delay_rewards"][i],
                    traj_delay_rewards,
                    # traj_dataset["rewards"][i],
                    # traj_dataset["delay_rewards"][i],
                    # traj_dataset["returns"][i],
                ],
                ["strategy"],
                # ["delay", "strategy"],
                # ["raw", "delay", "return"],
                config,
                suffix=f"{i}_{strategy}",
            )

        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]

    return traj_dataset


def load_d4rl_traj_buffer(config):
    traj_dataset = delay_traj_dataset(config)

    buffer = SampleBatch(
        obs=np.concatenate(traj_dataset["observations"], axis=0),
        obs_next=np.concatenate(traj_dataset["next_observations"], axis=0),
        act=np.concatenate(traj_dataset["actions"], axis=0),
        rew=np.expand_dims(
            np.squeeze(np.concatenate(traj_dataset["delay_rewards"])), 1
        ),
        ret=np.expand_dims(
            np.squeeze(np.concatenate(traj_dataset["returns"])), 1
        ),
        done=np.expand_dims(
            np.squeeze(np.concatenate(traj_dataset["terminals"])), 1
        ),
    )

    logger.info("obs shape: {}", buffer.obs.shape)
    logger.info("obs_next shape: {}", buffer.obs_next.shape)
    logger.info("act shape: {}", buffer.act.shape)
    logger.info("rew shape: {}", buffer.rew.shape)
    logger.info("ret shape: {}", buffer.ret.shape)
    logger.info("done shape: {}", buffer.done.shape)
    logger.info("Episode reward: {}", buffer.rew.sum() / np.sum(buffer.done))
    logger.info("Number of terminals on: {}", np.sum(buffer.done))
    return buffer


def plot_ep_reward(data_list: list, names: list, config: dict, suffix=""):
    plt.figure()
    for data, name in zip(data_list, names):
        plt.plot(range(len(data)), data, label=name)
    plt.xlabel("t")
    plt.ylabel("rew")
    plt.title(f"{config['task']}-delay_mode_{config['delay_mode']}")
    plt.legend()

    fig_name = f"delay-mode_{config['delay_mode']}"
    if config["delay_mode"] == "random":
        fig_name = f"{fig_name}_delay-min_{config['delay_min']}_delay-max_{config['delay_max']}"
    elif config["delay_mode"] == "constant":
        fig_name = f"{fig_name}_delay_{config['delay']}"
    else:
        raise NotImplementedError()

    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/{fig_name}_{suffix}.png")


def plot_reward_dist(data_list: list, names: list, config: dict, suffix=""):
    plt.figure()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for idx, ax in enumerate(axes):
        if idx >= len(data_list):
            break
        ax.hist(data_list[idx], color="blue", edgecolor="black", bins=1000)
        ax.set_xlabel("val")
        ax.set_ylabel("proportion")
        ax.set_title(names[idx])

    fig_name = f"delay-mode_{config['delay_mode']}"
    if config["delay_mode"] == "random":
        fig_name = f"{fig_name}_delay-min_{config['delay_min']}_delay-max_{config['delay_max']}"
    elif config["delay_mode"] == "constant":
        fig_name = f"{fig_name}_delay_{config['delay']}"
    else:
        raise NotImplementedError()

    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/{fig_name}_distribution_{suffix}.png")


if __name__ == "__main__":
    args = argsparser()
    # if not os.path.exists(args.dataset_dir):
    #     os.makedirs(args.dataset_dir)

    config = vars(args)
    config["log_path"] = f"{proj_path}/logs"
    """extract transition buffer"""
    # load_d4rl_buffer(config)

    """extract traj dataset"""
    # traj_dataset = delay_traj_dataset(config)
    # traj_dataset = delay_transition_dataset(config)

    """extract traj buffer"""
    load_d4rl_traj_buffer(config)
