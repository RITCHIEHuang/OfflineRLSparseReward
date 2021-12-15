from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm
from loguru import logger

from torch.utils.data.dataloader import DataLoader

from datasets.traj_dataset import TrajDataset

from offlinerl.utils.data import SampleBatch
from offlinerl.utils.config import parse_config
from offlinerl.evaluation import OnlineCallBackFunction, CallBackFunctionList
from offlinerl.evaluation.d4rl import d4rl_eval_fn
from offlinerl.algo.custom import reward_shaper, reward_decoposer, reward_giver
from offlinerl.config.algo import (
    shaping_config,
    decomposer_config,
    reward_giver_config,
)
from utils.plot_util import plot_ep_reward

EPS = 1e-8


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


def load_traj_buffer(traj_dataset):
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


# Strategies
def scale_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    all_rewards = np.concatenate(traj_dataset["delay_rewards"])
    reward_max = all_rewards.max()
    reward_min = all_rewards.min()
    reward_mean = all_rewards.mean()
    reward_std = all_rewards.std()
    scale_type = config["strategy"].split("_")[-1]
    for i, traj_length in enumerate(traj_dataset["length"]):
        if scale_type == "scale":
            traj_delay_rewards = traj_dataset["delay_rewards"][i] / (
                reward_max - reward_min + EPS
            )
        elif scale_type == "minmax":
            traj_delay_rewards = (
                traj_dataset["delay_rewards"][i] - reward_min
            ) / (reward_max - reward_min + EPS)
        elif scale_type == "zscore":
            traj_delay_rewards = (
                traj_dataset["delay_rewards"][i] - reward_mean
            ) / (reward_std + EPS)
        else:
            raise NotImplementedError()

        if i in plot_traj_idx_list:
            plot_ep_reward(
                [
                    traj_dataset["rewards"][i],
                    traj_delay_rewards,
                ],
                ["raw", "strategy"],
                config,
                suffix=f"{i}_scale",
            )
        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]
    return traj_dataset


def minmax_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    tmp = np.array(
        [
            traj_dataset["returns"][i][0]
            for i in range(len(traj_dataset["returns"]))
        ]
    )
    return_max = np.max(tmp)
    return_min = np.min(tmp)

    for i, traj_length in enumerate(traj_dataset["length"]):
        traj_delay_rewards = traj_dataset["delay_rewards"][i].copy()
        traj_delay_rewards = (
            traj_dataset["delay_rewards"][i]
            - return_min / traj_dataset["length"][i]
        ) / (return_max - return_min + EPS)

        if i in plot_traj_idx_list:
            plot_ep_reward(
                [
                    traj_dataset["rewards"][i],
                    traj_delay_rewards,
                ],
                ["raw", "strategy"],
                config,
                suffix=f"{i}_minmax",
            )
        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]
    return traj_dataset


def interval_average_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    for i, traj_length in tqdm(enumerate(traj_dataset["length"])):
        traj_delay_rewards = traj_dataset["delay_rewards"][i].copy()
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

        if i in plot_traj_idx_list:
            plot_ep_reward(
                [
                    # traj_dataset["delay_rewards"][i],
                    traj_dataset["rewards"][i],
                    traj_delay_rewards,
                    # traj_dataset["delay_rewards"][i],
                    # traj_dataset["returns"][i],
                ],
                ["raw", "strategy"],
                config,
                suffix=f"{i}_interval_average",
            )

        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]
    return traj_dataset


def interval_ensemble_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    logger.info(f"Training reward giver model start...")
    traj_dataset = interval_average_strategy(
        traj_dataset, config, plot_traj_idx_list
    )

    buffer = load_traj_buffer(traj_dataset)
    algo_config = parse_config(reward_giver_config)
    algo_config["project"] = config["project"]
    algo_config["task"] = config["task"]
    algo_config["log_path"] = config["log_path"]
    algo_config["log_to_wandb"] = False
    algo_config["exp_name"] = f"{config['exp_name']}-reward_giver"
    algo_init = reward_giver.algo_init(algo_config)
    algo_trainer = reward_giver.AlgoTrainer(algo_init, algo_config)
    init_reward_giver_model = deepcopy(algo_trainer.get_policy())

    algo_trainer.train(buffer, None, None)

    trained_reward_giver_model = algo_trainer.get_policy()
    device = algo_trainer.device

    logger.info(f"Training reward giver model end...")
    for i, traj_length in enumerate(traj_dataset["length"]):
        smooth_delay_rewards = traj_dataset["delay_rewards"][i].copy()
        with torch.no_grad():
            traj_delay_rewards = torch.from_numpy(
                traj_dataset["delay_rewards"][i]
            ).to(device)

            traj_obs = torch.from_numpy(traj_dataset["observations"][i]).to(
                device
            )
            traj_act = torch.from_numpy(traj_dataset["actions"][i]).to(device)
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
        trained_reward_redistribution = torch.empty_like(traj_delay_rewards)

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
                traj_delay_rewards[interval_start_idx : episode_idx + 1].sum()
                / init_reward_pre[interval_start_idx : episode_idx + 1].sum()
            )

            init_reward_redistribution[
                interval_start_idx : episode_idx + 1
            ] = (
                init_reward_pre[interval_start_idx : episode_idx + 1]
                * init_weights
            )
            trained_weights = (
                traj_delay_rewards[interval_start_idx : episode_idx + 1].sum()
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

        traj_delay_rewards = trained_reward_redistribution.cpu().numpy()
        if i in plot_traj_idx_list:
            plot_ep_reward(
                [
                    traj_dataset["rewards"][i],
                    smooth_delay_rewards,
                    init_reward_redistribution.cpu().numpy(),
                    traj_delay_rewards,
                ],
                ["non-delay", "smoothed", "init", "trained"],
                config,
                suffix=f"{i}_interval_ensemble_compare",
            )

        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]
    return traj_dataset


def episodic_average_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    for i, traj_length in enumerate(traj_dataset["length"]):
        traj_delay_rewards = np.ones_like(traj_dataset["delay_rewards"][i]) * (
            np.sum(traj_dataset["delay_rewards"][i]) / traj_length
        )
        if i in plot_traj_idx_list:
            plot_ep_reward(
                [
                    # traj_dataset["delay_rewards"][i],
                    traj_delay_rewards,
                    traj_dataset["rewards"][i],
                    # traj_dataset["delay_rewards"][i],
                    # traj_dataset["returns"][i],
                ],
                ["strategy", "raw"],
                # ["delay", "strategy"],
                # ["raw", "delay", "return"],
                config,
                suffix=f"{i}_episodic_average",
            )
        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]
    return traj_dataset


def episodic_ensemble_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    logger.info(f"Training reward giver model start...")
    traj_dataset = episodic_average_strategy(
        traj_dataset, config, plot_traj_idx_list
    )

    buffer = load_traj_buffer(traj_dataset)
    algo_config = parse_config(reward_giver_config)
    algo_config["project"] = config["project"]
    algo_config["task"] = config["task"]
    algo_config["log_path"] = config["log_path"]
    algo_config["log_to_wandb"] = False
    algo_config["exp_name"] = f"{config['exp_name']}-reward_giver"
    algo_init = reward_giver.algo_init(algo_config)
    algo_trainer = reward_giver.AlgoTrainer(algo_init, algo_config)
    init_reward_giver_model = deepcopy(algo_trainer.get_policy())

    algo_trainer.train(buffer, None, None)

    trained_reward_giver_model = algo_trainer.get_policy()
    device = algo_trainer.device

    logger.info(f"Training reward giver model end...")
    for i, traj_length in enumerate(traj_dataset["length"]):
        smooth_delay_rewards = traj_dataset["delay_rewards"][i].copy()
        with torch.no_grad():
            traj_delay_rewards = torch.from_numpy(
                traj_dataset["delay_rewards"][i]
            ).to(device)

            traj_obs = torch.from_numpy(traj_dataset["observations"][i]).to(
                device
            )
            traj_act = torch.from_numpy(traj_dataset["actions"][i]).to(device)
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
            init_reward_redistribution = init_reward_pre * init_rescale_weight
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

            traj_delay_rewards = trained_reward_redistribution.cpu().numpy()
            if i in plot_traj_idx_list:
                plot_ep_reward(
                    [
                        smooth_delay_rewards,
                        init_reward_redistribution.cpu().numpy(),
                        trained_reward_redistribution.cpu().numpy(),
                    ],
                    ["smoothed", "init", "trained"],
                    config,
                    suffix=f"{i}_episodic_ensemble_compare",
                )

        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]
    return traj_dataset


def transformer_decompose_strategy(
    traj_dataset, config, plot_traj_idx_list=[]
):
    # train decompose model
    logger.info(f"Training Transformer decompose model start...")
    dataset = TrajDataset(traj_dataset)

    algo_config = parse_config(decomposer_config)
    algo_config["project"] = config["project"]
    algo_config["task"] = config["task"]
    algo_config["log_path"] = config["log_path"]
    algo_config["log_to_wandb"] = False

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

    for i, traj_length in enumerate(traj_dataset["length"]):
        with torch.no_grad():
            traj_delay_rewards = torch.from_numpy(
                traj_dataset["delay_rewards"][i]
            ).to(device)

            traj_obs = torch.from_numpy(traj_dataset["observations"][i]).to(
                device
            )
            traj_act = torch.from_numpy(traj_dataset["actions"][i]).to(device)
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

            traj_delay_rewards = (
                trained_reward_redistribution.squeeze(-1).cpu().numpy()
            )
            if i in plot_traj_idx_list:
                plot_ep_reward(
                    [
                        traj_delay_rewards.cpu().numpy(),
                        init_reward_redistribution.cpu().numpy(),
                        trained_reward_redistribution.cpu().numpy(),
                    ],
                    ["raw", "init", "trained"],
                    config,
                    suffix=f"{i}_transformer_decompose_compare",
                )

        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]

    return traj_dataset


def pg_reshaping_strategy(traj_dataset, config, plot_traj_idx_list=[]):
    # train reshaping model
    logger.info(f"Training PG reshaping model start...")
    algo_config = parse_config(shaping_config)
    algo_config.update(
        {
            "policy_mode": "random",
            "shaping_version": "v2",
        }  # proximal
    )
    algo_config["project"] = config["project"]
    algo_config["task"] = config["task"]
    algo_config["log_path"] = config["log_path"]
    algo_config["log_to_wandb"] = False
    algo_config[
        "exp_name"
    ] = f"{config['exp_name']}-reward_shaper-policy_mode-{algo_config['policy_mode']}-shaping_version-{algo_config['shaping_version']}"

    train_buffer = load_traj_buffer(traj_dataset)

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

    for i, traj_length in enumerate(traj_dataset["length"]):
        with torch.no_grad():
            traj_delay_rewards = torch.from_numpy(
                traj_dataset["delay_rewards"][i]
            ).to(device)
            traj_next_obs = torch.from_numpy(
                traj_dataset["next_observations"][i]
            ).to(device)
            traj_obs = torch.from_numpy(traj_dataset["observations"][i]).to(
                device
            )

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
                    ["raw", "init", "trained"],
                    config,
                    suffix=f"{i}_pg_reshaping_compare",
                )
            traj_delay_rewards = trained_delay_rewards.cpu().numpy()

        traj_dataset["delay_rewards"][i] = traj_delay_rewards
        traj_dataset["returns"][i] = np.cumsum(traj_delay_rewards[::-1])[::-1]

    return traj_dataset
