import gym
import numpy as np
from tqdm import tqdm
from loguru import logger

import d4rl

from utils.exp_util import setup_exp_args
from utils.plot_util import plot_ep_reward

from datasets.qlearning_dataset import qlearning_dataset
from datasets.strategy import load_traj_buffer
from datasets import STRATEGY_MAPPING


def delay_traj_dataset(config):
    """Generate trajectory data for delay-reward

    Args:
        config (dict):  dataset configuration
    """
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

    if config["delay_mode"] == "none":
        delay = 1
    elif config["delay_mode"] == "constant":
        delay = config["delay"]
    elif config["delay_mode"] == "random":
        delay = np.random.randint(config["delay_min"], config["delay_max"] + 1)
    else:
        raise NotImplementedError()

    trans_idx = 0
    last_delay_idx = 0

    plot_traj_idx_list = [
        np.random.randint(0, len(episode_ends)) for _ in range(5)
    ]
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
                last_delay_idx = trans_idx + 1

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
                suffix=f"{ep}_delayed_compare",
            )

    # strategy processing
    if "strategy" in config:
        strategies = config["strategy"]
        logger.info(f"Deal with delay rewards by strategies: {strategies} !!!")

        if strategies == "none":
            pass
        else:
            # for multi strategies, process by order
            strategies = strategies.split(",")
            for strategy in strategies:
                strategy = strategy.strip()
                strategy_func = STRATEGY_MAPPING[strategy]
                traj_dataset = strategy_func(
                    traj_dataset, config, plot_traj_idx_list
                )

    logger.info(
        f"Task: {config['task']}, data size: {len(raw_rewards)}, traj num: {len(traj_dataset['length'])}"
    )
    return traj_dataset


def load_d4rl_buffer(config):
    traj_dataset = delay_traj_dataset(config)
    traj_buffer = load_traj_buffer(traj_dataset)
    reward_scale = config.get("reward_scale", 1.0)
    reward_shift = config.get("reward_shift", 0.0)
    traj_buffer.rew = (traj_buffer.rew + reward_shift) * reward_scale

    return traj_buffer


if __name__ == "__main__":
    config = setup_exp_args()
    # config["task"] = "antmaze-umaze-diverse-v2"
    # config["task"] = "hopper-medium-replay-v0"
    # config["strategy"] = "interval_average"
    # config["delay_mode"] = "none"
    if config["log_to_wandb"]:
        config["log_to_wandb"] = False

    if "exp_name" not in config:
        config["exp_name"] = "example"

    """extract traj dataset"""
    traj_dataset = delay_traj_dataset(config)

    """extract traj buffer"""
    # load_d4rl_buffer(config)
