import argparse
import os

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger

from torch.utils.data.dataloader import DataLoader

import d4rl
from offlinerl.utils.data import SampleBatch

from utils.d4rl_tasks import task_list
from utils.io_util import proj_path


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
        "--task",
        help="task name",
        type=str,
        default="d4rl-walker2d-expert-v0",
        choices=task_list,
    )

    return parser.parse_args()


def qlearning_dataset(env, dataset=None, terminate_on_end=False, **kwargs):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.

    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().

    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset["rewards"].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    time_out_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if "timeouts" in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset["observations"][i].astype(np.float32)
        new_obs = dataset["observations"][i + 1].astype(np.float32)
        action = dataset["actions"][i].astype(np.float32)
        reward = dataset["rewards"][i].astype(np.float32)
        done_bool = bool(dataset["terminals"][i])
        time_out = bool(dataset["timeouts"][i])

        if use_timeouts:
            final_timestep = dataset["timeouts"][i]
        else:
            final_timestep = episode_step == env._max_episode_steps - 1
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            episode_step = 0
            continue
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        time_out_.append(time_out)
        episode_step += 1

    return {
        "observations": np.array(obs_),
        "actions": np.array(action_),
        "next_observations": np.array(next_obs_),
        "rewards": np.array(reward_),
        "terminals": np.array(done_),
        "timeouts": np.array(time_out_),
    }


def trans_traj_dataset(config):
    """Generate trajectory data

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
    plot = True

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

        if plot:
            plot_ep_reward(
                traj_rewards,
                traj_delay_rewards,
                traj_returns,
                config,
            )
            plot = False

    logger.info(
        f"Task: {config['task']}, data size: {len(raw_rewards)}, traj num: {len(traj_dataset['length'])}"
    )
    return traj_dataset


def trans_dataset(config):
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
    plot = True

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

        if plot:
            plot_ep_reward(
                raw_rewards[last_idx : episode_end_idx + 1],
                delay_rewards[last_idx : episode_end_idx + 1],
                returns[last_idx : episode_end_idx + 1],
                config,
            )
            plot = False

        trans_idx = episode_end_idx + 1

    dataset["rewards"] = delay_rewards
    dataset["returns"] = returns
    logger.info(
        f"[TransDataset] Task: {config['task']}, data size: {len(raw_rewards)}"
    )
    return dataset


def plot_ep_reward(raw_rewards, delay_rewards, returns, config):
    assert len(raw_rewards) == len(delay_rewards)
    plt.plot(
        range(len(raw_rewards)),
        raw_rewards,
        "b.-",
    )
    plt.plot(
        range(len(delay_rewards)),
        delay_rewards,
        "r.-",
    )
    plt.plot(
        range(len(returns)),
        returns,
        "g.-",
    )
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.title(f"{config['task']}-delay_mode_{config['delay_mode']}_reward")
    plt.legend(["Raw", "Delayed"])

    if config["delay_mode"] == "random":
        fig_name = f"delay-mode_{config['delay_mode']}_delay-min_{config['delay_min']}_delay-max_{config['delay_max']}.png"
    elif config["delay_mode"] == "constant":
        fig_name = (
            f"delay-mode_{config['delay_mode']}_delay_{config['delay']}.png"
        )
    else:
        raise NotImplementedError()

    fig_dir = f"{proj_path}/assets/{config['task']}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/{fig_name}")


def load_d4rl_buffer(config):
    dataset = trans_dataset(config)
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


def load_d4rl_traj_buffer(config):
    from datasets.traj_dataset import TrajDataset

    BATCH_SIZE = 1
    dataset = TrajDataset(config)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


if __name__ == "__main__":
    args = argsparser()
    # if not os.path.exists(args.dataset_dir):
    #     os.makedirs(args.dataset_dir)

    """extract transition buffer"""
    # load_d4rl_buffer(vars(args))

    """extract traj dataset"""
    # traj_dataset = trans_traj_dataset(vars(args))
    traj_dataset = trans_dataset(vars(args))

    """extract traj buffer"""
    # load_d4rl_traj_buffer(vars(args))
