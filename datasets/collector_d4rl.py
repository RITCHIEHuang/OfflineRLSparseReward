import argparse
import os

import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import d4rl

task_list = [
    "halfcheetah-random-v0",
    "halfcheetah-medium-v0",
    "halfcheetah-expert-v0",
    "halfcheetah-medium-replay-v0",
    "halfcheetah-medium-expert-v0",
    "walker2d-random-v0",
    "walker2d-medium-v0",
    "walker2d-expert-v0",
    "walker2d-medium-replay-v0",
    "walker2d-medium-expert-v0",
    "hopper-random-v0",
    "hopper-medium-v0",
    "hopper-expert-v0",
    "hopper-medium-replay-v0",
    "hopper-medium-expert-v0",
    "ant-random-v0",
    "ant-medium-v0",
    "ant-expert-v0",
    "ant-medium-replay-v0",
    "ant-medium-expert-v0",
]


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("Dataset collector for d4rl")
    parser.add_argument("--delay", help="delay steps", type=int, default=20)
    parser.add_argument(
        "--dataset_dir",
        help="directory to store datasets",
        type=str,
        default="dataset",
    )
    parser.add_argument(
        "--task",
        help="task name",
        type=str,
        default="walker2d-expert-v0",
        choices=task_list,
    )

    return parser.parse_args()


def trans_dataset(args):
    env = gym.make(args.task)
    dataset = env.get_dataset()
    raw_rewards = dataset["rewards"]
    raw_terminals = dataset["terminals"]
    raw_timeouts = dataset["timeouts"]
    episode_ends = np.argwhere(np.logical_or(raw_terminals == True, raw_timeouts == True))
    data_size = len(raw_rewards)
    if episode_ends[-1][0] + 1 != data_size:
        episode_ends = np.append(episode_ends, data_size - 1)
    delay_rewards = np.zeros_like(raw_rewards)

    trans_idx = 0
    plot = True
    for ep in tqdm(range(len(episode_ends))):
        last_idx = trans_idx
        episode_end_idx = episode_ends[ep]
        while trans_idx < episode_end_idx:
            delay_idx = min(trans_idx + args.delay, episode_end_idx)
            delay_rewards[delay_idx - 1] = np.sum(
                raw_rewards[trans_idx:delay_idx]
            )
            # print(trans_idx, delay_idx, np.sum(raw_rewards[trans_idx:delay_idx]))
            trans_idx += args.delay

        if plot:
            plot_ep_reward(
                raw_rewards[last_idx:episode_end_idx],
                delay_rewards[last_idx:episode_end_idx],
            )
            plot = False

        trans_idx = episode_end_idx

    print(f"Task: {args.task}, data size: {len(raw_rewards)}")
    dataset["rewards"] = delay_rewards
    return dataset


def plot_ep_reward(raw_rewards, delay_rewards):
    assert len(raw_rewards) == len(delay_rewards)
    plt.plot(
        range(len(raw_rewards)),
        raw_rewards,
        "b.-",
        range(len(delay_rewards)),
        delay_rewards,
        "r.-",
    )
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.title(f"{exp_name} reward")
    plt.legend(["Raw","Delayed"])
    plt.savefig(f"{exp_name}_reward.png")


if __name__ == "__main__":
    args = argsparser()
    if not os.path.exists(args.dataset_dir):
        os.makedirs(args.dataset_dir)

    exp_name = f"{args.task}_delay_{args.delay}"

    dataset = trans_dataset(args)
