import os
import torch
from torch.utils import data
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from loguru import logger
import numpy as np

from offlinerl.utils.data import SampleBatch

from datasets.d4rl_dataset import trans_dataset, trans_traj_dataset
from algos.reward_decoposer import (
    TransformerRewardDecomposer,
    RandomNetRewardDecomposer,
    create_key_padding_mask,
)
from utils.io_util import proj_path


class TrajDataset(Dataset):
    def __init__(self, config) -> None:
        self.config = config

        traj_dataset = trans_traj_dataset(config)

        self.obs = traj_dataset["observations"]
        self.last_obs = [
            np.zeros(self.obs[0].shape, dtype=np.float32)
        ] + self.obs[1:]
        self.act = traj_dataset["actions"]
        self.delay_rew = traj_dataset["delay_rewards"]
        self.rew = traj_dataset["rewards"]
        self.done = traj_dataset["terminals"]
        self.next_obs = traj_dataset["next_observations"]

        tmp = np.array([x.sum() for x in self.delay_rew])
        self.return_max = np.max(tmp)
        self.return_min = np.min(tmp)
        self.return_mean = np.mean(tmp)
        self.return_std = np.std(tmp)

        self.length = traj_dataset["length"]
        self.max_length = max(self.length)
        self.obs_size = self.obs[0].shape[-1]
        self.act_size = self.act[0].shape[-1]

        logger.info(
            f"max length: {self.max_length}, obs_size: {self.obs_size}, act size: {self.act_size}"
        )

    def __len__(self):
        return len(self.length)

    def __getitem__(self, idx):
        sample = {}

        if "decomposed" in self.config:
            sample["terminals"] = self.done[idx]
            sample["actions"] = self.act[idx]
            sample["observations"] = self.obs[idx]
            sample["next_observations"] = self.next_obs[idx]
        valid_obs_act_pair = torch.cat(
            [torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx])],
            dim=-1,
        )
        valid_length = self.length[idx]
        obs_act_pair = torch.zeros(
            (self.max_length, self.obs_size + self.act_size)
        )
        obs_act_pair[:valid_length] = valid_obs_act_pair

        sample["obs_act_pair"] = obs_act_pair

        sample["obs"] = torch.zeros((self.max_length, self.obs_size))
        sample["obs"][:valid_length] = torch.from_numpy(self.obs[idx])

        sample["next_obs"] = torch.zeros((self.max_length, self.obs_size))
        sample["next_obs"][:valid_length] = torch.from_numpy(
            self.next_obs[idx]
        )

        sample["act"] = torch.zeros((self.max_length, self.act_size))
        sample["act"][:valid_length] = torch.from_numpy(self.act[idx])

        sample["last_obs"] = torch.zeros((self.max_length, self.obs_size))
        sample["last_obs"][:valid_length] = torch.from_numpy(
            self.last_obs[idx]
        )
        if "scale" in self.config:
            valid_rew = (torch.from_numpy(self.rew[idx])) * self.config[
                "scale"
            ]
        else:
            valid_rew = (torch.from_numpy(self.rew[idx]) - self.return_min) / (
                self.return_max - self.return_min
            )

        rew = torch.zeros((self.max_length,))
        rew[:valid_length] = valid_rew
        sample["reward"] = rew

        if "scale" in self.config:
            valid_delay_rew = (
                torch.from_numpy(self.delay_rew[idx])
            ) * self.config["scale"]
        elif self.config["shaping_method"] == "zscore":
            valid_delay_rew = (
                torch.from_numpy(self.delay_rew[idx]) - self.return_mean
            ) / (self.return_std)
        else:
            valid_delay_rew = (
                torch.from_numpy(self.delay_rew[idx])
                - self.return_min / valid_length
            ) / (self.return_max - self.return_min)

        delay_rew = torch.zeros((self.max_length,))
        delay_rew[:valid_length] = valid_delay_rew
        # print(f"norm delay_rew sum:{delay_rew.sum()},length:{valid_length},delay_rew_sum:{self.delay_rew[idx].sum()},return max:{self.return_max},return min:{self.return_min}")
        # if(delay_rew.sum()>1):
        #   exit(1)
        sample["delay_reward"] = delay_rew

        sample["length"] = valid_length
        return sample


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


def plot_dcompose_reward(exp_name, raw_rewards, delay_rewards):
    import matplotlib.pyplot as plt

    plt.cla()
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
    plt.legend(["Raw", "Dcomposed"])

    fig_dir = f"{proj_path}/assets/{exp_name}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(f"{fig_dir}/{exp_name}_reward.png")


def normalize_decomposed_reward_dataset(task, delay, shaping_method):
    BATCH_SIZE = 1
    device = "cpu"
    cpu_device = "cpu"
    dataset = TrajDataset(
        {
            "delay": delay,
            "task": task,
            "delay_mode": "constant",
            "decomposed": True,
            "shaping_method": shaping_method,
        }
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    new_dataset = {}
    new_dataset["observations"] = []
    new_dataset["actions"] = []
    new_dataset["terminals"] = []
    new_dataset["rewards"] = []
    new_dataset["next_observations"] = []
    for _, s in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            delay_reward = s["delay_reward"].to(device)
            new_dataset["observations"].append(
                s["observations"].squeeze(dim=0)
            )
            new_dataset["actions"].append(s["actions"].squeeze(dim=0))
            new_dataset["terminals"].append(s["terminals"].squeeze(dim=0))
            new_dataset["next_observations"].append(
                s["next_observations"].squeeze(dim=0)
            )
            new_dataset["rewards"].append(
                (delay_reward).squeeze(dim=0)[..., : s["length"][0]]
            )
    new_dataset["observations"] = torch.cat(new_dataset["observations"], dim=0)
    new_dataset["actions"] = torch.cat(new_dataset["actions"], dim=0)
    new_dataset["terminals"] = torch.cat(new_dataset["terminals"], dim=0)
    new_dataset["next_observations"] = torch.cat(
        new_dataset["next_observations"], dim=0
    )
    new_dataset["rewards"] = torch.cat(new_dataset["rewards"], dim=0)
    print("new_dataset_rewards:", new_dataset["rewards"].shape)
    return new_dataset


def random_decomposed_reward_dataset(task, delay):
    BATCH_SIZE = 1
    device = "cpu"
    cpu_device = "cpu"
    dataset = TrajDataset(
        {
            "delay": delay,
            "task": task,
            "delay_mode": "constant",
            "decomposed": True,
            "scale": 0.01,
        }
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    obs_act_pair_dim = dataset.act_size + dataset.obs_size
    model = RandomNetRewardDecomposer(dataset.obs_size, 512).to(device)
    new_dataset = {}
    new_dataset["observations"] = []
    new_dataset["actions"] = []
    new_dataset["terminals"] = []
    new_dataset["rewards"] = []
    new_dataset["next_observations"] = []
    for _, s in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            reward_pre = model(
                s["obs"].to(device), s["last_obs"].to(device)
            ).squeeze(dim=-1)
            delay_reward = s["delay_reward"].to(device)
            # print("obs dtype", s["obs"].shape)
            # print("last obs dtype", s["last_obs"].shape)
            # print("reward_pre:", reward_pre.shape)
            # print("delay_rew:", delay_reward.shape)
            new_dataset["observations"].append(
                s["observations"].squeeze(dim=0)
            )
            new_dataset["actions"].append(s["actions"].squeeze(dim=0))
            new_dataset["terminals"].append(s["terminals"].squeeze(dim=0))
            new_dataset["next_observations"].append(
                s["next_observations"].squeeze(dim=0)
            )
            new_dataset["rewards"].append(
                (reward_pre + delay_reward).squeeze(dim=0)[
                    ..., : s["length"][0]
                ]
            )
    new_dataset["observations"] = torch.cat(new_dataset["observations"], dim=0)
    new_dataset["actions"] = torch.cat(new_dataset["actions"], dim=0)
    new_dataset["terminals"] = torch.cat(new_dataset["terminals"], dim=0)
    new_dataset["next_observations"] = torch.cat(
        new_dataset["next_observations"], dim=0
    )
    new_dataset["rewards"] = torch.cat(new_dataset["rewards"], dim=0)
    print("new_dataset_rewards:", new_dataset["rewards"].shape)
    return new_dataset


def transformer_decomposed_reward_dataset(task, delay):
    BATCH_SIZE = 1
    device = "cuda:0"
    cpu_device = "cpu"
    dataset = TrajDataset(
        {
            "delay": delay,
            "task": task,
            "delay_mode": "constant",
            "decomposed": True,
            "scale": 0.01,
        }
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    obs_act_pair_dim = dataset.act_size + dataset.obs_size
    print(f"task:{task}")
    model = torch.load(f"../datasets/transformer-{task}.ckpt").to(device)
    # model = torch.load(f"../datasets/transformer.ckpt").to(device)
    new_dataset = {}
    new_dataset["observations"] = []
    new_dataset["actions"] = []
    new_dataset["terminals"] = []
    new_dataset["rewards"] = []
    new_dataset["next_observations"] = []
    for _, s in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            key_padding_mask = create_key_padding_mask(
                s["length"], dataset.max_length
            ).to(device)
            reward_pre = model(
                s["obs_act_pair"].to(device), key_padding_mask=key_padding_mask
            ).squeeze(dim=-1)
            delay_reward = s["delay_reward"].to(device)
            new_dataset["observations"].append(
                s["observations"].squeeze(dim=0)
            )
            new_dataset["actions"].append(s["actions"].squeeze(dim=0))
            new_dataset["terminals"].append(s["terminals"].squeeze(dim=0))
            new_dataset["next_observations"].append(
                s["next_observations"].squeeze(dim=0)
            )
            reward_redistribution = (
                reward_redistributed(reward_pre, delay_reward, s["length"])
                .to(cpu_device)
                .squeeze(dim=0)
            )
            new_dataset["rewards"].append(
                reward_redistribution[..., : s["length"][0]]
            )
    new_dataset["observations"] = torch.cat(new_dataset["observations"], dim=0)
    new_dataset["actions"] = torch.cat(new_dataset["actions"], dim=0)
    new_dataset["terminals"] = torch.cat(new_dataset["terminals"], dim=0)
    new_dataset["next_observations"] = torch.cat(
        new_dataset["next_observations"], dim=0
    )
    new_dataset["rewards"] = torch.cat(new_dataset["rewards"], dim=0)
    return new_dataset


def pg_shaping_reward_dataset(
    config, model_path=None, normalize_type="min_max"
):
    dataset = trans_dataset(config)
    for k, v in dataset.items():
        dataset[k] = torch.from_numpy(v)

    if model_path is None:
        model_path = "../logs/d4rl-walker2d-medium-replay-v0-delay_mode-constant-delay-100--reward_shaper-random-v2-seed-2021_2021-09-09_22-08-16-394/models/99.pt"
    shaping_model = torch.load(model_path).to("cpu")

    with torch.no_grad():
        residual_reward = shaping_model(
            dataset["next_observations"]
        ) - shaping_model(dataset["observations"])
        residual_reward.squeeze_(-1)
    print("residual rewards:", residual_reward.shape)
    print("rewards:", dataset["rewards"].shape)
    dataset["rewards"] = dataset["rewards"] + residual_reward
    # returns = dataset["returns"]
    returns = dataset["rewards"]

    # min-max
    if normalize_type == "min_max":
        min_ret = np.min(returns)
        max_ret = np.max(returns)
        dataset["rewards"] = (dataset["rewards"] - min_ret) / (
            max_ret - min_ret + 1e-6
        )

    # z_score
    elif normalize_type == "z_score":
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        dataset["rewards"] = (dataset["rewards"] - mean_ret) / (std_ret + 1e-6)
    return dataset


def load_decomposed_d4rl_buffer(config):
    task = config["task"][5:] if "d4rl" in config["task"] else config["task"]
    if "shaping_method" in config and config["shaping_method"] == "random":
        logger.info("use random net decomposer")
        dataset = random_decomposed_reward_dataset(
            task,
            config["delay"],
        )
    elif config["shaping_method"] == "minmax":
        logger.info("use minmax decomposer")
        dataset = normalize_decomposed_reward_dataset(
            task, config["delay"], config["shaping_method"]
        )
    elif config["shaping_method"] == "zscore":
        logger.info("use zscore decomposer")
        dataset = normalize_decomposed_reward_dataset(
            task, config["delay"], config["shaping_method"]
        )
    elif config["shaping_method"] == "pg":
        logger.info("use policy gradient decomposer")
        dataset = pg_shaping_reward_dataset(config)
    else:
        logger.info("use transformer decomposer")
        dataset = transformer_decomposed_reward_dataset(
            task,
            config["delay"],
        )

    buffer = SampleBatch(
        obs=dataset["observations"].numpy(),
        obs_next=dataset["next_observations"].numpy(),
        act=dataset["actions"].numpy(),
        rew=np.expand_dims(np.squeeze(dataset["rewards"].numpy()), 1),
        done=np.expand_dims(np.squeeze(dataset["terminals"].numpy()), 1),
    )

    logger.info("obs shape: {}", buffer.obs.shape)
    logger.info("obs_next shape: {}", buffer.obs_next.shape)
    logger.info("act shape: {}", buffer.act.shape)
    logger.info("rew shape: {}", buffer.rew.shape)
    logger.info("done shape: {}", buffer.done.shape)
    logger.info("Episode reward: {}", buffer.rew.sum() / np.sum(buffer.done))
    logger.info("Number of terminals on: {}", np.sum(buffer.done))
    return buffer


if __name__ == "__main__":
    task = "hopper-medium-replay-v0"
    delay = 100
    # decomposed_reward_dataset(task,delay)
    BATCH_SIZE = 16
    device = "cuda:1"
    cpu_device = "cpu"
    dataset = TrajDataset(
        {"delay": delay, "task": task, "delay_mode": "constant"}
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    obs_act_pair_dim = dataset.act_size + dataset.obs_size
    model = TransformerRewardDecomposer(obs_act_pair_dim, 512).to(device)
    rew_gap = dataset.return_max - dataset.return_min

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(2000):
        for batch, s in enumerate(dataloader):
            key_padding_mask = create_key_padding_mask(
                s["length"], dataset.max_length
            ).to(device)
            reward_pre = model(
                s["obs_act_pair"].to(device), key_padding_mask=key_padding_mask
            ).squeeze(dim=-1)
            reward_mask = torch.where(
                key_padding_mask.view(len(s["length"]), dataset.max_length, 1)
                == 0,
                1,
                0,
            )
            delay_reward = s["delay_reward"].to(device)
            returns = delay_reward.sum(dim=-1)
            main_loss = (
                torch.mean(
                    reward_pre[range(len(s["length"])), s["length"] - 1]
                    - returns[:, None]
                )
                ** 2
            )
            aux_loss = torch.mean(reward_pre - returns[..., None]) ** 2
            loss = main_loss + aux_loss * 0.5
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"batch:{batch},loss:{loss.item()}")
            if batch > 0 and batch == 10:
                torch.save(model, f"transformer-{task}.ckpt")
                with torch.no_grad():
                    reward_redistribution = reward_redistributed(
                        reward_pre, delay_reward, s["length"]
                    ).to(cpu_device)
                    plot_dcompose_reward(
                        task,
                        (s["reward"][0])[..., : s["length"][0]],
                        reward_redistribution[0][..., : s["length"][0]],
                    )

    print(1)
