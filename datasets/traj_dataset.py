import os
import torch
from torch.utils import data
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

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
        valid_delay_rew_zscore = (
            torch.from_numpy(self.delay_rew[idx])
            - self.return_mean / valid_length
        ) / (self.return_std)
        valid_delay_rew_minmax = (
            torch.from_numpy(self.delay_rew[idx])
            - self.return_min / valid_length
        ) / (
            self.return_max - self.return_min
        )  # - 0.5 / valid_length

        if "scale" in self.config:
            valid_delay_rew = (
                torch.from_numpy(self.delay_rew[idx])
            ) * self.config["scale"]
        elif self.config["shaping_method"] == "zscore":
            valid_delay_rew = valid_delay_rew_zscore

        else:
            valid_delay_rew = valid_delay_rew_minmax

        zscore_rew = torch.zeros((self.max_length,))
        zscore_rew[:valid_length] = valid_delay_rew_zscore
        minmax_rew = torch.zeros((self.max_length,))
        minmax_rew[:valid_length] = valid_delay_rew_minmax

        delay_rew = torch.zeros((self.max_length,))
        delay_rew[:valid_length] = valid_delay_rew
        # print(f"norm delay_rew sum:{delay_rew.sum()},length:{valid_length},delay_rew_sum:{self.delay_rew[idx].sum()},return max:{self.return_max},return min:{self.return_min}")
        # if(delay_rew.sum()>1):
        #   exit(1)
        sample["delay_reward"] = delay_rew
        sample["zscore_reward"] = zscore_rew
        sample["minmax_reward"] = minmax_rew

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


def plot_dcompose_reward_sns(
    exp_name,
    data_list,
    length,
    names=["raw", "decomposed"],
    suffix="",
):
    plt.cla()
    concated_data_list = []
    for data in data_list:
        concated_data_list.append(np.reshape(data, [length, 1]))
    r = np.concatenate(concated_data_list, 1)
    df = pd.DataFrame(r, columns=names)
    df["steps"] = list(range(length))
    df = df.melt("steps", var_name="cols", value_name="vals")
    sns_plot = sns.lineplot(
        data=df, x="steps", y="vals", hue="cols"
    ).set_title(exp_name)
    fig_dir = f"{proj_path}/assets/{exp_name}"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    sns_plot.figure.savefig(f"{fig_dir}/{exp_name}_reward_{suffix}.png")


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

    plot_traj_idx_list = [np.random.randint(0, len(dataset)) for _ in range(5)]
    for traj_idx, s in tqdm(enumerate(dataloader)):
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
            plot_dcompose_reward_sns(
                task + "_" + "minmax_zscore",
                [
                    (s["zscore_reward"][0])[..., : s["length"][0]].numpy().T,
                    s["minmax_reward"][0][..., : s["length"][0]].numpy().T,
                ],
                s["length"][0],
                names=["zscore", "minmax"],
                suffix=traj_idx,
            )
            plot_dcompose_reward_sns(
                task + "_" + "minmax",
                [
                    (s["reward"][0])[..., : s["length"][0]].numpy().T,
                    s["minmax_reward"][0][..., : s["length"][0]].numpy().T,
                ],
                s["length"][0],
                names=["raw", "minmax"],
                suffix=traj_idx,
            )
            plot_dcompose_reward_sns(
                task + "_" + "zscore",
                [
                    (s["reward"][0])[..., : s["length"][0]].numpy().T,
                    s["zscore_reward"][0][..., : s["length"][0]].numpy().T,
                ],
                s["length"][0],
                names=["raw", "zscore"],
                suffix=traj_idx,
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

    plot_traj_idx_list = [np.random.randint(0, len(dataset)) for _ in range(5)]
    for traj_idx, s in tqdm(enumerate(dataloader)):
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
    config, model_path=None, normalize_type="min_max", model_phase="init"
):
    model_mapping = {
        "init": "../logs/d4rl-walker2d-medium-replay-v0-delay_mode-constant-delay-100--reward_shaper-bc-v2-seed-2021_2021-09-15_15-29-42-068/models/0.pt",
        "trained": "../logs/d4rl-walker2d-medium-replay-v0-delay_mode-constant-delay-100--reward_shaper-bc-v2-seed-2021_2021-09-15_15-29-42-068/models/270.pt",
    }

    if model_path is None:
        model_path = model_mapping[model_phase]
    shaping_model = torch.load(model_path).to("cpu")
    trained_shaping_model = torch.load(model_mapping["trained"]).to("cpu")

    BATCH_SIZE = 1
    device = "cpu"
    cpu_device = "cpu"
    dataset = TrajDataset(
        {
            "delay": config["delay"],
            "task": config["task"],
            "delay_mode": config["delay_mode"],
            "decomposed": True,
            "shaping_method": normalize_type,
        }
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    new_dataset = {}
    new_dataset["observations"] = []
    new_dataset["actions"] = []
    new_dataset["terminals"] = []
    new_dataset["rewards"] = []
    new_dataset["next_observations"] = []

    plot_traj_idx_list = [np.random.randint(0, len(dataset)) for _ in range(5)]

    for traj_idx, s in tqdm(enumerate(dataloader)):
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

            residual_reward = shaping_model(
                s["next_obs"].to(device)
            ) - shaping_model(s["obs"].to(device))
            residual_reward.squeeze_(-1)
            new_reward = delay_reward + residual_reward

            # trained_reward
            new_residual = trained_shaping_model(
                s["next_obs"].to(device)
            ) - trained_shaping_model(s["obs"].to(device))
            new_residual.squeeze_(-1)
            trained_reward = delay_reward + new_residual

            new_dataset["rewards"].append(
                (new_reward).squeeze(dim=0)[..., : s["length"][0]]
            )

            if traj_idx in plot_traj_idx_list:
                plot_dcompose_reward_sns(
                    config["task"] + "_" + f"pg_{normalize_type}",
                    [
                        s["minmax_reward"][0][..., : s["length"][0]].numpy().T,
                        new_dataset["rewards"][-1].numpy().T,
                        (trained_reward)
                        .squeeze(dim=0)[..., : s["length"][0]]
                        .numpy()
                        .T,
                    ],
                    length=s["length"][0],
                    names=["raw", "decomposed_init", "decomposed_trained"],
                    suffix=f"{traj_idx}_{model_phase}",
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
    exit(0)
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
