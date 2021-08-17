import torch

from torch.utils.data import Dataset, DataLoader
from loguru import logger
import numpy as np

from datasets.d4rl_dataset import trans_traj_dataset
from algos.reward_decoposer import (
    TransformerRewardDecomposer,
    create_key_padding_mask,
)


class TrajDataset(Dataset):
    def __init__(self, config) -> None:
        self.config = config

        traj_dataset = trans_traj_dataset(config)

        self.obs = traj_dataset["observations"]
        self.act = traj_dataset["actions"]
        self.delay_rew = traj_dataset["delay_rewards"]
        self.rew = traj_dataset["rewards"]
        rew = np.array([x.sum() for x in self.rew])
        self.rew_max = np.max(rew)
        self.rew_min = np.min(rew)
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
        valid_rew = (torch.from_numpy(self.rew[idx])-self.rew_min)/(self.rew_max-self.rew_min)
        rew = torch.zeros((self.max_length,))
        rew[:valid_length] = valid_rew
        sample["reward"] = rew

        valid_delay_rew = (torch.from_numpy(self.delay_rew[idx])-self.rew_min)/(self.rew_max-self.rew_min)
        delay_rew = torch.zeros((self.max_length,))
        delay_rew[:valid_length] = valid_delay_rew
        sample["delay_reward"] = delay_rew

        sample["length"] = valid_length
        return sample


def reward_redistributed(predictions,rewards,length):
    redistributed_reward = predictions[..., 1:] - predictions[..., :-1]
    redistributed_reward = torch.cat([predictions[..., :1], redistributed_reward], dim=1)
    returns = rewards.sum(dim=1)
    predicted_returns = redistributed_reward.sum(dim=1)
    prediction_error = returns - predicted_returns
    redistributed_reward += prediction_error[..., None] / redistributed_reward.shape[1]
    return redistributed_reward

def plot_dcompose_reward(exp_name,raw_rewards, delay_rewards):
    import matplotlib.pyplot as plt
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
    plt.savefig(f"{exp_name}_reward.png")

if __name__ == "__main__":
    BATCH_SIZE = 16
    device = "cuda:0"
    cpu_device = 'cpu'
    task = "walker2d-expert-v0"
    dataset = TrajDataset({"delay": 20, "task": task})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    obs_act_pair_dim = dataset.act_size + dataset.obs_size
    model = TransformerRewardDecomposer(obs_act_pair_dim, 512).to(device)
    loss_fn = torch.nn.MSELoss()
    loss_fn_1 = torch.nn.MSELoss()
    rew_gap = dataset.rew_max-dataset.rew_min

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
            delay_reward = s['delay_reward'].to(device)
            returns = delay_reward.sum(dim=-1)
            main_loss = torch.mean(reward_pre[range(len(s["length"])),s["length"]-1]-returns[:,None])**2
            aux_loss = torch.mean(reward_pre-delay_reward)**2
            loss = main_loss + aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"batch:{batch},loss:{loss.item()}")
            if batch>0 and batch % 100==0:
                with torch.no_grad():
                    import pandas as pd
                    reward_redistribution = reward_redistributed(reward_pre,delay_reward,None).to(cpu_device)
                    plot_dcompose_reward(task, s['reward'][0],reward_redistribution[0])

    print(1)
