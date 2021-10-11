import torch
import numpy as np
from loguru import logger

from torch.utils.data import Dataset


class TrajDataset(Dataset):
    def __init__(self, traj_dataset) -> None:
        self.obs = traj_dataset["observations"]
        self.act = traj_dataset["actions"]
        self.delay_rew = traj_dataset["delay_rewards"]
        self.rew = traj_dataset["rewards"]
        self.done = traj_dataset["terminals"]
        self.next_obs = traj_dataset["next_observations"]

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

        sample["obs"] = torch.zeros((self.max_length, self.obs_size))
        sample["obs"][:valid_length] = torch.from_numpy(self.obs[idx])

        sample["next_obs"] = torch.zeros((self.max_length, self.obs_size))
        sample["next_obs"][:valid_length] = torch.from_numpy(
            self.next_obs[idx]
        )

        sample["act"] = torch.zeros((self.max_length, self.act_size))
        sample["act"][:valid_length] = torch.from_numpy(self.act[idx])

        sample["reward"] = torch.zeros((self.max_length,))
        sample["reward"][:valid_length] = torch.from_numpy(self.rew[idx])

        sample["length"] = valid_length

        return sample
