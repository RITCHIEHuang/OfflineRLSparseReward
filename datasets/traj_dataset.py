from offlinerl.utils.data import sample
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from loguru import logger
import numpy as np

from datasets.d4rl_dataset import trans_traj_dataset
from algos.reward_decoposer import (
    TransformerRewardDecomposer,
    create_key_padding_mask,
)
from offlinerl.utils.data import SampleBatch


class TrajDataset(Dataset):
    def __init__(self, config) -> None:
        self.config = config

        traj_dataset = trans_traj_dataset(config)

        self.obs = traj_dataset["observations"]
        self.act = traj_dataset["actions"]
        self.delay_rew = traj_dataset["delay_rewards"]
        self.rew = traj_dataset["rewards"]
        self.done = traj_dataset["terminals"]
        self.next_obs = traj_dataset["next_observations"]
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

        sample["terminals"]=self.done[idx]
        sample["actions"]=self.act[idx]
        sample["observations"]=self.obs[idx]
        sample["next_observations"]=self.next_obs[idx]
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
    for i,l in enumerate(length):
        redistributed_reward[i,l:]=0
    predicted_returns = redistributed_reward.sum(dim=1)
    prediction_error = returns - predicted_returns
    for i,l in enumerate(length):
        redistributed_reward[i,:l]+=prediction_error[i,None]/l
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

def decomposed_reward_dataset(task):
    BATCH_SIZE = 1
    device = "cuda:0"
    cpu_device = 'cpu'
    dataset = TrajDataset({"delay": 20, "task": task,"delay_mode":"constant"})
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    obs_act_pair_dim = dataset.act_size + dataset.obs_size
    model = torch.load("../datasets/transformer.ckpt").to(device)
    new_dataset = {}
    new_dataset["observations"]=[]
    new_dataset["actions"]=[]
    new_dataset["terminals"]=[]
    new_dataset["rewards"]=[]
    new_dataset["next_observations"]=[]
    for _, s in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            key_padding_mask = create_key_padding_mask(
                s["length"], dataset.max_length
            ).to(device)
            reward_pre = model(
                s["obs_act_pair"].to(device), key_padding_mask=key_padding_mask
            ).squeeze(dim=-1)
            delay_reward = s['delay_reward'].to(device)
            new_dataset["observations"].append(s["observations"].squeeze(dim=0))
            new_dataset["actions"].append(s["actions"].squeeze(dim=0))
            new_dataset["terminals"].append(s["terminals"].squeeze(dim=0))
            new_dataset["next_observations"].append(s["next_observations"].squeeze(dim=0))
            reward_redistribution = reward_redistributed(reward_pre,delay_reward,s["length"]).to(cpu_device).squeeze(dim=0)
            new_dataset["rewards"].append(reward_redistribution[...,:s["length"][0]])
    new_dataset["observations"]=torch.cat(new_dataset["observations"],dim=0)
    new_dataset["actions"]=torch.cat(new_dataset["actions"],dim=0)
    new_dataset["terminals"]=torch.cat(new_dataset["terminals"],dim=0)
    new_dataset["next_observations"]=torch.cat(new_dataset["next_observations"],dim=0)
    new_dataset["rewards"]=torch.cat(new_dataset["rewards"],dim=0)
    return new_dataset
    
def load_decomposed_d4rl_buffer(config):
    dataset = decomposed_reward_dataset(
        config["task"][5:] if "d4rl" in config["task"] else config["task"]
    )
    buffer = SampleBatch(
        obs=dataset["observations"],
        obs_next=dataset["next_observations"],
        act=dataset["actions"],
        rew=np.expand_dims(np.squeeze(dataset["rewards"]), 1),
        done=np.expand_dims(np.squeeze(dataset["terminals"]), 1),
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
    task = "walker2d-expert-v0"
    decomposed_reward_dataset(task)
    # BATCH_SIZE = 16
    # device = "cuda:0"
    # cpu_device = 'cpu'
    # dataset = TrajDataset({"delay": 20, "task": task})
    # dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # obs_act_pair_dim = dataset.act_size + dataset.obs_size
    # model = TransformerRewardDecomposer(obs_act_pair_dim, 512).to(device)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn_1 = torch.nn.MSELoss()
    # rew_gap = dataset.rew_max-dataset.rew_min

    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    # for epoch in range(2000):
    #     for batch, s in enumerate(dataloader):
    #         key_padding_mask = create_key_padding_mask(
    #             s["length"], dataset.max_length
    #         ).to(device)
    #         reward_pre = model(
    #             s["obs_act_pair"].to(device), key_padding_mask=key_padding_mask
    #         ).squeeze(dim=-1)
    #         reward_mask = torch.where(
    #             key_padding_mask.view(len(s["length"]), dataset.max_length, 1)
    #             == 0,
    #             1,
    #             0,
    #         )
    #         delay_reward = s['delay_reward'].to(device)
    #         returns = delay_reward.sum(dim=-1)
    #         main_loss = torch.mean(reward_pre[range(len(s["length"])),s["length"]-1]-returns[:,None])**2
    #         aux_loss = torch.mean(reward_pre-delay_reward)**2
    #         loss = main_loss + aux_loss
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #         print(f"batch:{batch},loss:{loss.item()}")
    #         if batch>0 and batch % 100==0:
    #             with torch.no_grad():
    #                 import pandas as pd
    #                 reward_redistribution = reward_redistributed(reward_pre,delay_reward,s["length"]).to(cpu_device)
    #                 plot_dcompose_reward(task, s['reward'][0],reward_redistribution[0])

    print(1)
