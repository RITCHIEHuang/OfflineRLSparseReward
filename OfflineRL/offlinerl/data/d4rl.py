import os
import pickle

import gym
import d4rl
import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch


def load_d4rl_buffer(task):
    task = task[5:]
    env = gym.make(task)
    dataset = d4rl.qlearning_dataset(env)
    act_limit = 1 - 1e-5
    buffer = SampleBatch(
        obs=dataset["observations"],
        obs_next=dataset["next_observations"],
        act=np.clip(dataset["actions"], -act_limit, act_limit),
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
