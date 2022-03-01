from collections import OrderedDict

import numpy as np

from offlinerl.utils.env import get_env
from offlinerl.utils.atari_utils import make_pytorch_env


def atari_eval_fn(task, max_episode_steps=27000, num_eval_steps=10000):
    env = make_pytorch_env(
        get_env(task), episode_life=False, clip_rewards=False, scale=True
    )

    def atari_eval(policy):
        episode_rewards = []
        episode_lengths = []
        num_steps = 0

        while True:
            state = env.reset()
            episode_steps = 0
            episode_return = 0.0
            done = False
            while (not done) and episode_steps <= max_episode_steps:
                state = state[np.newaxis]
                action = policy.get_action(state).item()
                next_state, reward, done, _ = env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward
                state = next_state

            if num_steps > num_eval_steps:
                break

            episode_lengths.append(episode_steps)
            episode_rewards.append(episode_return)

        rew_mean = np.mean(episode_rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean"] = rew_mean
        res["Length_Mean"] = len_mean

        return res

    return atari_eval
