from collections import OrderedDict

import numpy as np

from offlinerl.utils.env import get_env


def run_episode(args):
    env, policy, _ = args
    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state)
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1
    return rewards, lengths


def gym_eval_fn(task, eval_episodes=100):
    env = get_env(task)

    def d4rl_eval(policy):
        episode_rewards = []
        episode_lengths = []
        for _ in range(eval_episodes):
            state, done = env.reset(), False
            rewards = 0
            lengths = 0
            while not done:
                state = state[np.newaxis]
                action = policy.get_action(state)
                state, reward, done, _ = env.step(action[0])
                rewards += reward
                lengths += 1

            episode_rewards.append(rewards)
            episode_lengths.append(lengths)

        rew_mean = np.mean(episode_rewards)
        len_mean = np.mean(episode_lengths)

        res = OrderedDict()
        res["Reward_Mean"] = rew_mean
        res["Length_Mean"] = len_mean

        return res

    return d4rl_eval
