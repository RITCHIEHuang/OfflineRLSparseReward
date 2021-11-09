from copy import deepcopy

import numpy as np
from collections import OrderedDict
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

from offlinerl.utils.env import get_env


Mujoco_REF_MIN_SCORE = deepcopy(REF_MIN_SCORE)
Mujoco_REF_MAX_SCORE = deepcopy(REF_MAX_SCORE)

for env in ["HalfCheetah", "Hopper", "Walker2d", "Ant"]:
    for dset in ["low", "medium", "high", "expert"]:
        for train_num in [10, 100, 1000]:
            dset_name = env + "-v3" + "-" + dset + "-" + str(train_num)
            Mujoco_REF_MIN_SCORE[dset_name] = REF_MIN_SCORE[
                env.lower() + "-random-v0"
            ]
            Mujoco_REF_MAX_SCORE[dset_name] = REF_MAX_SCORE[
                env.lower() + "-random-v0"
            ]


def d4rl_score(task, rew_mean, len_mean):
    split_list = task.split("-")
    domain = split_list[0]
    task = task[len(domain) + 1 :]

    score = (
        (rew_mean - Mujoco_REF_MIN_SCORE[task])
        / (Mujoco_REF_MAX_SCORE[task] - Mujoco_REF_MIN_SCORE[task])
        * 100
    )
    return score


def d4rl_eval_fn(task, eval_episodes=100):
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
                state, reward, done, _ = env.step(action)
                rewards += reward
                lengths += 1

            episode_rewards.append(rewards)
            episode_lengths.append(lengths)

        rew_mean = np.mean(episode_rewards)
        len_mean = np.mean(episode_lengths)

        score = d4rl_score(task, rew_mean, len_mean)

        res = OrderedDict()
        res["Reward_Mean"] = rew_mean
        res["Length_Mean"] = len_mean
        res["D4rl_Score"] = score

        return res

    return d4rl_eval
