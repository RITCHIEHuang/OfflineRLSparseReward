from copy import deepcopy
from collections import OrderedDict

import numpy as np
from d4rl.infos import REF_MIN_SCORE, REF_MAX_SCORE

from offlinerl.utils.env import get_env


D4RL_REF_MIN_SCORE = deepcopy(REF_MIN_SCORE)
D4RL_REF_MAX_SCORE = deepcopy(REF_MAX_SCORE)

for env in ["HalfCheetah", "Hopper", "Walker2d", "Ant"]:
    for dset in ["low", "medium", "high", "expert"]:
        for train_num in [10, 100, 1000]:
            dset_name = env + "-v3" + "-" + dset + "-" + str(train_num)
            D4RL_REF_MIN_SCORE[dset_name] = REF_MIN_SCORE[
                env.lower() + "-random-v0"
            ]
            D4RL_REF_MAX_SCORE[dset_name] = REF_MAX_SCORE[
                env.lower() + "-random-v0"
            ]


def d4rl_score(task, rew_mean, len_mean):
    split_list = task.split("-")
    domain = split_list[0]
    task = task[len(domain) + 1 :]

    score = (
        (rew_mean - D4RL_REF_MIN_SCORE[task])
        / (D4RL_REF_MAX_SCORE[task] - D4RL_REF_MIN_SCORE[task])
        * 100
    )
    return score


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


def d4rl_eval_fn(task, eval_episodes=100):
    env = get_env(task)

    # def d4rl_eval(policy):
    #     results = []
    #     for i in range(eval_episodes):
    #         results.append(run_episode((env, policy, i)))

    #     rew_mean = np.mean(list(map(lambda x: x[0], results)))
    #     len_mean = np.mean(list(map(lambda x: x[1], results)))

    #     score = d4rl_score(task, rew_mean, len_mean)

    #     res = OrderedDict()
    #     res["Reward_Mean"] = rew_mean
    #     res["Length_Mean"] = len_mean
    #     res["D4rl_Score"] = score

    #     return res

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


neorl_scores = {
    "HalfCheetah-v3-low-10": 3260,
    "HalfCheetah-v3-low-100": 3200,
    "Hopper-v3-low-10": 515,
    "Hopper-v3-low-100": 514,
    "Walker2d-v3-low-10": 1749,
    "Walker2d-v3-low-100": 1433,
}

for k, v in neorl_scores.items():
    print(k, round(d4rl_score("neorl-" + k, v, None), 1))
