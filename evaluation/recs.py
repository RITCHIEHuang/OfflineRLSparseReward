from copy import deepcopy

import numpy as np
from collections import OrderedDict
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

for env in ["recs"]:
    for dset in ["", "random", "replay"]:
        if len(dset) != 0:
            dset_name = env + "-" + dset + "-v0"
        else:
            dset_name = env + "-v0"
        D4RL_REF_MIN_SCORE[dset_name] = 0.0
        D4RL_REF_MAX_SCORE[dset_name] = 9.0


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
    retentions = 0
    clicks = 0
    lengths = 0
    while not done:
        state = state[np.newaxis]
        action = policy.get_action(state)
        state, reward, done, info = env.step(action)
        rewards += reward
        retentions += info["reward"]["retention"]
        clicks += info["reward"]["click"]
        lengths += 1
    return (rewards, lengths, retentions, clicks)


def recs_eval_fn(task, eval_episodes=100):
    env = get_env(task)
    results = []

    def recs_eval(policy):
        for i in range(eval_episodes):
            results.append(run_episode((env, policy, i)))

        rew_mean = np.mean(list(map(lambda x: x[0], results)))
        len_mean = np.mean(list(map(lambda x: x[1], results)))
        retention_mean = np.mean(list(map(lambda x: x[2], results)))
        click_mean = np.mean(list(map(lambda x: x[3], results)))

        score = d4rl_score(task, rew_mean, len_mean)

        res = OrderedDict()
        res["Reward_Mean"] = rew_mean
        res["Retention_Mean"] = retention_mean
        res["Click_Mean"] = click_mean
        res["Length_Mean"] = len_mean
        res["D4rl_Score"] = score

        return res

    return recs_eval


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
