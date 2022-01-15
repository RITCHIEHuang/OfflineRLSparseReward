from copy import deepcopy
import time
from multiprocessing import Pool

import numpy as np

from offlinerl.utils.env import get_env
from offlinerl.utils.net.common import MLP
from offlinerl.utils.net.discrete import QPolicyWrapper


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
        start_time = time.time()
        for i in range(eval_episodes):
            results.append(run_episode((env, policy, i)))

        rew_mean = np.mean(list(map(lambda x: x[0], results)))
        len_mean = np.mean(list(map(lambda x: x[1], results)))
        retention_mean = np.mean(list(map(lambda x: x[2], results)))
        click_mean = np.mean(list(map(lambda x: x[3], results)))

        print("reward_mean", rew_mean)
        print("len_mean", len_mean)
        print("retention_mean", retention_mean)
        print("click_mean", click_mean)
        end_time = time.time()
        print("Timing", end_time - start_time)

    return recs_eval


def recs_eval_fn_mp(task, eval_episodes=100):
    env = get_env(task)

    def recs_eval(policy):
        cpu_policy = deepcopy(policy)
        cpu_policy.to("cpu")
        start_time = time.time()
        args_list = [
            (deepcopy(env), deepcopy(cpu_policy), i)
            for i in range(eval_episodes)
        ]
        p = Pool(8)
        results = p.map_async(run_episode, args_list)
        results = results.get()
        p.close()

        rew_mean = np.mean(list(map(lambda x: x[0], results)))
        len_mean = np.mean(list(map(lambda x: x[1], results)))
        retention_mean = np.mean(list(map(lambda x: x[2], results)))
        click_mean = np.mean(list(map(lambda x: x[3], results)))
        print("reward_mean", rew_mean)
        print("len_mean", len_mean)
        print("retention_mean", retention_mean)
        print("click_mean", click_mean)
        end_time = time.time()
        print("Timing", end_time - start_time)

    return recs_eval


if __name__ == "__main__":
    policy = QPolicyWrapper(MLP(120, 10, 256, 2)).to("cuda:7")
    # recs_eval_fn("recs-recs-random-v0")(policy)
    recs_eval_fn_mp("recs-recs-random-v0")(policy)
