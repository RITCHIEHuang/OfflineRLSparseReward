import os
import gym
import numpy as np

import collections
import pickle

import rec_env

datasets = []

for env_name in ["recs"]:
    for dataset_type in ["random"]:
        for dataset_size in ["large"]:
            name = f"{env_name}-{dataset_type}-{dataset_size}-v0"
            if os.path.exists(f"{name}.pkl"):
                print("Dataset exists!!!")
                continue
            env = gym.make(name)
            dataset = env.get_dataset()

            N = dataset["rewards"].shape[0]
            data_ = collections.defaultdict(list)

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset["terminals"][i])

                for k in [
                    "observations",
                    "next_observations",
                    "actions",
                    "rewards",
                    "terminals",
                ]:
                    data_[k].append(dataset[k][i])
                if done_bool:
                    episode_step = 0
                    episode_data = {}
                    for k in data_:
                        episode_data[k] = np.array(data_[k])
                    paths.append(episode_data)
                    data_ = collections.defaultdict(list)
                episode_step += 1

            returns = np.array([np.sum(p["rewards"]) for p in paths])
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            print(f"Number of samples collected: {num_samples}")
            print(
                f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
            )

            with open(f"{name}.pkl", "wb") as f:
                pickle.dump(paths, f)
