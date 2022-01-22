import os
import gym
import numpy as np

import collections
import pickle

import neorl


datasets = []

for env_name in ["HalfCheetah-v3", "Hopper-v3", "Walker2d-v3"]:
    for dataset_type in ["low"]:
        for dataset_num in [10, 100]:
            name = f"{env_name}-{dataset_type}-{dataset_num}"
            if os.path.exists(f"{name}.pkl"):
                print("Dataset exists!!!")
                continue
            env = neorl.make(env_name)
            dataset, _ = env.get_dataset(
                data_type=dataset_type, train_num=dataset_num, need_val=False
            )

            N = dataset["reward"].shape[0]
            data_ = collections.defaultdict(list)

            episode_step = 0
            paths = []
            for i in range(N):
                done_bool = bool(dataset["done"][i])

                for k, map_k in zip(
                    [
                        "obs",
                        "next_obs",
                        "action",
                        "reward",
                        "done",
                    ],
                    [
                        "observations",
                        "next_observations",
                        "actions",
                        "rewards",
                        "terminals",
                    ],
                ):
                    data_[map_k].append(dataset[k][i])
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
