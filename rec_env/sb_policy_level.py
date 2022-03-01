import gym

import rec_env

import numpy as np

from stable_baselines3 import DQN

from rec_env.sb_callback import EvalCallback
from rec_env.sb_buffer import ReplayBuffer

from tqdm import tqdm


env = gym.make("recs-v0")
eval_env = gym.make("recs-v0")

dataset_config = {
    "medium": {
        "buffer_size": 200_000,
        "sample_size": 1000_000,
    },
    "expert": {
        "buffer_size": 1000_000,
        "sample_size": 1000_000,
    },
}


data_level = "medium"
config = dataset_config[data_level]
buffer_size = int(config["buffer_size"])
model = DQN(
    "MlpPolicy",
    env,
    buffer_size=buffer_size,
    replay_buffer_class=ReplayBuffer,
    verbose=1,
    batch_size=256,
    target_update_interval=2000,
    tensorboard_log="./logs",
    seed=10,
)
eval_callback = EvalCallback(eval_env, n_eval_episodes=100)

model.learn(total_timesteps=buffer_size, callback=eval_callback)

total_steps = config["sample_size"]

# eval
rewards = []
retentions = []
clicks = []
steps = []
observations = []
actions = []
dones = []
timeouts = []
next_observations = []
step = 0
global_done = False


def generator():
    while step < total_steps:
        print("steps:", step)
        yield


for _ in tqdm(generator()):
    episode_reward = 0
    episode_retention = 0
    episode_click = 0
    obs = env.reset()
    while True:
        observations.append(obs)
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        next_observations.append(obs)
        episode_reward += reward
        episode_retention += info["reward"]["retention"]
        episode_click += info["reward"]["click"]
        rewards.append(info["reward"]["retention"])
        retentions.append(info["reward"]["retention"])
        dones.append(done)
        timeouts.append(0)

        step += 1
        if step == total_steps:
            global_done = True
            break

        if done:
            break
    if global_done:
        break

obs_shape = env.reset().shape
act_dim = 1
print("obs shape:", obs_shape)
observations = np.reshape(np.array(observations), (-1, obs_shape[0]))
actions = np.reshape(np.array(actions), (-1, act_dim))
next_observations = np.reshape(np.array(next_observations), (-1, obs_shape[0]))
rewards = np.reshape(np.array(actions), -1)
timeouts = np.reshape(np.array(timeouts), -1)
terminals = np.reshape(np.array(dones), -1)
retentions = np.reshape(np.array(retentions), -1)
np.savez_compressed(
    f"./data/recs-{data_level}-v0.npz",
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    timeouts=timeouts,
    next_observations=next_observations,
    retentions=retentions,
)
