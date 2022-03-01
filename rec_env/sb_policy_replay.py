import gym

import rec_env

import numpy as np

from stable_baselines3 import DQN

from rec_env.sb_callback import EvalCallback
from rec_env.sb_buffer import ReplayBuffer

env = gym.make("recs-v0")
eval_env = gym.make("recs-v0")

buffer_size = int(1_000_000)
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


def save_buffer():
    buffer = model.replay_buffer
    obs_dim = buffer.obs_shape[0]
    act_dim = buffer.action_dim
    np.savez_compressed(
        f"data/recs-replay-v0.npz",
        observations=buffer.observations[: buffer.size(), ...].reshape(
            -1, obs_dim
        ),
        actions=buffer.actions[: buffer.size(), ...].reshape(-1, act_dim),
        rewards=buffer.rewards[: buffer.size(), ...].reshape(
            -1,
        ),
        retentions=buffer.retentions[: buffer.size(), ...].reshape(
            -1,
        ),
        clicks=buffer.clicks[: buffer.size(), ...].reshape(
            -1,
        ),
        terminals=buffer.dones[: buffer.size(), ...].reshape(
            -1,
        ),
        timeouts=buffer.timeouts[: buffer.size(), ...].reshape(
            -1,
        ),
        next_observations=buffer.next_observations[
            : buffer.size(), ...
        ].reshape(-1, obs_dim),
    )

    print("Save buffer", buffer.size())


# eval
rewards = []
retentions = []
clicks = []
steps = []
for _ in range(100):
    episode_reward = 0
    episode_retention = 0
    episode_click = 0
    step = 0
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_retention += info["reward"]["retention"]
        episode_click += info["reward"]["click"]

        step += 1

        if done:
            break

    rewards.append(episode_reward)
    retentions.append(episode_retention)
    clicks.append(episode_click)
    steps.append(step)

print("mean episode_reward", np.mean(rewards))
print("mean episode_retention", np.mean(retentions))
print("mean episode_click", np.mean(clicks))
print("mean episode_steps", np.mean(steps))

save_buffer()
