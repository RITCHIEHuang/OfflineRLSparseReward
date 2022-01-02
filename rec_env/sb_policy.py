import gym

import rec_env

import numpy as np

from stable_baselines3 import PPO, DQN

from rec_env.sb_callback import EvalCallback


env = gym.make("recs-v0")
eval_env = gym.make("recs-v0")
# print(env.observation_space)
# print(env.action_space.n)
# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model = DQN("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
eval_callback = EvalCallback(eval_env, n_eval_episodes=100)

model.learn(total_timesteps=3000000, callback=eval_callback)


def save_buffer():
    buffer = model.replay_buffer
    obs_dim = buffer.obs_shape[0]
    act_dim = buffer.action_dim
    np.savez_compressed(
        f"data/recs-replay.npz",
        observations=buffer.observations[: buffer.size(), ...].reshape(
            -1, obs_dim
        ),
        actions=buffer.actions[: buffer.size(), ...].reshape(-1, act_dim),
        rewards=buffer.rewards[: buffer.size(), ...].reshape(-1, 1),
        terminals=buffer.dones[: buffer.size(), ...].reshape(-1, 1),
        timeouts=buffer.timeouts[: buffer.size(), ...].reshape(-1, 1),
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
for _ in range(10):
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
