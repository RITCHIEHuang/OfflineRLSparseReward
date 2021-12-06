import gym
import rec_env

import numpy as np

from stable_baselines3 import PPO, DQN

env = gym.make("recs-random-v0")

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
# model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)


# eval
rewards = []
steps = []
for _ in range(10):
    episode_reward = 0
    step = 0
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step += 1

        # env.render()
        if done:
            break

    rewards.append(episode_reward)
    steps.append(step)

print("mean episode_rewards", np.mean(rewards))
print("mean episode_steps", np.mean(steps))
