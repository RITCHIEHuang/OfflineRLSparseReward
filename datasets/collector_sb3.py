import os
import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from d3rlpy.dataset import MDPDataset

from envs.delay_reward import DelayedRewardEnv

checkpoint_dir = "checkpoint"
dataset_dir = "dataset"
env_name = "Walker2d-v3"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# delay_freq = 20
delay_freq = 1
exp_name = f"{env_name}_delay_{delay_freq}"
env = DelayedRewardEnv(env_name, delay_freq)
eval_env = gym.make(env_name)

# setup SAC algorithm
sac = SAC("MlpPolicy", env, verbose=1, tensorboard_log=f"tb_logs/{exp_name}")
checkpoint_callback = CheckpointCallback(
    save_freq=10000, save_path=f"{checkpoint_dir}/{exp_name}"
)


def train(sac, time_steps=1000000):
    sac.learn(time_steps, callback=checkpoint_callback, eval_env=eval_env)


def collect_dataset(
    sac,
    checkpoint_path,
    env,
    n_episodes=10,
    action_noise_std=0.05,
    level="expert",
):
    sac = sac.load(checkpoint_path)
    mean_rew, std_rew = evaluate_policy(sac, eval_env)
    print(f"mean reward: {mean_rew}, std reward: {std_rew}")

    dataset = collect_episodes(sac, env, n_episodes, action_noise_std)
    dataset.dump(f"{dataset_dir}/{env_name}_{level}_{n_episodes}.h5")


def collect_episodes(sac, env, n_episodes, action_noise_std=0.0):
    observations = []
    actions = []
    rewards = []
    terminals = []

    total_timesteps = []
    episode_rewards = []

    num_collected_episodes, num_collected_steps = 0, 0
    action_noise = NormalActionNoise(0.0, action_noise_std)

    while num_collected_episodes < n_episodes:
        obs = env.reset()
        done = False
        episode_reward, episode_timesteps = 0.0, 0
        while not done:
            # Select action randomly or according to policy
            sac._last_obs = obs
            action, scaled_action = sac._sample_action(0, action_noise)
            # Rescale and perform action
            new_obs, reward, done, infos = env.step(action)

            episode_timesteps += 1
            num_collected_steps += 1
            episode_reward += reward

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(done)

            obs = new_obs
            if done:
                num_collected_episodes += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

    print(
        f"[collect episodes] num_collected_episodes: {num_collected_episodes}, episode_rewards mean: {np.mean(episode_rewards)}, episode_timesteps mean: {np.mean(total_timesteps)}, num_collected_steps: {num_collected_steps}"
    )
    # transform to `MDPDataset`
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    dataset = MDPDataset(observations, actions, rewards, terminals)
    return dataset


if __name__ == "__main__":
    collect_dataset(
        sac,
        checkpoint_path=f"{checkpoint_dir}/{exp_name}/rl_model_1000000_steps.zip",
        env=env,
        n_episodes=100,
    )
