import os
import time
import numpy as np

from gym import spaces

from recsim.agent import AbstractEpisodicRecommenderAgent

from envs.rec_env.cs_env import create_env


class RandomAgent(AbstractEpisodicRecommenderAgent):
    def __init__(self, observation_space, action_space):
        if len(observation_space["doc"].spaces) < len(action_space.nvec):
            raise RuntimeError("Slate size larger than size of the corpus.")
        super(RandomAgent, self).__init__(action_space)

    def step(self, reward, observation):
        action = np.random.choice(len(observation["doc"]), self._slate_size)
        return action


def create_agent(environment):
    return RandomAgent(environment.observation_space, environment.action_space)


def run_episode(ep, env, agent, max_steps_per_episode=27000):
    step_number = 0
    total_reward = 0.0

    start_time = time.time()

    observation = env.reset()
    action = agent.begin_episode(observation)

    # Keep interacting until we reach a terminal state.
    while True:
        last_observation = observation
        observation, reward, done, info = env.step(action)

        # Update environment-specific metrics with responses to the slate.
        env.update_metrics(observation["response"], info)

        total_reward += reward
        step_number += 1

        if done:
            break
        elif step_number == max_steps_per_episode:
            # Stop the run loop once we reach the true end of episode.
            break
        else:
            action = agent.step(reward, observation)

        print(f"cum next time: {observation['user']['cum_next_time']}")

    agent.end_episode(reward, observation)

    time_diff = time.time() - start_time

    print(
        f"Episode {ep} with steps: {step_number}, total reward: {total_reward} in {time_diff} seconds."
    )


max_episodes = 10

cs_environment = create_env()
random_agent = create_agent(cs_environment)

paths = []
for i in range(max_episodes):
    path = run_episode(i, cs_environment, random_agent)
    paths.append(path)
