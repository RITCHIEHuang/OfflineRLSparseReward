import os
import time
import numpy as np

import tensorflow as tf

from gym import spaces

from recsim.simulator import environment
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


# initial_observation = cs_environment.reset()
# print("User Observable Features")
# print(initial_observation["user"])
# print("User Response")
# print(initial_observation["response"])
# print("Document Observable Features")
# for doc_id, doc_features in initial_observation["doc"].items():
#     print("ID:", doc_id, "features:", doc_features)


# print("=" * 100)
# print("Document observation space")
# for key, space in cs_environment.observation_space["doc"].spaces.items():
#     print(key, ":", space)
# print("Response observation space")
# print(cs_environment.observation_space["response"])
# print("User observation space")
# print(cs_environment.observation_space["user"])


def log_one_step(
    env,
    user_obs,
    doc_obs,
    slate,
    responses,
    reward,
    is_terminal,
    sequence_example,
):
    """Adds one step of agent-environment interaction into SequenceExample.

    Args:
      user_obs: An array of floats representing user state observations
      doc_obs: A list of observations of the documents
      slate: An array of indices to doc_obs
      responses: A list of observations of responses for items in the slate
      reward: A float for the reward returned after this step
      is_terminal: A boolean for whether a terminal state has been reached
      sequence_example: A SequenceExample proto for logging current episode
    """

    def _add_float_feature(feature, values):
        feature.feature.add(float_list=tf.train.FloatList(value=values))

    def _add_int64_feature(feature, values):
        feature.feature.add(int64_list=tf.train.Int64List(value=values))

    if episode_writer is None:
        return
    fl = sequence_example.feature_lists.feature_list

    if isinstance(env.environment, environment.MultiUserEnvironment):
        for i, (
            single_user,
            single_slate,
            single_user_responses,
            single_reward,
        ) in enumerate(zip(user_obs, slate, responses, reward)):
            user_space = list(env.observation_space.spaces["user"].spaces)[i]
            _add_float_feature(
                fl["user_%d" % i], spaces.flatten(user_space, single_user)
            )
            _add_int64_feature(fl["slate_%d" % i], single_slate)
            _add_float_feature(fl["reward_%d" % i], [single_reward])
            for j, response in enumerate(single_user_responses):
                resp_space = env.observation_space.spaces["response"][i][0]
                for k in response:
                    _add_float_feature(
                        fl["response_%d_%d_%s" % (i, j, k)],
                        spaces.flatten(resp_space, response),
                    )
    else:  # single-user environment
        _add_float_feature(
            fl["user"],
            spaces.flatten(env.observation_space.spaces["user"], user_obs),
        )
        _add_int64_feature(fl["slate"], slate)
        for i, response in enumerate(responses):
            resp_space = env.observation_space.spaces["response"][0]
            for k in response:
                _add_float_feature(
                    fl["response_%d_%s" % (i, k)],
                    spaces.flatten(resp_space, response),
                )
        _add_float_feature(fl["reward"], [reward])

    for i, doc in enumerate(list(doc_obs.values())):
        doc_space = list(env.observation_space.spaces["doc"].spaces.values())[
            i
        ]
        _add_float_feature(fl["doc_%d" % i], spaces.flatten(doc_space, doc))

    _add_int64_feature(fl["is_terminal"], [is_terminal])


def run_episode(ep, env, agent, episode_writer, max_steps_per_episode=27000):
    step_number = 0
    total_reward = 0.0

    start_time = time.time()

    sequence_example = tf.train.SequenceExample()
    observation = env.reset()
    action = agent.begin_episode(observation)

    # Keep interacting until we reach a terminal state.
    while True:
        last_observation = observation
        observation, reward, done, info = env.step(action)
        log_one_step(
            env,
            last_observation["user"],
            last_observation["doc"],
            action,
            observation["response"],
            reward,
            done,
            sequence_example,
        )
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
    if episode_writer is not None:
        episode_writer.write(sequence_example.SerializeToString())

    time_diff = time.time() - start_time

    print(
        f"Episode {ep} with steps: {step_number}, total reward: {total_reward} in {time_diff} seconds."
    )


output_dir = "./output"
log_file = "log"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

episode_writer = tf.io.TFRecordWriter(os.path.join(output_dir, log_file))

max_episodes = 10

cs_environment = create_env()
random_agent = create_agent(cs_environment)

for i in range(max_episodes):
    run_episode(i, cs_environment, random_agent, episode_writer)
