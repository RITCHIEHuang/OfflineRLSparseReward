import os
from collections import defaultdict
import time
import numpy as np


from rec_env.env import get_recs_env
from utils.io_util import proj_path


class RandomAgent(object):
    def __init__(self, env):
        observation_space = env.raw_observation_space
        self.num_candidates = env.environment._num_candidates
        self._slate_size = env.environment._slate_size
        if len(observation_space["doc"].spaces) < self._slate_size:
            raise RuntimeError("Slate size larger than size of the corpus.")

    def step(self, observation, extra=None):
        action = np.random.choice(self.num_candidates, self._slate_size)
        return action


def score_func(user_interest, doc_obs):
    score = (user_interest @ doc_obs["emb"]) / np.sqrt(
        user_interest @ user_interest * doc_obs["emb"] @ doc_obs["emb"]
    )
    return score


class GreedyAgent(object):
    def __init__(self, env):
        observation_space = env.raw_observation_space
        self.num_candidates = env.environment._num_candidates
        self._slate_size = env.environment._slate_size
        if len(observation_space["doc"].spaces) < self._slate_size:
            raise RuntimeError("Slate size larger than size of the corpus.")

    def step(self, observation, extra=None):
        user_interest = extra["user"]["interest"]
        doc_obs = extra["doc"]
        scores = [score_func(user_interest, x) for x in doc_obs.values()]
        action = []
        for _ in range(self._slate_size):
            action.append(np.argmax(scores))
            scores = scores[1:]
        return action


AGENT_MAPPING = {"random": RandomAgent, "greedy": GreedyAgent}


def create_agent(environment, agent_type: RandomAgent):
    return agent_type(environment)


def run_episode(ep, env, agent, max_steps_per_episode=27000):
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    timeouts = []
    clicks = []
    retentions = []

    step_number = 0
    total_reward = 0.0

    start_time = time.time()
    observation, info = env.reset(True)
    # Keep interacting until we reach a terminal state.
    while True:
        step_number += 1
        action = agent.step(observation, info["raw_obs"])
        observations.append(observation)
        observation, reward, done, info = env.step(action)
        next_observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        terminals.append(done)
        clicks.append(info["reward"]["click"])
        retentions.append(info["reward"]["retention"])
        timeouts.append(bool(step_number == max_steps_per_episode))

        # Update environment-specific metrics with responses to the slate.
        env.update_metrics(info["raw_obs"]["response"], info)

        total_reward += reward

        if done or step_number == max_steps_per_episode:
            break
        # print(f"cum next time: {observation['user']['cum_next_time']}")

    time_diff = time.time() - start_time

    print(
        f"Episode {ep} with steps: {step_number}, total reward: {total_reward} in {time_diff} seconds."
    )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
        "timeouts": timeouts,
        "clicks": clicks,
        "retentions": retentions,
    }


def sample_store(n_traj=100, agent_name="random"):
    cs_environment = get_recs_env()
    random_agent = create_agent(
        cs_environment, agent_type=AGENT_MAPPING[agent_name]
    )

    traj = defaultdict(list)
    for i in range(n_traj):
        path = run_episode(i, cs_environment, random_agent)

        for k, v in path.items():
            traj[k].append(v)

    for k in [
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "terminals",
        "timeouts",
        "clicks",
        "retentions",
    ]:
        print(f"{k} shape", np.concatenate(traj[k]).shape)

    data_dir = f"{proj_path}/rec_env/data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.savez_compressed(
        f"{data_dir}/recs-{agent_name}-{n_traj}.npz",
        observations=np.concatenate(traj["observations"]),
        actions=np.concatenate(traj["actions"]),
        rewards=np.concatenate(traj["rewards"]),
        terminals=np.concatenate(traj["terminals"]),
        timeouts=np.concatenate(traj["timeouts"]),
        next_observations=np.concatenate(traj["next_observations"]),
        clicks=np.concatenate(traj["clicks"]),
        retentions=np.concatenate(traj["retentions"]),
    )

    print('=' * 80)
    print("average reward", np.sum(np.concatenate(traj["rewards"])) / n_traj)
    print("average click", np.sum(np.concatenate(traj["clicks"])) / n_traj)
    print("average retention", np.sum(np.concatenate(traj["retentions"])) / n_traj)
    print('=' * 80)

if __name__ == "__main__":
    sample_store(agent_name="random")
    # sample_store(agent_name="greedy")
