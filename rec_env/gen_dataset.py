from collections import defaultdict
import time
import numpy as np

from recsim.agent import AbstractEpisodicRecommenderAgent

from rec_env.env import get_recs_env


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


def create_agent(environment):
    return RandomAgent(environment)
    # return GreedyAgent(environment)


def run_episode(ep, env, agent, max_steps_per_episode=27000):
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    timeouts = []

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
    }


def sample_store(n_traj=100):
    cs_environment = get_recs_env()
    random_agent = create_agent(cs_environment)

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
    ]:
        print(np.concatenate(traj[k]).shape)

    np.savez_compressed(
        f"data/recs-random-{n_traj}.npz",
        observations=np.concatenate(traj["observations"]),
        actions=np.concatenate(traj["actions"]),
        rewards=np.concatenate(traj["rewards"]),
        terminals=np.concatenate(traj["terminals"]),
        timeouts=np.concatenate(traj["timeouts"]),
        next_observations=np.concatenate(traj["next_observations"]),
    )

    print("average reward", np.sum(np.concatenate(traj["rewards"])) / n_traj)


if __name__ == "__main__":
    sample_store()
