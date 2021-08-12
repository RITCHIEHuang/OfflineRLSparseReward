from d3rlpy.datasets import MDPDataset
import argparse
import d3rlpy

from d3rlpy.algos import COMBO
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
from d3rlpy.gpu import Device
import gym
from sklearn.model_selection import train_test_split


def main(args):
    dataset = MDPDataset.load(
        "../datasets/dataset/Walker2d-v3_delay_20_expert_100.h5"
    )
    env_name = "Walker2d-v3"
    env = gym.make(env_name)

    d3rlpy.seed(42)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    device = None if args.gpu is None else Device(args.gpu)

    dynamics = ProbabilisticEnsembleDynamics(batch_size=256, use_gpu=device)

    dynamics.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_epochs=100,
        n_steps_per_epoch=500,
        logdir="logs",
        tensorboard_dir="tb_logs",
        scorers={
            "obs_error": dynamics_observation_prediction_error_scorer,
            "reward_error": dynamics_reward_prediction_error_scorer,
        },
    )

    combo = COMBO(
        q_func_factory=args.q_func, dynamics=dynamics, use_gpu=device
    )

    combo.fit(
        train_episodes,
        eval_episodes=test_episodes,
        n_epochs=1000,
        n_steps_per_epoch=5000,
        logdir="logs",
        tensorboard_dir="tb_logs",
        scorers={
            "environment": evaluate_on_environment(env),
            "td_error": td_error_scorer,
            "discounted_advantage": discounted_sum_of_advantage_scorer,
            "value_scale": average_value_estimation_scorer,
            "value_std": value_estimation_std_scorer,
            "action_diff": continuous_action_diff_scorer,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q-func",
        type=str,
        default="mean",
        choices=["mean", "qr", "iqn", "fqf"],
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    main(args)
