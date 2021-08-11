import gym

from d3rlpy.datasets import MDPDataset 
from d3rlpy.algos import SAC
from d3rlpy.online.explorers import ConstantEpsilonGreedy, NormalNoise
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.metrics.scorer import evaluate_on_environment

from envs.delay_reward import DelayedRewardEnv

env_name = "Walker2d-v3"
# delay_freq = 20
delay_freq = 1

env = DelayedRewardEnv(env_name, delay_freq)
eval_env = gym.make(env_name)


# setup SAC algorithm
sac = SAC(
    batch_size=256,
    actor_learning_rate=1e-4,
    critic_learning_rate=3e-4,
    temp_learning_rate=1e-4,
    use_gpu=True,
)

# setup explorer
explorer = NormalNoise(std=0.05)

# setup replay buffer
buffer = ReplayBuffer(maxlen=1000000, env=env)

# start training
sac.fit_online(
    env,
    buffer,
    #     explorer,
    eval_env=eval_env,
    n_steps=1000000,
    n_steps_per_epoch=1000,
    update_interval=1,
    update_start_step=1000,
    save_interval=1,
    save_metrics=True,
    show_progress=True,
    logdir="logs",
    tensorboard_dir="tb_logs",
    experiment_name=f"SAC_{env_name}_delay_{delay_freq}",
)
