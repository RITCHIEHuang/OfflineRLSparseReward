import gym
#gym.logger.set_level(gym.logger.DEBUG)
from offlinerl.algo import algo_select
from offlinerl.evaluation import CallBackFunctionList
from offlinerl.utils.env import get_env

from datasets import d4rl_dataset
from utils.exp_util import setup_exp_args
from evaluation.d4rl_score import d4rl_eval_fn
from gym.wrappers import Monitor
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, AntEnv, HopperEnv, Walker2dEnv
def my_get_env(task:str):
    if 'hopper' in task:
        return HopperEnv()
    elif 'walker' in task:
        return Walker2dEnv()
    elif 'halfcheetah' in task:
        return HalfCheetahEnv()

def record_video(task,policy,exp_name):
    env = my_get_env(task)
    env = Monitor(env, f'./{exp_name}-video', force=True)
    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    max_step = 1000
    cur = 0
    while not done and cur<max_step:
        cur += 1
        print(f"step:{cur}\n")
        state = state[np.newaxis]
        action = policy.get_action(state)
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    # train_buffer = d4rl_dataset.load_d4rl_buffer(algo_config)

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    # callback_list = CallBackFunctionList(
    #     [d4rl_eval_fn(task=algo_config["task"])]
    # )
    p = algo_trainer.get_policy()
    record_video(algo_config["task"],p,algo_trainer.exp_name)

    #algo_trainer.train(train_buffer, None, callback_fn=callback_list)


# python train_d4rl.py --algo_name=mopo --task=walker2d-medium-replay-v0 --delay_mode=constant --delay=20 --strategy=interval_average
if __name__ == "__main__":
    args = setup_exp_args()

    run_algo(args)
