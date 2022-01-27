# import gym
# gym.logger.set_level(gym.logger.DEBUG)
import cv2
from tqdm import tqdm
import os
from offlinerl.algo import algo_select
#from offlinerl.evaluation import CallBackFunctionList
#from offlinerl.utils.env import get_env

from utils.exp_util import setup_exp_args
from gym.wrappers import Monitor
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv, HopperEnv, Walker2dEnv

def mp4_to_frame(dir,file):
    print(f"transform {file} to frames!")
    vidcap = cv2.VideoCapture(file)
    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(dir+os.sep+"frame%d.jpg" % count, image)     
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1

def my_get_env(task:str):
    if 'hopper' in task:
        return HopperEnv()
    elif 'walker' in task:
        return Walker2dEnv()
    elif 'halfcheetah' in task:
        return HalfCheetahEnv()

def record_video(task,policy,exp_name):
    env = my_get_env(task)
    directory = f'./{exp_name}-video'
    env = Monitor(env, directory, force=True)
    state, done = env.reset(), False
    rewards = 0
    lengths = 0
    max_step = 1000
    cur = 0
    for _ in tqdm(range(max_step)):
        cur += 1
        state = state[np.newaxis]
        action = policy.get_action(state)
        state, reward, done, _ = env.step(action)
        rewards += reward
        lengths += 1
        if done:
            break
    env.close()
    mp4_files = filter(lambda x:x.endswith('mp4'),os.listdir(directory))
    mp4_files = map(lambda x:directory+os.sep+x,mp4_files)
    for mp4 in mp4_files:
        mp4_to_frame(directory,mp4)


def run_algo(kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    # train_buffer = d4rl_dataset.load_d4rl_buffer(algo_config)

    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)

    # callback_list = CallBackFunctionList(
    #     [d4rl_eval_fn(task=algo_config["task"])]
    # )
    p = algo_trainer.get_policy()
    record_video(algo_config["task"],p,kwargs["model_path"][2:-3])

    #algo_trainer.train(train_buffer, None, callback_fn=callback_list)


# python train_d4rl.py --algo_name=mopo --task=walker2d-medium-replay-v0 --delay_mode=constant --delay=20 --strategy=interval_average
if __name__ == "__main__":
    args = setup_exp_args()

    run_algo(args)
