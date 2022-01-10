import gym
import rec_env

from elegantrl.agents.AgentSAC import AgentSAC, AgentModSAC
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.envs.Gym import get_gym_env_args
from elegantrl.train.config import Arguments
from elegantrl.train.run import train_and_evaluate

get_gym_env_args(gym.make("recs-v0"), if_print=True)

env_func = gym.make
env_args = {
    "env_num": 1,
    "env_name": "recs-v0",
    "max_step": 4000,
    "state_dim": 120,
    "action_dim": 10,
    "if_discrete": True,
    "if_per_or_gae": True,
    "target_return": 9,
    "id": "recs-v0",
}

args = Arguments(agent=AgentDQN(), env_func=env_func, env_args=env_args)

args.net_dim = 512
args.max_memo = int(1e6)
args.reward_scale = 1
args.batch_size = 128
args.target_step = 2000

args.eval_gap = 256
args.eval_times1 = 4
args.eval_times2 = 16
args.break_step = int(8e7)
args.if_allow_break = False
args.worker_num = 4
args.learner_gpus = [1, 2, 3, 4, 5, 6]

train_and_evaluate(args)
