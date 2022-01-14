from gym.envs.registration import register
from utils.io_util import proj_path

register(
    id="recs-random-v0",
    entry_point="rec_env.offline_env:get_recs_offline_env",
    max_episode_steps=4000,
    kwargs={
        "dataset_path": f"{proj_path}/rec_env/data/recs-random-10000.npz",
        "ref_min_score": 0.0,
        "ref_max_score": 9.0,
        "reward_key": "retentions",
        "reward_type": "retention"
    },
)

register(
    id="recs-replay-v0",
    entry_point="rec_env.offline_env:get_recs_offline_env",
    max_episode_steps=4000,
    kwargs={
        "dataset_path": f"{proj_path}/rec_env/data/recs-replay.npz",
        "ref_min_score": 0.0,
        "ref_max_score": 9.0,
        "reward_key": "retentions",
        "reward_type": "retention"
    },
)

register(
    id="recs-v0",
    entry_point="rec_env.env:get_recs_env",
    max_episode_steps=4000,
    kwargs={
        "reward_type": "click",
    }
)
