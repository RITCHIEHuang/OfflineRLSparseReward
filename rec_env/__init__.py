from gym.envs.registration import register
from utils.io_util import proj_path

register(
    id="recs-random-v0",
    entry_point="rec_env.env:get_recs_env",
    max_episode_steps=27000,
    kwargs={
        "dataset_path": f"{proj_path}/rec_env/data/recs-random-100.npz",
        "ref_min_score": 0.0,
        "ref_max_score": 1.0,
    },
)
