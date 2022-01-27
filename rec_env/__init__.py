from gym.envs.registration import register
from utils.io_util import proj_path

# deprecated
register(
    id="recs-random-v0",
    entry_point="rec_env.offline_env:get_recs_offline_env",
    max_episode_steps=4000,
    kwargs={
        "dataset_name": "recs-random-10000.npz",
        "dataset_path": f"{proj_path}/rec_env/data/recs-random-10000.npz",
        "ref_min_score": 0.0,
        "ref_max_score": 9.0,
        "reward_key": "retentions",
        "reward_type": "retention",
    },
)

register(
    id="recs-replay-v0",
    entry_point="rec_env.offline_env:get_recs_offline_env",
    max_episode_steps=4000,
    kwargs={
        "dataset_path": f"{proj_path}/rec_env/data/recs-replay.npz",
        "data_limit": int(1_000_000),
        "ref_min_score": 0.0,
        "ref_max_score": 9.0,
        "reward_key": "retentions",
        "reward_type": "retention",
    },
)

register(
    id="recs-medium-v0",
    entry_point="rec_env.offline_env:get_recs_offline_env",
    max_episode_steps=4000,
    kwargs={
        "dataset_path": f"{proj_path}/rec_env/data/recs-medium-v1.npz",
        "data_limit": int(1_000_000),
        "ref_min_score": 0.0,
        "ref_max_score": 9.0,
        "reward_key": "retentions",
        "reward_type": "retention",
    },
)

register(
    id="recs-random-large-v0",
    entry_point="rec_env.offline_env:get_recs_offline_env",
    max_episode_steps=4000,
    kwargs={
        "dataset_name": "recs-random-10000.npz",
        "dataset_path": "https://drive.google.com/uc?id=1qk0kJJBmTR6e1_zQlMV59IfkPEDCXHo7",
        "ref_min_score": 0.0,
        "ref_max_score": 9.0,
        "reward_key": "retentions",
        "reward_type": "retention",
    },
)


register(
    id="recs-v0",
    entry_point="rec_env.env:get_recs_env",
    max_episode_steps=4000,
    kwargs={
        # "reward_type": "click",
        "reward_type": "retention",
    },
)
