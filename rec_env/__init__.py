from gym.envs.registration import register

register(
    id="recs-random-v0",
    entry_point="rec_env.env:make_recs_env",
    max_episode_steps=1000,
    kwargs={
        "maze_map": maze_env.HARDEST_MAZE_TEST,
        "reward_type": "sparse",
        "dataset_url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_maze_v2/Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse_fixed.hdf5",
        "non_zero_reset": False,
        "eval": True,
        "maze_size_scaling": 4.0,
        "ref_min_score": 0.0,
        "ref_max_score": 1.0,
        "v2_resets": True,
    },
)
