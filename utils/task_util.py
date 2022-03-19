d4rl_task_list = {
    0: "halfcheetah-random-v0",
    1: "halfcheetah-medium-v0",
    2: "halfcheetah-medium-replay-v0",
    3: "halfcheetah-medium-expert-v0",
    4: "walker2d-random-v0",
    5: "walker2d-medium-v0",
    6: "walker2d-medium-replay-v0",
    7: "walker2d-medium-expert-v0",
    8: "hopper-random-v0",
    9: "hopper-medium-v0",
    10: "hopper-medium-replay-v0",
    11: "hopper-medium-expert-v0",
    # -----------------------------------------
    20: "antmaze-umaze-v0",
    21: "antmaze-umaze-diverse-v0",
    22: "antmaze-medium-play-v0",
    23: "antmaze-medium-diverse-v0",
    24: "antmaze-large-play-v0",
    25: "antmaze-large-diverse-v0",
    # -----------------------------------------
    30: "halfcheetah-random-v2",
    31: "halfcheetah-medium-v2",
    32: "halfcheetah-medium-replay-v2",
    33: "halfcheetah-medium-expert-v2",
    34: "walker2d-random-v2",
    35: "walker2d-medium-v2",
    36: "walker2d-medium-replay-v2",
    37: "walker2d-medium-expert-v2",
    38: "hopper-random-v2",
    39: "hopper-medium-v2",
    40: "hopper-medium-replay-v2",
    41: "hopper-medium-expert-v2",
    # -----------------------------------------
    50: "antmaze-umaze-v2",
    51: "antmaze-umaze-diverse-v2",
    52: "antmaze-medium-diverse-v2",
    53: "antmaze-medium-play-v2",
    54: "antmaze-large-play-v2",
    55: "antmaze-large-diverse-v2",
}

neorl_task_list = {
    0: "HalfCheetah-v3-low-10",
    1: "Hopper-v3-low-10",
    2: "Walker2d-v3-low-10",
    3: "HalfCheetah-v3-low-100",
    4: "Hopper-v3-low-100",
    5: "Walker2d-v3-low-100",
    6: "HalfCheetah-v3-low-1000",
    7: "Hopper-v3-low-1000",
    8: "Walker2d-v3-low-1000",
}

rec_task_list = {
    0: "recs-random-v0",
    1: "recs-medium-v0",
    2: "recs-medium-replay-v0",
    3: "recs-replay-v0",
    4: "recs-v0",
    5: "recs-random-large-v0",
}


def get_domain_by_task(task: str):
    if task in list(d4rl_task_list.values()):
        return "d4rl"
    elif task in list(neorl_task_list.values()):
        return "neorl"
    elif task in list(rec_task_list.values()):
        return "recs"
    else:
        return "gym"
