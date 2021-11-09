d4rl_task_list = {
    0: "halfcheetah-random-v0",
    1: "halfcheetah-medium-v0",
    2: "halfcheetah-expert-v0",
    3: "halfcheetah-medium-replay-v0",
    4: "halfcheetah-medium-expert-v0",
    5: "walker2d-random-v0",
    6: "walker2d-medium-v0",
    7: "walker2d-expert-v0",
    8: "walker2d-medium-replay-v0",
    9: "walker2d-medium-expert-v0",
    10: "hopper-random-v0",
    11: "hopper-medium-v0",
    12: "hopper-expert-v0",
    13: "hopper-medium-replay-v0",
    14: "hopper-medium-expert-v0",
    15: "ant-random-v0",
    16: "ant-medium-v0",
    17: "ant-expert-v0",
    18: "ant-medium-replay-v0",
    19: "ant-medium-expert-v0",
    20: "antmaze-medium-play-v0",
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


def get_domain_by_task(task: str):
    if task in list(d4rl_task_list.values()):
        return "d4rl"
    if task in list(neorl_task_list.values()):
        return "neorl"
