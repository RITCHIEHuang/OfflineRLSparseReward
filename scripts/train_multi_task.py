import argparse
import subprocess

from utils.task_util import d4rl_task_list, neorl_task_list, rec_task_list
from utils.gpu_util import get_gpu_count, get_gpu_memory_desc_iter

d4rl_template = (
    "sleep 1 && export CUDA_VISIBLE_DEVICES={0} && "
    "python train_d4rl.py --algo_name={1} --task={2} --delay_mode={3} --delay={4} --seed={5} --strategy={6} --reward_scale={7} --reward_shift={8} "
)
neorl_template = (
    "sleep 1 && export CUDA_VISIBLE_DEVICES={0} && "
    "python train_neorl.py --algo_name={1} --task={2} --delay_mode={3} --delay={4} --seed={5} --strategy={6} --reward_scale={7} --reward_shift={8} "
)

recs_template = (
    "sleep 1 && export CUDA_VISIBLE_DEVICES={0} && "
    "python train_recs.py --algo_name={1} --task={2} --delay_mode={3} --delay={4} --seed={5} --strategy={6} --reward_scale={7} --reward_shift={8} "
)


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("Experiment runner")
    parser.add_argument(
        "--algo_name", help="algorithm", type=str, default="cql"
    )
    parser.add_argument(
        "--strategy",
        help="delay rewards strategy, can be multiple strategies seperated by  `,`",
        type=str,
        default="none",
    )
    parser.add_argument(
        "--domain",
        help="name of experiment domain",
        type=str,
        default="d4rl",
        choices=["d4rl", "neorl", "recs"],
    )
    parser.add_argument(
        "--task_id",
        help="task id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--reward_scale", help="scale for reward", type=float, default=1.0
    )
    parser.add_argument(
        "--reward_shift", help="shift for reward", type=float, default=0.0
    )
    return parser.parse_args()


NUM_GPU = get_gpu_count()
print(f"num_gpu: {NUM_GPU} gpus.")

args = argsparser()
domain = args.domain
# algos = ["iql"]
# algos = ["mopo"]
algos = [args.algo_name]

# algos = ["mopo"]
# algos = ["bc", "bcq", "cql", "mopo"]

# delay_modes = ["constant", "random"]
delay_modes = ["none"]
seeds = [10, 100, 1000]
delays = [1]

# strategies = ["none"]
strategies = [args.strategy]
reward_scale = args.reward_scale
reward_shift = args.reward_shift
# strategies = ["interval_average"]
# strategies = [
#     "none",
#     "minmax",
#     "zscore",
#     "episodic_average",
#     "episodic_random",
#     "episodic_ensemble",
#     "interval_average",
#     "interval_random",
#     "interval_ensemble",
#     "transformer_decompose",
#     "pg_reshaping",
# ]

template = None
if domain == "d4rl":
    template = d4rl_template
    try:
        task = d4rl_task_list[args.task_id]
    except Exception:
        exit(-1)

elif domain == "neorl":
    template = neorl_template
    try:
        task = neorl_task_list[args.task_id]
    except Exception:
        exit(-1)
elif domain == "recs":
    template = recs_template
    try:
        task = rec_task_list[args.task_id]
    except Exception:
        exit(-1)
else:
    raise NotImplementedError()

gpu_id_iter = get_gpu_memory_desc_iter()

print(f"Train task: {task}")
process_buffer = []

for algo in algos:
    for delay_mode in delay_modes:
        for delay in delays:
            for strategy in strategies:
                for seed in seeds:
                    param_list = [
                        next(gpu_id_iter),
                        algo,
                        task,
                        delay_mode,
                        delay,
                        seed,
                        strategy,
                        reward_scale,
                        reward_shift,
                    ]
                    str_command = template.format(*param_list)

                    process = subprocess.Popen(str_command, shell=True)
                    process_buffer.append(process)

output = [p.wait() for p in process_buffer]
