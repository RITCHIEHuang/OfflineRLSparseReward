import argparse
import subprocess

from utils.d4rl_tasks import task_list

template = (
    "sleep 1 && export CUDA_VISIBLE_DEVICES={0} && "
    "python train_d4rl.py --algo_name={1} --task={2} --delay_mode={3} --delay={4} --seed={5} --strategy={6} "
)

count_gpu_shell = "nvidia-smi | grep 'GeFor' | wc -l"


def argsparser():
    # Experiment setting
    parser = argparse.ArgumentParser("Experiment runner")
    parser.add_argument(
        "--algo_name",
        help="algorithm",
        type=str,
        default="bc",
        choices=["bcq", "cql", "mopo"],
    )
    parser.add_argument("--seed", help="random seed", type=int, default=2021)
    parser.add_argument(
        "--task_id",
        help="task id",
        type=int,
        default=0,
        choices=list(range(len(task_list))),
    )
    return parser.parse_args()


def get_gpu_count():
    p = subprocess.Popen(
        count_gpu_shell,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print("Subprogram output: [{}]".format(line))
            out = line.decode("utf-8")
    return int(out)


NUM_GPU = get_gpu_count()
print("num_gpu: {NUM_GPU} gpus.")

args = argsparser()
# algo = args.algo_name

algos = ["mopo"]
# algos = ["bc", "bcq", "cql", "mopo"]

# delay_modes = ["constant", "random"]
delay_modes = ["constant"]

delays = [100]

strategies = ["interval_average"]
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


task = task_list[args.task_id]
seed = args.seed
gpu_id = 0

print(f"Train task: {task}, seed: {seed}")
process_buffer = []

for algo in algos:
    for delay_mode in delay_modes:
        for delay in delays:
            for strategy in strategies:
                param_list = [
                    gpu_id % NUM_GPU,
                    algo,
                    task,
                    delay_mode,
                    delay,
                    seed,
                    strategy,
                ]
                str_command = template.format(*param_list)

                process = subprocess.Popen(str_command, shell=True)
                process_buffer.append(process)

                gpu_id += 1

output = [p.wait() for p in process_buffer]
