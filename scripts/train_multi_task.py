import subprocess

from loguru import logger

template = (
    "sleep 1 && export CUDA_VISIBLE_DEVICES={0} && "
    "python train_d4rl.py --algo_name={1} --task={2} "
)

count_gpu_shell = "nvidia-smi | grep 'GeFor' | wc -l"


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


algos = ["bcq", "combo", "cql", "mopo"]
tasks = ["walker2d-medium-replay-v0"]

NUM_GPU = get_gpu_count()

logger.info(f"Train tasks with: {NUM_GPU} gpus.")
gpu_id = 0

process_buffer = []
for task in tasks:
    for algo in algos:
        param_list = [gpu_id % NUM_GPU, algo, task]
        str_command = template.format(*param_list)

        process = subprocess.Popen(str_command, shell=True)
        process_buffer.append(process)

        gpu_id += 1

output = [p.wait() for p in process_buffer]
