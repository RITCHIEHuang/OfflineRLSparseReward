import subprocess as sp
import os

def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return list(zip(range(len(memory_free_values)),memory_free_values))

def get_gpu_memory_desc():
    l = get_gpu_memory()
    return sorted(l, key=lambda x:x[1],reverse=True)

def get_gpu_memory_desc_iter():
    l = get_gpu_memory_desc()
    length = len(l)
    gpu_id = 0
    while True:
        yield l[gpu_id%length][0]
        gpu_id += 1

if __name__ == '__main__':

    print(get_gpu_memory_desc())
    it = get_gpu_memory_desc_iter()
    print(next(it))
    print(next(it))