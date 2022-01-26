import torch

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

actor_features = 64
actor_layers = 2

batch_size = 64
steps_per_epoch = 1000
max_epoch = 300
eval_epoch = 10

actor_lr = 5e-5

# tune
params_tune = {
    "actor_lr": {"type": "continuous", "value": [1e-4, 1e-3]},
}

# tune
grid_tune = {
    "actor_lr": [1e-3],
}
