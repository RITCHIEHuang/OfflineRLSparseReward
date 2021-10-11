import torch

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42

bc_policy_path = None
bc_policy_save_path = None

policy_mode = "random"  # bc
shaping_version = "v1"

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

actor_features = 256
actor_layers = 2

batch_size = 256
steps_per_epoch = 1000
max_epoch = 500

actor_lr = 1e-3

shaping_lr = 1e-4
hidden_layer_size = 256
hidden_layers = 2

# tune
grid_tune = {
    "actor_lr": [1e-3],
    "shaping_lr": [1e-3],
}
