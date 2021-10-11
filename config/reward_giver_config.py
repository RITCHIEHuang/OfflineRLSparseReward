import torch

task = "Hopper-v3"
task_data_type = "low"
task_train_num = 99

seed = 42


device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None


batch_size = 256
steps_per_epoch = 1000
max_epoch = 500

reward_lr = 1e-3
hidden_layer_size = 256
hidden_layers = 2

# tune
grid_tune = {
    "reward_lr": [1e-3],
}
