import torch

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 256
hidden_layers = 2

batch_size = 128
buffer_size = 1e6
max_epoch = 10000
steps_per_epoch = 1000
eval_epoch = 10

lr = 3e-4
target_update_interval = 50
discount = 0.99
soft_target_tau = 0

# tune
grid_tune = {
    "lr": [1e-4, 3e-4, 1e-3],
    "batch_size": [64, 128, 256, 512],
}
