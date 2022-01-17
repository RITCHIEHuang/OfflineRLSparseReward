import torch

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 256
hidden_layers = 2
num_heads = 20

batch_size = 256
max_epoch = 1000
steps_per_epoch = 3000
eval_epoch = 10

lr = 1e-4
target_update_interval = 2000
discount = 0.99
soft_target_tau = 1.0

# tune
grid_tune = {
    "lr": [1e-4, 3e-4, 1e-3],
    "batch_size": [64, 128, 256, 512],
}
