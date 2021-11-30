import torch

seed = 42


device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None


batch_size = 256
steps_per_epoch = 1000
max_epoch = 100

reward_lr = 3e-4
hidden_layer_size = 256
hidden_layers = 2

# tune
grid_tune = {
    "reward_lr": [1e-3],
}
