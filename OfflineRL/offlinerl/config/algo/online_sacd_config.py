import torch
from utils.io_util import proj_path

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 256
hidden_layers = 2

buffer_size = 1e6
batch_size = 128
warmup_size = 10000
target_update_interval = 8000
update_interval = 4
steps_per_epoch = 1
max_epoch = 10000
max_step = 1e7
eval_epoch = 50

learnable_alpha = True
actor_lr = 1e-4
critic_lr = 1e-4
discount = 0.99
soft_target_tau = 1.0

# tune
grid_tune = {
    "actor_lr": [1e-4, 3e-4, 1e-3],
}
