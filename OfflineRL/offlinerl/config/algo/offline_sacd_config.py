import torch

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 256
hidden_layers = 2


batch_size = 256
steps_per_epoch = 1000
max_epoch = 500
eval_epoch = 10
target_update_interval = 2000

learnable_alpha = True
actor_lr = 1e-4
critic_lr = 1e-4
discount = 0.99
soft_target_tau = 1.0

# tune
grid_tune = {
    "actor_lr": [1e-4, 3e-4, 1e-3],
}
