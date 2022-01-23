import torch

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

max_epoch = 2000
steps_per_epoch = 1000
eval_epoch = 50
policy_bc_steps = 0

batch_size = 256
hidden_layer_size = 256
hidden_layers = 2
layer_num = 2
actor_lr = 3e-4
critic_lr = 3e-4

discount = 0.99
soft_target_tau = 5e-3

q_update_period = 1
policy_update_period = 1
target_update_period = 1

# update
# antmaze
# beta = 0.1
# quantile = 0.9
# gym
beta = 1.0 / 3
quantile = 0.7
clip_score = 100


# tune
grid_tune = {
    # "soft_target_tau": [1e-2, 5e-2, 1e-3, 5e-3],
    # "actor_lr": [1e-4, 3e-4],
    "clip_score": [20, 50, 100, 150, 200],
    "beta": [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
    "quantile": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
}
