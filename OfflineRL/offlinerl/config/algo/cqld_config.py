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
hidden_layers = 3
layer_num = 2
actor_lr = 1e-4
critic_lr = 3e-4
use_automatic_entropy_tuning = True
target_entropy = 0.0
discount = 0.99
soft_target_tau = 1.0
target_update_interval = 2000

# min Q
explore = 1.0
temp = 1.0
min_q_weight = 5.0

# lagrange
lagrange_thresh = 1

# extra params
backup_type = "min"  # "min"
backup_entropy = False

discrete = False

# tune
grid_tune = {
    "actor_lr": [1e-4, 3e-4],
    "min_q_weight": [1, 2, 5, 10],
    "lagrange_thresh": [-1, 5, 10],
}
