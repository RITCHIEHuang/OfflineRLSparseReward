import torch

from utils.io_util import proj_path

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

hidden_layer_size = 256
hidden_layers = 2
num_quantiles = 20

log_data_path = f"{proj_path}/rec_env/data"
# buffer_type = "avg_traj"  # log_transition
buffer_type = "log_transition"  # log_transition
buffer_size = 1e6
batch_size = 256
warmup_size = 10000
max_step = 1e7
max_epoch = 10000
eval_epoch = 10

lr = 1e-4
exploration_init_eps = 0.2
exploration_final_eps = 0.02
target_update_interval = 1000
discount = 0.99
soft_target_tau = 1.0

# tune
grid_tune = {
    "lr": [1e-4, 3e-4, 1e-3],
    "batch_size": [64, 128, 256, 512],
}
