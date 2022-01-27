import torch

seed = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
obs_shape = None
act_shape = None
max_action = None

# transformer params
d_model = 512
nhead = 4
hidden_features = 128
hidden_layers = 4
dropout = 0.1

batch_size = 32 
steps_per_epoch = 1000
max_epoch = 100

lr = 1e-3
