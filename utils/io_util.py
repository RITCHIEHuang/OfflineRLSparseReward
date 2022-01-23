import os

proj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(
    os.path.dirname(os.path.dirname(proj_path)), "/data/OfflineSparseReward"
)
