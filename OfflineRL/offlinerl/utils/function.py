import torch
from torch.functional import F


def soft_clamp(x: torch.Tensor, _min=None, _max=None):
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


def get_linear_fn(start: float, end: float):
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = ``end_fraction``.
    This is used in DQN for linearly annealing the exploration fraction
    (epsilon for the epsilon-greedy strategy).
    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :return:
    """

    def func(progress_remaining: float) -> float:
        return (
            start + (1 - progress_remaining) * (end - start) 
        )

    return func
