from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F


class ActorProb(nn.Module):
    """Simple actor network (output with a Categorical distribution) with MLP.
    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.
    """

    def __init__(
        self,
        preprocess_net: nn.Module,
        action_num: int,
        hidden_layer_size: int = 128,
    ) -> None:
        super().__init__()
        self.preprocess = preprocess_net
        self.head = nn.Linear(hidden_layer_size, action_num)

    def forward(
        self,
        s: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: s -> logits -> (mu, sigma)."""
        logits, h = self.preprocess(s, state)
        probs = F.softmax(self.head(logits), dim=1)

        return Categorical(probs)
