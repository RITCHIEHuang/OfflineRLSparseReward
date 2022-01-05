from typing import Any, Dict, Tuple, Union, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from offlinerl.utils.net.common import MLP, DiscretePolicy


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


class CategoricalActor(nn.Module, DiscretePolicy):
    def __init__(self, obs_dim, action_dim, hidden_size, hidden_layers):
        super().__init__()
        self.backbone = MLP(
            in_features=obs_dim,
            out_features=action_dim,
            hidden_features=hidden_size,
            hidden_layers=hidden_layers,
        )

    def policy_infer(self, obs):
        probs = self(obs).probs
        greedy_actions = torch.argmax(probs, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        logits = self.backbone(obs)
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs)


class QuantileQNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        hidden_layers: int,
        norm: str = None,
        hidden_activation: str = "leakyrelu",
        output_activation: str = "identity",
        n_quantile: int = 200,
    ):
        super().__init__()
        self.n = n_quantile
        self.action_dim = action_dim
        self.q_backbone = MLP(
            obs_dim,
            1024,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation,
            output_activation,
        )
        self.quantile_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, n_quantile * action_dim),
        )

    def forward(self, obs):
        batch_size = obs.shape[0]
        out = self.q_backbone(obs)
        out = self.quantile_head(out).view(batch_size, self.n, self.action_dim)
        return out

    def q_value(self, obs):
        quantile = self(obs)
        q = quantile.mean(dim=1)
        return q


class QPolicyWrapper(nn.Module, DiscretePolicy):
    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net

    def policy_infer(self, obs):
        q_values = self.q_net(obs)
        greedy_actions = torch.argmax(q_values, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        return self.policy_infer(obs)


class QuantileQPolicyWrapper(nn.Module, DiscretePolicy):
    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net

    def policy_infer(self, obs):
        q_values = self.q_net.q_value(obs)
        greedy_actions = torch.argmax(q_values, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        return self.policy_infer(obs)
