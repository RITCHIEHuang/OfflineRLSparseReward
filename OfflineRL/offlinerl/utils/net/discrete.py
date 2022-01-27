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
    def __init__(self, obs_dim, action_dim, hidden_size, hidden_layers, hidden_activation='leakyrelu'):
        super().__init__()
        self.backbone = MLP(
            in_features=obs_dim,
            out_features=hidden_size,
            hidden_features=hidden_size,
            hidden_layers=hidden_layers,
            hidden_activation=hidden_activation
        )
        self.front = MLP(in_features=hidden_size,out_features=action_dim,hidden_layers=1,hidden_features=action_dim)

    def policy_infer(self, obs):
        probs = self(obs).probs
        greedy_actions = torch.argmax(probs, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        emb= self.backbone(obs)
        logits = self.front(emb)
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs)


def combine_function(raw_outputs, combine_type):
    # raw_outputs [batch, N, action]
    N = raw_outputs.shape[1]
    if combine_type == "identity":
        return raw_outputs
    elif combine_type == "random":
        # REM random convex combination
        stochastic_matrix = torch.rand((N, 1), device=raw_outputs.device)
        stochastic_matrix /= torch.norm(
            stochastic_matrix, p=1, dim=0, keepdim=True
        )  # [K, convex]
        q_convex = torch.einsum(
            "bka,kc->bca", raw_outputs, stochastic_matrix
        )  # [batch, 1, action]

        return q_convex
    elif combine_type == "mean":
        # [batch, action]
        return torch.mean(raw_outputs, dim=1)
    elif combine_type == "min":
        # [batch, action]
        return torch.min(raw_outputs, dim=1)
    elif combine_type == "max":
        # [batch, action]
        return torch.max(raw_outputs, dim=1)
    else:
        raise NotImplementedError()


class MultiHeadQNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        hidden_layers: int,
        norm: str = None,
        hidden_activation: str = "leakyrelu",
        output_activation: str = "identity",
        n_head: int = 20,
        combine_type: str = "identity",  # ["identity", "random", "min", "mean", "max"]
    ):
        super().__init__()
        self.n_head = n_head
        self.combine_type = combine_type

        self.action_dim = action_dim
        self.q_backbone = MLP(
            obs_dim,
            n_head * action_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation,
            output_activation,
        )

    def forward(self, obs):
        out = self.q_backbone(obs).view(-1, self.n_head, self.action_dim)
        return out

    def q_convex(self, obs):
        unorder_outs = self(obs)
        return combine_function(unorder_outs, "random")

    def q_value(self, obs):
        unorder_outs = self(obs)
        return combine_function(unorder_outs, "mean")


class MultiQNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        hidden_layers: int,
        norm: str = None,
        hidden_activation: str = "leakyrelu",
        output_activation: str = "identity",
        num_networks: int = 10,
        combine_type: str = "random",  # ["identity", "random", "min", "mean", "max"]
    ):
        super().__init__()
        self.num_networks = num_networks
        self.action_dim = action_dim
        self.combine_type = combine_type

        self.qnets = [
            MLP(
                obs_dim,
                action_dim,
                hidden_size,
                hidden_layers,
                norm,
                hidden_activation,
                output_activation,
            )
            for _ in range(num_networks)
        ]

    def forward(self, obs):
        unorder_outs = [qnet(obs) for qnet in self.qnets]
        unorder_outs = torch.stack(unorder_outs, dim=1)  # [batch, K, action]
        return unorder_outs

    def q_convex(self, obs):
        unorder_outs = self(obs)
        return combine_function(unorder_outs, "random")

    def q_value(self, obs):
        unorder_outs = self(obs)
        return combine_function(unorder_outs, "mean")


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
class QPolicyWrapperWithFront(nn.Module, DiscretePolicy):
    def __init__(self, q_net,front):
        super().__init__()
        self.q_net = q_net
        self.front=front

    def policy_infer(self, obs):
        with torch.no_grad():
            emb = self.front(obs)
        q_values = self.q_net(emb)
        greedy_actions = torch.argmax(q_values, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        return self.policy_infer(obs)


class MultiHeadQPolicyWrapper(nn.Module, DiscretePolicy):
    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net

    def policy_infer(self, obs):
        q_values = self.q_net.q_value(obs)
        greedy_actions = torch.argmax(q_values, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        return self.policy_infer(obs)


class MultiNetQPolicyWrapper(nn.Module, DiscretePolicy):
    def __init__(self, q_net):
        super().__init__()
        self.q_net = q_net

    def policy_infer(self, obs):
        q_values = self.q_net.q_value(obs)
        greedy_actions = torch.argmax(q_values, dim=-1, keepdim=True)
        return greedy_actions

    def forward(self, obs):
        return self.policy_infer(obs)
