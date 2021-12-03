import torch
from torch import nn as nn
from torch.distributions import Normal

from offlinerl.utils.net.common import BasePolicy
from offlinerl.utils.net import continuous
from offlinerl.utils.net import discrete


class CategoricalPolicy(discrete.ActorProb, BasePolicy):
    def policy_infer(self, obs):
        probs = self(obs).probs
        greedy_actions = torch.argmax(probs, dim=-1, keepdim=True)
        return greedy_actions

    def get_action(self, obs):
        obs_tensor = torch.as_tensor(
            obs, device=next(self.parameters()).device, dtype=torch.float32
        )
        act = self.policy_infer(obs_tensor)
        act = act.detach().cpu().numpy()[0]
        return act


class GaussianPolicy(continuous.ActorProb, BasePolicy):
    LOG_SIG_MAX = 2.0
    LOG_SIG_MIN = -20.0

    def forward(
        self,
        obs,
        state=None,
        info={},
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        logits, h = self.preprocess(obs, state)
        mean = self.mu(logits)

        if self._c_sigma:
            log_std = torch.clamp(
                self.sigma(logits), min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX
            )
            std = log_std.exp()
        else:
            shape = [1] * len(mean.shape)
            shape[1] = -1
            log_std = self.sigma.view(shape) + torch.zeros_like(mean)
            std = log_std.exp()

        return Normal(mean, std)

    def policy_infer(self, obs):
        return torch.tanh(self(obs).mean)
