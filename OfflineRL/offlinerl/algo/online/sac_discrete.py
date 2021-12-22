# MOPO: Model-based Offline Policy Optimization
# https://arxiv.org/abs/2005.13239
# https://github.com/tianheyu927/mopo

import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.sac_policy import CategoricalPolicy
from offlinerl.utils.env import get_env


def prob_log_prob(dist, eps=1e-6):
    probs = dist.probs
    z = (probs == 0.0).float() * eps
    log_probs = torch.log(probs + z)
    return probs, log_probs


def algo_init(args):
    logger.info("Run algo_init function")

    setup_seed(args["seed"])

    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape

        obs_shape, action_shape = get_env_shape(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError

    net_a = Net(
        layer_num=args["hidden_layers"],
        state_shape=obs_shape,
        hidden_layer_size=args["hidden_layer_size"],
    )

    actor = CategoricalPolicy(
        preprocess_net=net_a,
        action_num=action_shape,
        hidden_layer_size=args["hidden_layer_size"],
    ).to(args["device"])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args["actor_lr"])

    log_alpha = torch.zeros(1, requires_grad=True, device=args["device"])
    alpha_optimizer = torch.optim.Adam([log_alpha], lr=args["actor_lr"])

    q1 = MLP(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="swish",
    ).to(args["device"])
    q2 = MLP(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="swish",
    ).to(args["device"])
    critic_optim = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()], lr=args["actor_lr"]
    )

    env = get_env(args["task"])

    return {
        "env": env,
        "actor": {"net": actor, "opt": actor_optim},
        "log_alpha": {"net": log_alpha, "opt": alpha_optimizer},
        "critic": {"net": [q1, q2], "opt": critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.env = algo_init["env"]
        self.actor = algo_init["actor"]["net"]
        self.actor_optim = algo_init["actor"]["opt"]

        self.log_alpha = algo_init["log_alpha"]["net"]
        self.log_alpha_optim = algo_init["log_alpha"]["opt"]

        self.q1, self.q2 = algo_init["critic"]["net"]
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init["critic"]["opt"]

        self.device = args["device"]

        self.args["target_entropy"] = np.log(self.args["action_shape"])

    def train(self, train_buffer, val_buffer, callback_fn):
        self.train_policy(train_buffer, val_buffer, callback_fn)

    def get_policy(self):
        return self.actor

    def train_policy(self, train_buffer, val_buffer, callback_fn):
        buffer = ModelBuffer(self.args["buffer_size"])

        for epoch in range(self.args["max_epoch"]):
            metrics = {"epoch": epoch}
            # collect data
            with torch.no_grad():
                obs = self.env.reset()
                obs_t = torch.tensor(obs, device=self.device)
                while True:
                    action = self.actor(obs_t).sample()
                    new_obs, reward, done, _ = self.env.step(action)

                    batch_data = Batch(
                        {
                            "obs": obs,
                            "act": action,
                            "rew": reward,
                            "ret": reward,
                            "done": done,
                            "obs_next": new_obs,
                        }
                    )
                    buffer.put(batch_data)

                    if done:
                        break
                    obs = new_obs

                # update
                for _ in range(self.args["steps_per_epoch"]):
                    batch = buffer.sample(self.args["batch_size"])
                    batch.to_torch(device=self.device)

                    sac_metrics = self._sac_update(batch)

                if epoch == 0 or (epoch + 1) % self.args["eval_epoch"] == 0:
                    res = callback_fn(self.get_policy())
                    metrics.update(res)

                metrics["reward"] = reward.mean().item()
                metrics["next_obs"] = new_obs.mean().item()

                metrics.update(sac_metrics)
            self.log_res(epoch, metrics)

        return self.get_policy()

    def _sac_update(self, batch_data):
        obs = batch_data["obs"]
        action = batch_data["act"]
        next_obs = batch_data["obs_next"]
        reward = batch_data["rew"]
        done = batch_data["done"]

        # update critic
        _q1 = self.q1(obs).gather(1, action.long())
        _q2 = self.q2(obs).gather(1, action.long())

        with torch.no_grad():
            next_action_dist = self.actor(next_obs)
            prob, log_prob = prob_log_prob(next_action_dist)

            _target_q1 = self.target_q1(next_obs)
            _target_q2 = self.target_q2(next_obs)
            alpha = torch.exp(self.log_alpha)
            y = reward + self.args["discount"] * (1 - done) * (
                prob * (torch.min(_target_q1, _target_q2) - alpha * log_prob)
            ).sum(dim=-1, keepdim=True)

        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            [*self.q1.parameters(), *self.q2.parameters()], max_norm=1.0
        )
        self.critic_optim.step()

        # soft target update
        self._sync_weight(
            self.target_q1,
            self.q1,
            soft_target_tau=self.args["soft_target_tau"],
        )
        self._sync_weight(
            self.target_q2,
            self.q2,
            soft_target_tau=self.args["soft_target_tau"],
        )

        if self.args["learnable_alpha"]:
            # update alpha
            entropy = -torch.sum(prob * log_prob, dim=-1, keepdim=True)
            alpha_loss = -torch.mean(
                self.log_alpha
                * (-entropy + self.args["target_entropy"]).detach()
            )

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                [self.log_alpha], max_norm=1.0
            )
            self.log_alpha_optim.step()

        # update actor
        action_dist = self.actor(obs)
        action_prob, action_log_prob = prob_log_prob(action_dist)

        entropy = -torch.sum(
            action_prob * action_log_prob, dim=-1, keepdim=True
        )
        q = torch.sum(
            action_prob * torch.min(self.q1(obs), self.q2(obs)),
            dim=-1,
            keepdim=True,
        )

        actor_loss = (-q - torch.exp(self.log_alpha) * entropy).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), max_norm=1.0
        )
        self.actor_optim.step()

        # DEBUGGING INFORMATION
        metrics = {}
        metrics["mean_action_log_prob"] = torch.mean(action_log_prob).item()
        metrics["mean_critic_loss"] = torch.mean(critic_loss).item()
        metrics["mean_Qmin"] = torch.mean(q).item()
        metrics["log_alpha"] = self.log_alpha.item()
        metrics["mean_obs"] = torch.mean(obs).item()
        metrics["mean_next_obs"] = torch.mean(next_obs).item()
        metrics["mean_alpha_loss"] = torch.mean(alpha_loss).item()
        metrics["mean_actor_loss"] = torch.mean(actor_loss).item()
        return metrics
