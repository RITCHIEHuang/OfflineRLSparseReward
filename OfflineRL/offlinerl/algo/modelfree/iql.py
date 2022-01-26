# Offline Reinforcement Learning with Implicit Q-Learning
# https://arxiv.org/abs/2110.06169
# https://github.com/ikostrikov/implicit_q_learning

# Refer: https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/iql_trainer.py

import copy

import torch
import numpy as np
from torch import nn
from torch import optim
from loguru import logger
import os

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP
from offlinerl.utils.net.continuous import GaussianActor
from offlinerl.utils.exp import setup_seed


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

    actor = GaussianActor(
        obs_dim=obs_shape,
        action_dim=action_shape,
        hidden_size=args["hidden_layer_size"],
        hidden_layers=args["hidden_layers"],
    ).to(args["device"])

    actor_optim = optim.Adam(actor.parameters(), lr=args["actor_lr"])

    q1 = MLP(
        obs_shape + action_shape,
        1,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
    ).to(args["device"])
    q2 = MLP(
        obs_shape + action_shape,
        1,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
    ).to(args["device"])
    qf_optim = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()], lr=args["critic_lr"]
    )
    v = MLP(
        obs_shape,
        1,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
    ).to(args["device"])
    vf_optim = torch.optim.Adam(v.parameters(), lr=args["critic_lr"])

    nets = {
        "actor": {"net": actor, "opt": actor_optim},
        "qf_critic": {"net": [q1, q2], "opt": qf_optim},
        "vf_critic": {"net": v, "opt": vf_optim},
    }

    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]

        self.qf1, self.qf2 = algo_init["qf_critic"]["net"]
        self.qf_opt = algo_init["qf_critic"]["opt"]
        self.qf1_target = copy.deepcopy(self.qf1)
        self.qf2_target = copy.deepcopy(self.qf2)

        self.vf = algo_init["vf_critic"]["net"]
        self.vf_opt = algo_init["vf_critic"]["opt"]

        self.critic_criterion = nn.MSELoss()

        self._n_train_steps_total = 0
        self._current_epoch = 0
        if self.args["load_model"] and self.args["load_epoch"]:
            self.load_model(os.path.join(self.models_save_dir, str(self.args["load_epoch"]) + ".pt"))

    def _train(self, batch):
        self._current_epoch += 1
        self._n_train_steps_total += 1
        batch = batch.to_torch(dtype=torch.float32, device=self.args["device"])
        rewards = batch.rew
        terminals = batch.done
        obs = batch.obs
        actions = batch.act
        next_obs = batch.obs_next

        """
        QF Loss
        """
        obs_act = torch.cat([obs, actions], dim=-1)
        q1_pred = self.qf1(obs_act)
        q2_pred = self.qf2(obs_act)
        target_vf_pred = self.vf(next_obs).detach()

        q_target = (
            rewards
            + (1.0 - terminals) * self.args["discount"] * target_vf_pred
        )

        qf1_loss = self.critic_criterion(q1_pred, q_target)
        qf2_loss = self.critic_criterion(q2_pred, q_target)

        qf_loss = qf1_loss + qf2_loss

        """
        VF Loss
        """
        q_pred = torch.min(
            self.qf1_target(obs_act),
            self.qf2_target(obs_act),
        ).detach()
        vf_pred = self.vf(obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.args["quantile"] + vf_sign * (
            1 - self.args["quantile"]
        )
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        """
        Policy Loss
        """

        dist = self.actor(obs)
        log_pi = dist.log_prob(actions).sum(dim=-1, keepdim=True)

        if self._current_epoch < self.args["policy_bc_steps"]:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k
            gradient steps here, or not having it
            """
            policy_loss = -log_pi.mean()
        else:
            adv = q_pred - vf_pred
            exp_adv = torch.exp(adv / self.args["beta"])
            if self.args["clip_score"] is not None:
                exp_adv = torch.clamp(exp_adv, max=self.args["clip_score"])

            weights = exp_adv.detach()
            policy_loss = (-log_pi * weights).mean()

        if self._n_train_steps_total % self.args["q_update_period"] == 0:
            self.qf_opt.zero_grad()
            qf_loss.backward()
            self.qf_opt.step()

            self.vf_opt.zero_grad()
            vf_loss.backward()
            self.vf_opt.step()

        if self._n_train_steps_total % self.args["policy_update_period"] == 0:
            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

        """
        Soft Updates target network
        """
        if self._n_train_steps_total % self.args["target_update_period"] == 0:
            self._sync_weight(
                self.qf1_target, self.qf1, self.args["soft_target_tau"]
            )
            self._sync_weight(
                self.qf2_target, self.qf2, self.args["soft_target_tau"]
            )

        metrics = dict(
            log_pi=log_pi.mean().item(),
            policy_loss=policy_loss.item(),
            qf1_loss=qf1_loss.item(),
            qf2_loss=qf2_loss.item(),
            vf_loss=vf_loss.item(),
            average_qf1=q1_pred.mean().item(),
            average_qf2=q2_pred.mean().item(),
            average_vf=target_vf_pred.mean().item(),
            total_steps=self._n_train_steps_total,
        )
        return metrics

    def get_model(self):
        return self.actor

    def get_policy(self):
        return self.actor

    def train(self, train_buffer, val_buffer, callback_fn):
        for epoch in range(self.args["max_epoch"]):
            metrics = {"epoch": epoch}
            for step in range(1, self.args["steps_per_epoch"] + 1):
                train_data = train_buffer.sample(self.args["batch_size"])
                res = self._train(train_data)
                metrics.update(res)
            if epoch == 0 or (epoch + 1) % self.args["eval_epoch"] == 0:
                res = callback_fn(self.get_policy())
                metrics.update(res)
            self.log_res(epoch, metrics)

        return self.get_policy()
