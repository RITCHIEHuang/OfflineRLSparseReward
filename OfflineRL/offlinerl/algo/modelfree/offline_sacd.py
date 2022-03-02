import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP
from offlinerl.utils.net.discrete import (
    CategoricalActor,
)
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

    actor = CategoricalActor(
        obs_dim=obs_shape,
        action_dim=action_shape,
        hidden_size=args["hidden_layer_size"],
        hidden_layers=args["hidden_layers"],
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
        hidden_activation="relu",
    ).to(args["device"])
    q2 = MLP(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
    ).to(args["device"])

    critic_optim = torch.optim.Adam(
        [*q1.parameters(), *q2.parameters()], lr=args["actor_lr"]
    )

    return {
        "actor": {"net": actor, "opt": actor_optim},
        "log_alpha": {"net": log_alpha, "opt": alpha_optimizer},
        "critic": {"net": [q1, q2], "opt": critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.actor = algo_init["actor"]["net"]
        self.actor_optim = algo_init["actor"]["opt"]

        self.log_alpha = algo_init["log_alpha"]["net"]
        self.log_alpha_optim = algo_init["log_alpha"]["opt"]

        self.q1, self.q2 = algo_init["critic"]["net"]
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)
        self.critic_optim = algo_init["critic"]["opt"]

        self.device = args["device"]

        self.args["target_entropy"] = np.log(self.args["action_shape"]) * 0.98

        self.total_train_steps = 0
        self.train_epoch = 0

    def train(self, train_buffer, val_buffer, callback_fn):
        for epoch in range(self.args["max_epoch"]):
            metrics = {
                "epoch": epoch,
                "step": self.total_train_steps,
            }
            for step in range(1, self.args["steps_per_epoch"] + 1):
                train_data = train_buffer.sample(self.args["batch_size"])
                res = self._train(train_data)
                metrics.update(res)
            if epoch == 0 or (epoch + 1) % self.args["eval_epoch"] == 0:
                res = callback_fn(self.get_policy())
                metrics.update(res)
            self.log_res(epoch, metrics, save_model=False)

        return self.get_policy()

    def get_policy(self):
        return self.actor

    def _train(self, batch_data):
        batch_data.to_torch(dtype=torch.float32, device=self.device)
        obs = batch_data["obs"]
        action = batch_data["act"]
        next_obs = batch_data["obs_next"]
        reward = batch_data["rew"].unsqueeze(-1)
        done = batch_data["done"].unsqueeze(-1)

        # update critic
        _q1 = self.q1(obs).gather(-1, action.long())
        _q2 = self.q2(obs).gather(-1, action.long())

        with torch.no_grad():
            alpha = torch.exp(self.log_alpha)
            next_action_dist = self.actor(next_obs)
            probs = next_action_dist.probs
            entropy = next_action_dist.entropy().unsqueeze(-1)
            _target_q1 = self.target_q1(next_obs)
            _target_q2 = self.target_q2(next_obs)
            target_q = probs * torch.min(_target_q1, _target_q2)

            y = reward + self.args["discount"] * (1 - done) * (
                target_q.sum(dim=-1, keepdim=True) + alpha * entropy
            )

        critic_loss = ((y - _q1) ** 2).mean() + ((y - _q2) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # soft target update
        if self.total_train_steps % self.args["target_update_interval"] == 0:
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

        action_dist = self.actor(obs)
        probs = action_dist.probs
        entropy = action_dist.entropy().unsqueeze(-1)

        # update actor
        with torch.no_grad():
            q1 = self.q1(obs)
            q2 = self.q2(obs)
            q = torch.min(q1, q2)
            alpha = torch.exp(self.log_alpha)

        q = torch.sum(q * probs, dim=-1, keepdim=True)
        actor_loss = (-q - alpha * entropy).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        if self.args["learnable_alpha"]:
            # update alpha
            entropy_error = -entropy.detach() + self.args["target_entropy"]
            alpha_loss = -(self.log_alpha * entropy_error).mean()

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optim.step()

        # DEBUGGING INFORMATION
        metrics = {}
        metrics["mean_action_prob"] = torch.mean(probs).item()
        metrics["mean_critic_loss"] = torch.mean(critic_loss).item()
        metrics["mean_entropy"] = torch.mean(entropy).item()
        metrics["mean_Qmin"] = torch.mean(q).item()
        metrics["log_alpha"] = self.log_alpha.item()
        metrics["mean_alpha_loss"] = torch.mean(alpha_loss).item()
        metrics["mean_actor_loss"] = torch.mean(actor_loss).item()
        return metrics
