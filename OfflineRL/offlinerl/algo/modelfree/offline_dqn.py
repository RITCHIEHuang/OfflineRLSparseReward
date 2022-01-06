import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import Batch
from offlinerl.utils.net.common import MLP
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.env import get_env
from offlinerl.utils.net.discrete import QPolicyWrapper


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

    q = MLP(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
    ).to(args["device"])
    critic_optim = torch.optim.Adam(q.parameters(), lr=args["lr"])

    return {
        "critic": {"net": q, "opt": critic_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.q = algo_init["critic"]["net"]
        self.actor = QPolicyWrapper(self.q)
        self.target_q = deepcopy(self.q)
        self.critic_optim = algo_init["critic"]["opt"]

        self.total_train_steps = 0

        self.device = args["device"]

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

    def get_policy(self):
        return self.q

    def _train(self, batch):
        self.total_train_steps += 1

        obs = batch["obs"]
        action = batch["act"]
        next_obs = batch["obs_next"]
        reward = batch["rew"].unsqueeze(-1)
        done = batch["done"].unsqueeze(-1)

        # update critic
        _q = self.q(obs).gather(-1, action.long())

        with torch.no_grad():
            next_q = self.target_q(next_obs)
            next_q, _ = next_q.max(dim=-1)
            next_q = next_q.reshape(-1, 1)
            y = reward + self.args["discount"] * (1 - done) * next_q

        critic_loss = ((y - _q) ** 2).mean()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.q.parameters(), max_norm=1.0
        )
        self.critic_optim.step()

        if self.total_train_steps % self.args["target_update_interval"] == 0:
            # soft target update
            self._sync_weight(
                self.target_q,
                self.q,
                soft_target_tau=self.args["soft_target_tau"],
            )

        self.actor.q_net = self.q

        # DEBUGGING INFORMATION
        metrics = {}
        metrics["mean_critic_loss"] = torch.mean(critic_loss).item()
        metrics["mean_Q"] = torch.mean(_q).item()
        metrics["exploration_rate"] = self.exploration_rate
        return metrics
