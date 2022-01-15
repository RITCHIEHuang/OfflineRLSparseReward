import torch
import torch.nn as nn
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.net.discrete import MultiHeadQNet, MultiHeadQPolicyWrapper


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

    q = MultiHeadQNet(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
        n_head=args["num_heads"],
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
        self.actor = MultiHeadQPolicyWrapper(self.q)
        self.target_q = deepcopy(self.q)
        self.critic_optim = algo_init["critic"]["opt"]

        self.total_train_steps = 0
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
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
            self.log_res(epoch, metrics, save_model=False)

        return self.get_policy()

    def get_policy(self):
        return self.actor

    def _calc_q(self, network, obs, actions):
        # update critic
        cur_convex = network.q_convex(obs)  # [batch, convex, action]
        action_idx = actions.unsqueeze(-1).expand(-1, 1, -1)
        _q = cur_convex.gather(-1, action_idx.long())
        return _q

    def _train(self, batch):
        self.total_train_steps += 1

        batch = batch.to_torch(dtype=torch.float32, device=self.args["device"])
        obs = batch["obs"]
        action = batch["act"]
        next_obs = batch["obs_next"]
        reward = batch["rew"].view(-1, 1, 1)
        done = batch["done"].view(-1, 1, 1)

        # update critic
        # [batch, 1, 1]
        cur_q_values = self._calc_q(self.q, obs, action)

        with torch.no_grad():
            next_q = self.target_q.q_convex(next_obs)
            next_action = torch.argmax(next_q, dim=-1)
            next_q_values = self._calc_q(self.target_q, next_obs, next_action)
            y = reward + self.args["discount"] * (1 - done) * next_q_values

        critic_loss = self.loss_fn(y, cur_q_values).sum(1).mean()

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
        metrics["mean_q_value"] = torch.mean(cur_q_values).item()
        metrics["mean_next_q_value"] = torch.mean(next_q_values).item()
        return metrics
