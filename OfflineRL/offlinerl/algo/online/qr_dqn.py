import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import (
    Batch,
    LoggedReplayBuffer,
    TrajAveragedReplayBuffer,
)
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.env import get_env
from offlinerl.utils.net.discrete import QuantileQPolicyWrapper, QuantileQNet
from offlinerl.utils.function import get_linear_fn


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

    q = QuantileQNet(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
        n_quantile=args["num_quantiles"],
    ).to(args["device"])
    critic_optim = torch.optim.Adam(q.parameters(), lr=args["lr"])

    env = get_env(args["task"])

    exploration_schedule = get_linear_fn(
        args["exploration_init_eps"],
        args["exploration_final_eps"],
    )

    return {
        "env": env,
        "critic": {"net": q, "opt": critic_optim},
        "exploration_schedule": exploration_schedule,
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.env = algo_init["env"]
        self.q = algo_init["critic"]["net"]
        self.target_q = deepcopy(self.q)
        self.actor = QuantileQPolicyWrapper(self.q)
        self.critic_optim = algo_init["critic"]["opt"]
        self.exploration_schedule = algo_init["exploration_schedule"]

        self.exploration_rate = self.exploration_schedule(1.0)
        self.total_train_steps = 0
        self.train_epoch = 0
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.device = args["device"]

        tau = torch.linspace(0, 1, self.args["num_quantiles"] + 1)
        self.tau = ((tau[:-1] + tau[1:]) / 2).view(1, -1, 1).to(self.device)

        if self.args["buffer_type"] == "log_transition":
            self.replay_buffer = LoggedReplayBuffer(
                self.args["buffer_size"],
                log_path=f'{self.args["log_data_path"]}/QRDQN/{self.env.spec.id}',
            )
        elif self.args["buffer_type"] == "avg_traj":
            self.replay_buffer = TrajAveragedReplayBuffer(
                self.args["buffer_size"],
            )

    def train(self, callback_fn):
        self.train_policy(callback_fn)

    def get_policy(self):
        return self.q

    def train_policy(self, callback_fn):

        while (
            self.train_epoch <= self.args["max_epoch"]
            or self.total_train_steps <= self.args["max_step"]
        ):
            metrics = {
                "epoch": self.train_epoch,
                "step": self.total_train_steps,
            }
            # collect data
            obs = self.env.reset()
            traj_batch = None
            while True:
                if np.random.rand() < self.exploration_rate:
                    action = np.array([self.env.action_space.sample()])
                else:
                    with torch.no_grad():
                        obs_t = torch.tensor(obs, device=self.device).float()
                        obs_t = obs_t.unsqueeze(0)

                        action = self.actor(obs_t)[0].long()
                        action = action.cpu().numpy()

                new_obs, reward, done, info = self.env.step(action)
                batch_data = Batch(
                    {
                        "obs": np.expand_dims(obs, 0),
                        "act": np.expand_dims(action, 0),
                        "rew": [reward],
                        "done": [done],
                        "obs_next": np.expand_dims(new_obs, 0),
                        "retention": [info["reward"]["retention"]],
                    }
                )
                if isinstance(self.replay_buffer, LoggedReplayBuffer):
                    self.replay_buffer.put(batch_data)
                else:
                    if traj_batch is None:
                        traj_batch = batch_data
                    else:
                        traj_batch = Batch.cat([traj_batch, batch_data])

                if done:
                    break
                obs = new_obs

                self.total_train_steps += 1
                if self.total_train_steps >= self.args["warmup_size"]:
                    batch = self.replay_buffer.sample(self.args["batch_size"])
                    batch.to_torch(device=self.device)

                    dqn_metrics = self._qr_dqn_update(batch)
                    metrics.update(dqn_metrics)

            if isinstance(self.replay_buffer, TrajAveragedReplayBuffer):
                self.replay_buffer.put(traj_batch)

            if (
                self.train_epoch == 0
                or (self.train_epoch + 1) % self.args["eval_epoch"] == 0
            ):
                res = callback_fn(self.actor)
                metrics.update(res)

            self.log_res(self.train_epoch, metrics)
            self.train_epoch += 1

        return self.get_policy()

    def _calc_quantiles(self, network, obs, actions):
        # update critic
        cur_quantiles = network(obs)
        action_idx = actions.unsqueeze(-1).expand(
            -1, self.args["num_quantiles"], -1
        )
        _q = cur_quantiles.gather(-1, action_idx.long())
        return _q

    def _qr_dqn_update(self, batch_data):
        obs = batch_data["obs"]
        action = batch_data["act"]
        next_obs = batch_data["obs_next"]
        reward = (
            batch_data["rew"]
            .view(-1, 1, 1)
            .repeat(1, self.args["num_quantiles"], 1)
        )
        done = (
            batch_data["done"]
            .view(-1, 1, 1)
            .repeat(1, self.args["num_quantiles"], 1)
        )

        # update critic
        # [batch, N, 1]
        cur_quantiles = self._calc_quantiles(self.q, obs, action)

        with torch.no_grad():
            next_q = self.target_q.q_value(next_obs)
            next_action = torch.argmax(next_q, dim=-1, keepdim=True)
            next_quantiles = self._calc_quantiles(
                self.target_q, next_obs, next_action
            )
            y = reward + self.args["discount"] * (1 - done) * next_quantiles

        huber_loss = self.loss_fn(y, cur_quantiles)

        with torch.no_grad():
            diff = y - cur_quantiles
            delta = (diff < 0).float()

        critic_loss = (torch.abs(self.tau - delta) * huber_loss).sum(1).mean()

        # print("next_q", next_q.shape)
        # print("next_a", next_action.shape)
        # print("next_quantiles", next_quantiles.shape)
        # print("y", y.shape)
        # print("cur_quantiles", cur_quantiles.shape)
        # print("tau", self.tau.shape)
        # print("delta", delta.shape)
        # print("huberloss", huber_loss.shape)

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

        self.exploration_rate = self.exploration_schedule(
            1.0 - 1.0 * self.train_epoch / self.args["max_epoch"]
        )
        self.actor.q_net = self.q

        # DEBUGGING INFORMATION
        metrics = {}
        metrics["mean_critic_loss"] = torch.mean(critic_loss).item()
        metrics["mean_huber_loss"] = torch.mean(huber_loss).item()
        metrics["mean_q_quantile"] = torch.mean(cur_quantiles).item()
        metrics["mean_next_q_quantile"] = torch.mean(next_quantiles).item()
        metrics["exploration_rate"] = self.exploration_rate
        return metrics
