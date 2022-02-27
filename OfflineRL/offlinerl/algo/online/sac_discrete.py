import torch
import numpy as np
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import (
    Batch,
    LoggedReplayBuffer,
    TrajAveragedReplayBuffer,
)
from offlinerl.utils.net.common import MLP
from offlinerl.utils.net.discrete import CategoricalActor
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.env import get_env


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

        self.total_train_steps = 0
        self.train_epoch = 0

        if self.args["buffer_type"] == "log_transition":
            self.replay_buffer = LoggedReplayBuffer(
                self.args["buffer_size"],
                log_path=f'{self.args["log_data_path"]}/SACD/{self.env.spec.id}',
            )
        elif self.args["buffer_type"] == "avg_traj":
            self.replay_buffer = TrajAveragedReplayBuffer(
                self.args["buffer_size"],
            )

    def train(self, callback_fn):
        self.train_policy(callback_fn)

    def get_policy(self):
        return self.actor

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
                with torch.no_grad():
                    obs_t = torch.tensor(obs, device=self.device).float()
                    obs_t = obs_t.unsqueeze(0)
                    action = self.actor(obs_t).sample()
                    action = action.cpu().numpy()
                    new_obs, reward, done, info = self.env.step(action[0])

                if self.total_train_steps >= self.args["warmup_size"]:
                    for _ in range(self.args["steps_per_epoch"]):
                        batch = self.replay_buffer.sample(
                            self.args["batch_size"]
                        )
                        batch.to_torch(device=self.device)

                        sac_metrics = self._sac_update(batch)
                        metrics.update(sac_metrics)

                batch_data = Batch(
                    {
                        "obs": np.expand_dims(obs, 0),
                        "act": np.expand_dims(action, 0),
                        "rew": [reward],
                        "done": [done],
                        "obs_next": np.expand_dims(new_obs, 0),
                        # "retention": [info["reward"]["retention"]],
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

            if isinstance(self.replay_buffer, TrajAveragedReplayBuffer):
                self.replay_buffer.put(traj_batch)

            if (
                self.train_epoch == 0
                or (self.train_epoch + 1) % self.args["eval_epoch"] == 0
            ):
                res = callback_fn(self.get_policy())
                metrics.update(res)

            self.log_res(self.train_epoch, metrics)
            self.train_epoch += 1

        return self.get_policy()

    def _sac_update(self, batch_data):
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

        # print("action", action.shape)
        # print("reward", reward.shape)
        # print("done", done.shape)
        # print("entropy", entropy.shape)
        # print("target_q", target_q.shape)
        # print("y", y.shape)
        # print("q1", _q1.shape)

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
        actor_loss = -(q + alpha * entropy).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), max_norm=1.0
        )
        self.actor_optim.step()

        if self.args["learnable_alpha"]:
            # update alpha
            log_prob = -entropy.detach() + self.args["target_entropy"]
            alpha_loss = -(self.log_alpha * log_prob).mean()

            self.log_alpha_optim.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                [self.log_alpha], max_norm=1.0
            )
            self.log_alpha_optim.step()

        # DEBUGGING INFORMATION
        metrics = {}
        metrics["mean_action_prob"] = torch.mean(probs).item()
        metrics["mean_critic_loss"] = torch.mean(critic_loss).item()
        metrics["mean_Qmin"] = torch.mean(q).item()
        metrics["log_alpha"] = self.log_alpha.item()
        metrics["mean_alpha_loss"] = torch.mean(alpha_loss).item()
        metrics["mean_actor_loss"] = torch.mean(actor_loss).item()
        return metrics
