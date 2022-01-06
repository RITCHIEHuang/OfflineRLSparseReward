import torch
import torch.nn as nn
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

    q = MLP(
        obs_shape,
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
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
        self.actor = QPolicyWrapper(self.q)
        self.target_q = deepcopy(self.q)
        self.critic_optim = algo_init["critic"]["opt"]
        self.exploration_schedule = algo_init["exploration_schedule"]

        self.exploration_rate = self.exploration_schedule(1.0)
        self.total_train_steps = 0
        self.train_epoch = 0
        self.loss_fn = nn.SmoothL1Loss()

        self.device = args["device"]

    def train(self, callback_fn):
        self.train_policy(callback_fn)

    def get_policy(self):
        return self.q

    def train_policy(self, callback_fn):
        buffer = ModelBuffer(self.args["buffer_size"])

        while (
            self.train_epoch <= self.args["max_epoch"]
            or self.total_train_steps <= self.args["max_step"]
        ):
            metrics = {"epoch": self.train_epoch}
            # collect data
            obs = self.env.reset()
            while True:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, device=self.device).float()
                    obs_t = obs_t.unsqueeze(0)

                    if np.random.rand() < self.exploration_rate:
                        action = np.array([self.env.action_space.sample()])
                    else:
                        action = self.actor(obs_t)[0].long()
                        action = action.cpu().numpy()
                    new_obs, reward, done, _ = self.env.step(action)

                batch_data = Batch(
                    {
                        "obs": np.expand_dims(obs, 0),
                        "act": np.expand_dims(action, 0),
                        "rew": [reward],
                        "done": [done],
                        "obs_next": np.expand_dims(new_obs, 0),
                    }
                )
                buffer.put(batch_data)

                if done:
                    break
                obs = new_obs

                if len(buffer) >= self.args["warmup_size"]:
                    batch = buffer.sample(self.args["batch_size"])
                    batch.to_torch(device=self.device)

                    dqn_metrics = self._dqn_update(batch)
                    metrics.update(dqn_metrics)

            if (
                self.train_epoch == 0
                or (self.train_epoch + 1) % self.args["eval_epoch"] == 0
            ):
                res = callback_fn(self.actor)
                metrics.update(res)

            self.log_res(self.train_epoch, metrics)
            self.train_epoch += 1

        return self.get_policy()

    def _dqn_update(self, batch_data):
        self.total_train_steps += 1

        obs = batch_data["obs"]
        action = batch_data["act"]
        next_obs = batch_data["obs_next"]
        reward = batch_data["rew"].unsqueeze(-1)
        done = batch_data["done"].unsqueeze(-1)

        # update critic
        _q = self.q(obs).gather(-1, action.long())

        with torch.no_grad():
            next_q = self.target_q(next_obs)
            next_q, _ = next_q.max(dim=-1)
            next_q = next_q.reshape(-1, 1)
            y = reward + self.args["discount"] * (1 - done) * next_q

        critic_loss = self.loss_fn(_q, y)

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
        metrics["mean_Q"] = torch.mean(_q).item()
        metrics["exploration_rate"] = self.exploration_rate
        return metrics
