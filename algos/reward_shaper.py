import torch
from copy import deepcopy
from loguru import logger
import numpy as np
from tqdm import tqdm

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.continuous import GaussianActor
from offlinerl.utils.net.common import MLP
from offlinerl.utils.exp import setup_seed


def algo_init(args):
    logger.info("Run algo_init function")

    setup_seed(args["seed"])

    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
        max_action = args["max_action"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape, get_env_action_range

        obs_shape, action_shape = get_env_shape(args["task"])
        max_action, _ = get_env_action_range(args["task"])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError

    actor = GaussianActor(
        obs_shape, action_shape, args["actor_features"], args["actor_layers"]
    ).to(args["device"])

    actor_optim = torch.optim.Adam(actor.parameters(), lr=args["actor_lr"])

    shaping_net = MLP(
        obs_shape,
        1,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
        output_activation="tanh",
    ).to(args["device"])

    shaping_optim = torch.optim.Adam(
        shaping_net.parameters(), lr=args["shaping_lr"]
    )
    return {
        "actor": {"net": actor, "opt": actor_optim},
        "shaping_net": {"net": shaping_net, "opt": shaping_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.policy_mode = args["policy_mode"]
        self.shaping_version = args["shaping_version"]
        self.actor = algo_init["actor"]["net"]
        self.actor_optim = algo_init["actor"]["opt"]

        self.shaping_net = algo_init["shaping_net"]["net"]
        self.shaping_optim = algo_init["shaping_net"]["opt"]

        self.batch_size = self.args["batch_size"]
        self.device = self.args["device"]

        self.best_actor = deepcopy(self.actor)
        self.best_loss = float("inf")

    def train(
        self,
        train_buffer,
        val_buffer,
        callback_fn,
    ):
        if self.policy_mode != "random":
            logger.info(f"{'*' * 30}Train BC policy start ....")
            if self.args["bc_policy_path"] is not None:
                self.actor = torch.load(
                    self.args["bc_policy_path"], map_location="cpu"
                ).to(self.device)
            else:
                self.train_bc_policy(train_buffer, callback_fn)
                if self.args["bc_policy_save_path"] is not None:
                    torch.save(self.actor, self.args["bc_policy_save_path"])

            logger.info(f"{'*' * 30}Train BC policy finish.")

        # self.actor.requires_grad_(False)
        # prepare data
        obs = torch.from_numpy(train_buffer["obs"]).to(self.device)
        act = torch.from_numpy(train_buffer["act"]).to(self.device)
        ret = torch.from_numpy(train_buffer["ret"]).to(self.device)

        max_returns = ret.max()
        min_returns = ret.min()
        ret = (
            2 * ((ret - min_returns) / (max_returns - min_returns + 1e-6)) - 1
        )

        action_dist = self.actor(obs)
        log_probs = action_dist.log_prob(act).sum(-1, keepdim=True)
        logp_grad_norms = self._calc_grads(log_probs)
        max_grad_norms = logp_grad_norms.max()
        min_grad_norms = logp_grad_norms.min()
        logp_grad_norms = (
            2
            * (
                (logp_grad_norms - min_grad_norms)
                / (max_grad_norms - min_grad_norms + 1e-6)
            )
            - 1
        )

        train_buffer["logp_grad_norms"] = logp_grad_norms.cpu().numpy()
        train_buffer["returns"] = ret.cpu().numpy()

        logger.info(f"{'*' * 30}Train reward shaping net start ...")
        if self.shaping_version == "v1":
            """Origin target"""
            logger.info("train shaping net v1")
            self.train_shaping_net_v1(
                train_buffer, val_buffer, log_probs, callback_fn
            )
        elif self.shaping_version == "v2":
            """Approximation target"""
            logger.info("train shaping net v2")
            self.train_shaping_net_v2(train_buffer, val_buffer, callback_fn)
        else:
            raise NotImplementedError()

        logger.info(f"{'*' * 30}Train reward shaping net finish.")

    def train_bc_policy(self, buffer, callback_fn):
        data_size = len(buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(
            range(data_size), (train_size, val_size)
        )
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.batch_size

        idxs = np.arange(train_buffer.shape[0])
        for epoch in range(100):
            np.random.shuffle(idxs)
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[
                    batch_num * batch_size : (batch_num + 1) * batch_size
                ]
                batch = train_buffer[batch_idxs]
                self._train_bc_policy(self.actor, batch, self.actor_optim)

            val_loss = self._eval_bc_policy(self.actor, valdata)
            logger.debug(f"val loss: {val_loss}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_actor.load_state_dict(self.actor.state_dict())

            res = callback_fn(self.get_policy())
            res["loss"] = val_loss
            self.log_res(epoch, res)

        return self.get_policy()

    def train_shaping_net_v1(
        self, train_buffer, val_buffer, log_probs, callback_fn
    ):
        """

        Args:
            train_buffer (DataLoader): Batched Traj Loader
            val_buffer (DataLoader or None):
            callback_fn (Callable): callback function
        """
        train_buffer.to_torch(device=self.device)
        data_size = len(train_buffer)
        val_size = min(int(data_size * 0.90) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(
            range(data_size), (train_size, val_size)
        )
        train_buffer["logp"] = log_probs
        valdata = train_buffer[val_splits.indices]
        train_buffer = train_buffer[train_splits.indices]
        batch_size = self.batch_size

        idxs = np.arange(train_buffer.shape[0])
        for epoch in range(self.args["max_epoch"]):
            np.random.shuffle(idxs)
            for batch_num in tqdm(
                range(int(np.ceil(idxs.shape[-1] / batch_size)))
            ):
                batch_idxs = idxs[
                    batch_num * batch_size : (batch_num + 1) * batch_size
                ]
                batch = train_buffer[batch_idxs]
                self._train_shaping_net(
                    self.shaping_net, batch, self.shaping_optim
                )

            val_loss, shaping_reward = self._eval_shaping_net(
                self.shaping_net, valdata
            )
            res = {"loss": val_loss, "shaping_reward": shaping_reward}
            self.log_res(epoch, res)

    def _train_shaping_net(self, shaping_net, data, optim):
        obs = data["obs"]
        ret = data["returns"]
        logp = data["logp"]
        shaping_reward = ret - shaping_net(obs)

        _, logp_grads = self._calc_grads(logp, True)
        loss = -torch.norm((logp_grads * shaping_reward).sum(dim=0), p=2)

        optim.zero_grad()
        loss.backward()
        optim.step()

    def _calc_grads(self, log_probs, calc_logp_grads=False):
        logp_grad_norms = torch.empty_like(log_probs)
        if calc_logp_grads:
            logp_grads = []

        for t in range(log_probs.shape[0]):
            self.actor.zero_grad()
            grads = torch.autograd.grad(
                log_probs[t, 0],
                self.actor.parameters(),
                retain_graph=True,
            )
            flat_grads = torch.cat([grad.view(-1) for grad in grads])
            if calc_logp_grads:
                logp_grads.append(flat_grads.unsqueeze(dim=0))
            logp_grad_norms[t, 0] = torch.norm(flat_grads, p=2)

        if calc_logp_grads:
            return logp_grad_norms, torch.cat(logp_grads)
        return logp_grad_norms

    def _eval_shaping_net(self, shaping_net, valdata):
        valdata.to_torch(device=self.device)
        obs = valdata["obs"]
        ret = valdata["returns"]
        logp = valdata["logp"]
        _, logp_grads = self._calc_grads(logp, True)

        shaping_reward = ret - shaping_net(obs)
        val_loss = -torch.norm((logp_grads * shaping_reward).sum(dim=0), p=2)
        return val_loss.item(), shaping_reward.mean().item()

    def train_shaping_net_v2(self, train_buffer, val_buffer, callback_fn):
        train_buffer.to_torch(device=self.device)

        data_size = len(train_buffer)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(
            range(data_size), (train_size, val_size)
        )
        valdata = train_buffer[val_splits.indices]
        train_buffer = train_buffer[train_splits.indices]
        batch_size = self.batch_size

        idxs = np.arange(train_buffer.shape[0])
        for epoch in range(self.args["max_epoch"]):
            np.random.shuffle(idxs)
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[
                    batch_num * batch_size : (batch_num + 1) * batch_size
                ]
                batch = train_buffer[batch_idxs]
                obs = batch["obs"]
                ret = batch["returns"]
                logp_grad_norms = batch["logp_grad_norms"]
                shaping_reward = ret - self.shaping_net(obs)
                loss = (logp_grad_norms - shaping_reward).pow(2).mean()
                self.shaping_net.zero_grad()
                loss.backward()
                self.shaping_optim.step()

            # eval on valdata
            obs = valdata["obs"]
            ret = valdata["returns"]
            logp_grad_norms = valdata["logp_grad_norms"]
            shaping_reward = ret - self.shaping_net(obs)
            loss = (logp_grad_norms - shaping_reward).pow(2).mean()

            res = {"loss": loss.item()}
            self.log_res(epoch, res)

    def _train_bc_policy(self, policy, data, optim):
        data.to_torch(device=self.device)
        obs = data["obs"]
        action = data["act"]

        action_dist = policy(obs)
        loss = -action_dist.log_prob(action).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    def _eval_bc_policy(self, policy, valdata):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            obs = valdata["obs"]
            action = valdata["act"]

            action_dist = policy(obs)
            val_loss = ((action_dist.mean - action) ** 2).mean().item()
            return val_loss

    def get_policy(self):
        return self.best_actor

    def get_model(self):
        return self.shaping_net

    def save_model(self, model_path):
        torch.save(self.get_model(), model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        return model
