import torch
import torch.nn as nn
from copy import deepcopy
from loguru import logger
import numpy as np

from offlinerl.algo.base import BaseAlgo
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

    reward_net = MLP(
        obs_shape + action_shape,
        1,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="relu",
        # output_activation="tanh",
    ).to(args["device"])

    reward_optim = torch.optim.Adam(
        reward_net.parameters(), lr=args["reward_lr"]
    )
    return {
        "reward_net": {"net": reward_net, "opt": reward_optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.reward_net = algo_init["reward_net"]["net"]
        self.reward_optim = algo_init["reward_net"]["opt"]

        self.batch_size = self.args["batch_size"]
        self.device = self.args["device"]

        self.best_model = deepcopy(self.reward_net)
        self.best_loss = float("inf")

    def train(
        self,
        train_buffer,
        val_buffer,
        callback_fn,
    ):
        loss_fn = nn.MSELoss()

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
                action = batch["act"]
                rew = batch["rew"]
                obs_act = torch.cat([obs, action], dim=-1)
                pre_rew = self.reward_net(obs_act)

                loss = loss_fn(pre_rew, rew)
                self.reward_optim.zero_grad()
                loss.backward()
                self.reward_optim.step()

            with torch.no_grad():
                obs = valdata["obs"]
                action = valdata["act"]
                rew = valdata["rew"]
                obs_act = torch.cat([obs, action], dim=-1)
                pre_rew = self.reward_net(obs_act)

                val_loss = loss_fn(pre_rew, rew).item()
            logger.debug(f"val loss: {val_loss}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model.load_state_dict(self.best_model.state_dict())

            # res = callback_fn(self.get_policy())
            res = {}
            res["loss"] = val_loss
            self.log_res(epoch, res)

        return self.get_policy()

    def get_policy(self):
        return self.best_model

    def get_model(self):
        return self.best_model

    def save_model(self, model_path):
        torch.save(self.get_model(), model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)
        return model
