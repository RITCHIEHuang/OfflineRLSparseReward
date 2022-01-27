import torch
from copy import deepcopy
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.net.discrete import QPolicyWrapperWithFront
from offlinerl.utils.net.discrete import CategoricalActor


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
    bc_actor = CategoricalActor(
        obs_shape, action_shape, args["hidden_layer_size"], args["hidden_layers"],hidden_activation='tanh'
    ).to(args["device"])

    q = MLP(
        args["hidden_layer_size"],
        action_shape,
        args["hidden_layer_size"],
        args["hidden_layers"],
        norm=None,
        hidden_activation="tanh",
    ).to(args["device"])
    critic_optim = torch.optim.Adam(q.parameters(), lr=args["lr"])
    actor_optim = torch.optim.Adam(bc_actor.parameters(), lr=args["lr"])

    return {
            "critic": {"net": q, "opt": critic_optim,'bca':bc_actor},
            "actor":{"opt":actor_optim}
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.q = algo_init["critic"]["net"]
        self.bca = algo_init['critic']['bca']
        self.actor = QPolicyWrapperWithFront(self.q,self.bca.backbone)
        self.target_q = deepcopy(self.q)
        self.bcactor_optim = algo_init["actor"]["opt"]
        self.critic_optim = algo_init["critic"]["opt"]
        self.loss_fn = torch.nn.CrossEntropyLoss()

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
            self.log_res(epoch, metrics, save_model=False)

        return self.get_policy()

    def get_policy(self):
        return self.actor

    def _train(self, batch):
        self.total_train_steps += 1

        batch = batch.to_torch(dtype=torch.float32, device=self.args["device"])
        obs = batch["obs"]
        action = batch["act"]
        next_obs = batch["obs_next"]
        reward = batch["rew"].unsqueeze(-1)
        done = batch["done"].unsqueeze(-1)
        # update bca
        action_bca = batch["act"].squeeze(-1)

        action_dist = self.bca(obs)
        loss = self.loss_fn(action_dist.probs, action_bca.long())

        self.bcactor_optim.zero_grad()
        loss.backward()
        self.bcactor_optim.step()


        # update critic
        with torch.no_grad():
            emb = self.bca.backbone(obs)
        _q = self.q(emb).gather(-1, action.long())

        with torch.no_grad():
            emb = self.bca.backbone(next_obs)
            next_q = self.target_q(emb)
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
        # metrics["exploration_rate"] = self.exploration_rate
        return metrics
