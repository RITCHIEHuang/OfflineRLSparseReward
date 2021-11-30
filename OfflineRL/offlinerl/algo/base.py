import os
import uuid
import json
from abc import ABC, abstractmethod

import wandb
import torch
from collections import OrderedDict
from loguru import logger
from offlinerl.utils.exp import init_custom_exp_logger, init_exp_logger
from offlinerl.utils.io import create_dir


class BaseAlgo(ABC):
    def __init__(self, args):
        logger.info("Init AlgoTrainer")
        if "exp_name" not in args.keys():
            exp_name = str(uuid.uuid1()).replace("-", "")
        else:
            exp_name = args["exp_name"]

        if "log_path" in args.keys():
            repo = args["log_path"]
        else:
            repo = None

        self.log_to_wandb = args["log_to_wandb"]

        # setup tensorboard
        self.repo, self.exp_logger = init_custom_exp_logger(repo, exp_name)

        # setup wandb
        if self.log_to_wandb:
            wandb.init(
                name=args["exp_name"],
                group=args["task"],
                project="OfflineRL_DelayRewards",
                config=args,
            )

        # setup aim
        try:
            self.aim_exp_logger = init_exp_logger(
                repo=self.repo, experiment_name=exp_name
            )
            self.aim_exp_logger.set_params(args, name="hparams")
        except:
            logger.error(f"Error initializing Aim Logger !!!")
            self.aim_exp_logger = None

        # self.index_path = self.aim_exp_logger.repo.index_path
        self.index_path = f"{self.repo}/{exp_name}"
        self.models_save_dir = os.path.join(self.index_path, "models")
        self.metric_logs = OrderedDict()
        self.metric_logs_path = os.path.join(
            self.index_path, "metric_logs.json"
        )
        create_dir(self.models_save_dir)

        self.hparams_path = os.path.join(self.index_path, "hparams.json")
        with open(self.hparams_path, "w") as f:
            json.dump(args, f)

    def log_res(self, epoch, result):
        logger.info("Epoch : {}", epoch)

        if self.log_to_wandb:
            wandb.log(result)

        for k, v in result.items():
            logger.info("{} : {}", k, v)
            if self.aim_exp_logger:
                self.aim_exp_logger.track(
                    v,
                    name=k.split(" ")[0],
                    epoch=epoch,
                )

            self.exp_logger.add_scalar(k.split(" ")[0], v, epoch)
            self.exp_logger.flush()

        self.metric_logs[str(epoch)] = result
        with open(self.metric_logs_path, "w") as f:
            json.dump(self.metric_logs, f)
        self.save_model(os.path.join(self.models_save_dir, str(epoch) + ".pt"))

    @abstractmethod
    def train(
        self,
        history_buffer,
        eval_fn=None,
    ):
        pass

    def _sync_weight(self, net_target, net, soft_target_tau=5e-3):
        for o, n in zip(net_target.parameters(), net.parameters()):
            o.data.copy_(
                o.data * (1.0 - soft_target_tau) + n.data * soft_target_tau
            )

    @abstractmethod
    def get_policy(
        self,
    ):
        pass

    def save_model(self, model_path):
        torch.save(self.get_policy(), model_path)

    def load_model(self, model_path):
        model = torch.load(model_path)

        return model
