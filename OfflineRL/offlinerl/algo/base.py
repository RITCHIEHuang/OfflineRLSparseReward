from collections import OrderedDict
import os
import uuid
import json
from abc import ABC, abstractmethod

import wandb
import torch
from loguru import logger

from torch.utils.tensorboard import SummaryWriter


from offlinerl.utils.io import create_dir
from offlinerl.utils.logger import log_path


class BaseAlgo(ABC):
    def __init__(self, args):
        logger.info("Init AlgoTrainer")

        self.log_to_wandb = args["log_to_wandb"]

        if "exp_name" not in args.keys():
            exp_name = str(uuid.uuid1()).replace("-", "")
        else:
            exp_name = args["exp_name"]

        if "log_path" in args.keys():
            log_path_ = args["log_path"]
        else:
            log_path_ = log_path()

        tb_log_path = os.path.join(log_path_, "./logs")
        if not os.path.exists(tb_log_path):
            logger.info(
                "{} dir is not exist, create {}", tb_log_path, tb_log_path
            )
            os.makedirs(tb_log_path)

        # setup tensorboard
        self.exp_logger = SummaryWriter(log_dir=f"{tb_log_path}/{exp_name}")

        # setup wandb
        if self.log_to_wandb:
            wandb.init(
                dir=f"{log_path_}/wandb",
                name=args["exp_name"],
                group=args["task"],
                project=args["project"],
                config=args,
            )

        self.index_path = f"{tb_log_path}/{exp_name}"
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
