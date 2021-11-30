# pylint: disable=protected-access
import math
import torch
from torch import nn
from copy import deepcopy
from loguru import logger

import numpy as np
from tqdm import tqdm

from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    MultiheadAttention,
)

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.exp import setup_seed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerRewardDecomposer(nn.Module):
    def __init__(
        self,
        input_pair_dim,
        d_model,
        nhead=4,
        dim_ff=128,
        nlayers=4,
        dropout=0.1,
    ):
        super(TransformerRewardDecomposer, self).__init__()
        self.model_type = "Transformer"
        self.d_model = d_model
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.embedding_net = nn.Linear(input_pair_dim, d_model)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.self_attn = MultiheadAttention(d_model, 1, 0.1, batch_first=True)
        self.pos_encoder = PositionalEncoding(d_model)
        self.reward_ff = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.tril(torch.ones(sz, sz)) == 1
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src, key_padding_mask=None):
        # src is batch first
        sz = src.shape[1]
        src = self.embedding_net(src)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=key_padding_mask
        )
        pair_mask = torch.where(key_padding_mask == 0, 1, 0)[..., None]
        pair_importance = torch.softmax(
            self.ff(output) + (key_padding_mask * -1e9)[..., None], dim=1
        )
        # print(f"pair_importance shape:{pair_importance.shape}")
        # pair_importance = self.ff(output)
        output = pair_importance * output
        output = self.reward_ff(output) * pair_mask
        return torch.tanh(output)


def create_key_padding_mask(seq_of_pair_length, max_length):
    out = torch.ones(len(seq_of_pair_length), max_length)
    for i, e in enumerate(seq_of_pair_length):
        out[i, :e] = 0
    return out


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

    model = TransformerRewardDecomposer(
        input_pair_dim=obs_shape + action_shape,
        d_model=args["d_model"],
        nhead=args["nhead"],
        dim_ff=args["hidden_features"],
        nlayers=args["hidden_layers"],
        dropout=args["dropout"],
    ).to(args["device"])
    optim = torch.optim.Adam(model.parameters(), lr=args["lr"])

    return {
        "model": {"net": model, "opt": optim},
    }


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.model = self.args["model"]["net"]
        self.model_optim = self.args["model"]["optim"]

        self.batch_size = self.args["batch_size"]
        self.device = self.args["device"]

        self.best_model = deepcopy(self.model)
        self.best_loss = float("inf")

    def train(
        self,
        train_loader,
        val_loader,
        callback_fn,
    ):
        for epoch in range(self.args["max_epoch"]):
            ep_loss = 0
            for batch, s in enumerate(train_loader):
                key_padding_mask = create_key_padding_mask(
                    s["length"], dataset.max_length
                ).to(self.device)
                reward_pre = self.model(
                    s["obs_act_pair"].to(self.device),
                    key_padding_mask=key_padding_mask,
                ).squeeze(dim=-1)
                reward_mask = torch.where(
                    key_padding_mask.view(
                        len(s["length"]), dataset.max_length, 1
                    )
                    == 0,
                    1,
                    0,
                )
                delay_reward = s["delay_reward"].to(self.device)
                returns = delay_reward.sum(dim=-1)
                main_loss = (
                    torch.mean(
                        reward_pre[range(len(s["length"])), s["length"] - 1]
                        - returns[:, None]
                    )
                    ** 2
                )
                aux_loss = torch.mean(reward_pre - returns[..., None]) ** 2
                loss = main_loss + aux_loss * 0.5
                ep_loss += loss.item()

                self.model_optim.zero_grad()
                loss.backward()
                self.model_optim.step()

            if ep_loss < self.best_loss:
                self.best_loss = ep_loss
                self.best_model.load_state_dict(self.model.state_dict())

            # res = callback_fn(self.get_policy())
            res = {}
            res["loss"] = ep_loss
            self.log_res(epoch, res)

    def get_policy(self):
        return self.best_model


if __name__ == "__main__":

    obs_act_pair = torch.rand((3, 5, 10))  # [batch, seq_len, obs+act]

    # [batch, len]
    key_padding_mask = create_key_padding_mask([3, 4, 5], 5)

    model = TransformerRewardDecomposer(10, 512)
    reward_pre = model(obs_act_pair, key_padding_mask=key_padding_mask)
    print(reward_pre.shape)
