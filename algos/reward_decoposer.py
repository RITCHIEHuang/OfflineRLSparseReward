# pylint: disable=protected-access
import torch
from torch import nn
import math
from torch.nn import (
    TransformerEncoder,
    TransformerEncoderLayer,
    MultiheadAttention,
    parameter,
)
from torch.nn.modules.activation import ReLU
from torch.nn.utils.rnn import pad_sequence

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

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
            nn.Linear(d_model,d_model//2),
            nn.Tanh(),
            nn.Linear(d_model//2,1),
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
        pair_mask=torch.where(key_padding_mask==0,1,0)[...,None]
        pair_importance = torch.softmax(self.ff(output)+(key_padding_mask*-1e9)[...,None],dim=1)
        # print(f"pair_importance shape:{pair_importance.shape}")
        output = pair_importance*output
        output = self.reward_ff(output)*pair_mask
        return output


def create_key_padding_mask(seq_of_pair_length, max_length):
    out = torch.ones(len(seq_of_pair_length), max_length)
    for i, e in enumerate(seq_of_pair_length):
        out[i, :e] = 0
    return out


if __name__ == "__main__":

    obs_act_pair = torch.rand((3, 5, 10))  # [batch, seq_len, obs+act]

    # [batch, len]
    key_padding_mask = create_key_padding_mask([3, 4, 5], 5)

    model = TransformerRewardDecomposer(10, 512)
    reward_pre = model(obs_act_pair, key_padding_mask=key_padding_mask)
    print(reward_pre.shape)
