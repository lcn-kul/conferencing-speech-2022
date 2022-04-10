import math
from torch import Tensor, nn
import torch

from src.model.config import Config


class TransformerWrapper(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        # Normalization.
        self.norm = nn.BatchNorm1d(config.dim_extractor)

        # Position encoding.
        self.position_encoding = PositionalEncoding(config)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_extractor,
            dim_feedforward=config.dim_extractor,
            nhead=1,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:

        # Normalization.
        # Transform from (N, L, C) to (N, C, L) and back.
        x = self.norm(x.permute((0, 2, 1))).permute((0, 2, 1))

        # Position encoding + transformer.
        x = self.position_encoding(x)
        x = self.transformer_encoder(x, mask)

        x = self.dropout(x)

        return x


class PositionalEncoding(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        d_model: int = config.dim_extractor
        seq_len: int = config.feat_seq_len

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(2*seq_len) / d_model))
        pe = torch.zeros(1, seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe.expand(x.shape)
        return x

