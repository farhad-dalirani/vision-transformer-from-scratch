import torch
import torch.nn as nn

from vision_transformer.model.attention import MultiHeadSelfAttention
from vision_transformer.model.mlp_block import TransformerMLP


class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mha_attention_dropout_p: float,
        mha_proj_dropout_p: float,
        mlp_ratio: float,
        mlp_dropout_p: float,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mha = MultiHeadSelfAttention(
            dim=embed_dim,
            num_heads=num_heads,
            attention_dropout_p=mha_attention_dropout_p,
            proj_dropout_p=mha_proj_dropout_p,
        )

        self.norm2 = nn.LayerNorm(normalized_shape=embed_dim)
        self.mlp = TransformerMLP(
            embed_dim=embed_dim, mlp_ratio=mlp_ratio, dropout_p=mlp_dropout_p
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (B, N, D)
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x
