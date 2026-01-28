import torch
import torch.nn as nn

from vision_transformer.model.attention import MultiHeadSelfAttention
from vision_transformer.model.mlp_block import TransformerMLP


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block used in Vision Transformer (ViT).

    Implements the Transformer block described in the Vision Transformer
    paper. 
    The block consists of a multi-head self-attention
    (MSA) layer and a position-wise feed-forward MLP (FFN), each preceded
    by LayerNorm and followed by a residual connection.

    The computation follows:
        x = x + MSA(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Args:
        embed_dim: Token embedding dimension (D).
        num_heads: Number of attention heads.
        mha_attention_dropout_p: Dropout probability applied to attention
            weights in the multi-head self-attention layer.
        mha_proj_dropout_p: Dropout probability applied after the attention
            output projection.
        mlp_ratio: Expansion ratio for the hidden dimension of the MLP
            (e.g., 4.0 means hidden_dim = 4 * embed_dim).
        mlp_dropout_p: Dropout probability applied after each dense layer
            in the Transformer MLP.

    Shape:
        Input:  (B, N, D)
        Output: (B, N, D)
    """

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
