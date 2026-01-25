import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (MSA) module.

    Implements scaled dot-product self-attention with multiple heads
    as described in
    -  Attention Is All You Need paper
    , and used in Vision
    Transformers (ViT) paper.
        - AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE
        RECOGNITION AT SCALE

    Args:
        dim: Embedding dimension of input tokens.
        num_heads: Number of attention heads.
        attention_dropout_p: Dropout probability applied to attention
            weights after softmax.
        proj_dropout_p: Dropout probability applied after the output
            projection.

    Shape:
        Input:  (B, N, dim), where B is batch size and N is sequence length.
        Output: (B, N, dim).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout_p: float,
        proj_dropout_p: float,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                "Token dimension must be divisible by num_heads,"
                f" got dim {dim}, and num_heads {num_heads}"
            )

        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = int(self.dim // self.num_heads)
        self.attention_dropout_p = attention_dropout_p
        self.proj_dropout_p = proj_dropout_p
        self.scale = self.head_dim**-0.5

        # Linear projection to create query(Q), key (K), value (V) vectors for each
        # self-attention head from input.
        # We use this linear layer in a way that is equivalent to having three separate
        # linear layers for each head (fig.2 in Attention Is All You Need paper).
        self.qkv = nn.Linear(in_features=self.dim, out_features=self.dim * 3, bias=True)

        # Dropout layer after calculating attention score
        self.attention_dropout = nn.Dropout(p=self.attention_dropout_p)

        # Linear layer that applied on output of multi head attnetion (MHA)
        self.proj = nn.Linear(in_features=self.dim, out_features=self.dim, bias=True)
        self.proj_dropout = nn.Dropout(p=self.proj_dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # X: (B, N, D)
        B, N, D = x.shape
        if D != self.dim:
            raise ValueError(
                "Tokens embedding dimention multi head"
                f" attention to should be {self.dim}, but got {D}."
            )
        H = self.num_heads
        D_head = self.head_dim

        # --- Create Q, K, V for each attention head
        # qkv: (B, N, 3 * D)
        qkv: torch.Tensor = self.qkv(x)
        # qkv: (B, N, 3 * D) -> (B, N, 3, H, D_head)
        qkv = qkv.reshape(B, N, 3, H, D_head)
        # Change order of axis to be able to seprate Q, K, V and
        # calculate attention score.
        # qkv: (3, B, H, N, D_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        # Separate Q, K, V
        # Q, K, V, each: (B, H, N, D_head)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # Calulating attention score
        attention = (Q @ K.transpose(-2, -1)) * self.scale  # B, H, N, N
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        # Calculating new values of queries based on attention score and values.
        # Golbal refinement of each token.
        out = attention @ V  # B, H, N, D_head
        # Attaching output of all self-attention heads
        out = out.transpose(1, 2).reshape(B, N, D)  # B, N, D

        # Apply linear projection on each token
        # Local refinement of each token.
        out = self.proj(out)
        out = self.proj_dropout(out)

        return out
