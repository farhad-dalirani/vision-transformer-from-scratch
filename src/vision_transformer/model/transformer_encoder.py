from typing import Callable

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer encoder composed of a stack of encoder blocks.

    This module applies a sequence of Transformer encoder blocks to an input
    token sequence. Each block is created independently using `block_fn`,
    ensuring no weight sharing between blocks. An optional final layer
    normalization can be applied after the last encoder block, as used in
    Vision Transformers.

    Args:
        num_encoder_blocks: Number of encoder blocks (L) in the encoder stack.
        block_fn: A callable that returns a new instance of a Transformer
            encoder block when invoked. This is typically a factory function
            or lambda that constructs a `TransformerEncoderBlock`.
        final_norm: Optional layer normalization applied after the final
            encoder block. If None, no final normalization is applied.

    Shape:
        Input:  (B, N, D)
        Output: (B, N, D)

    Raises:
        ValueError: If `num_encoder_blocks` is less than or equal to zero.
    """

    def __init__(
        self,
        num_encoder_blocks: int,
        block_fn: Callable[[], nn.Module],
        final_norm: nn.Module | None = None,
    ):
        super().__init__()

        if num_encoder_blocks <= 0:
            raise ValueError(
                f"num_encoder_blocks must be > 0, got {num_encoder_blocks}"
            )

        self.transformer_encoder_blocks = nn.ModuleList(
            [block_fn() for _ in range(num_encoder_blocks)]
        )

        self.final_norm = final_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the encoder blocks sequentially to the input token sequence.

        Args:
            x: Input tensor of shape (B, N, D), where B is the batch size,
                N is the number of tokens, and D is the embedding dimension.

        Returns:
            Tensor of shape (B, N, D) after processing by all encoder blocks
            and the optional final normalization.
        """
        for block_i in self.transformer_encoder_blocks:
            x = block_i(x)
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x
