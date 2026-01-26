import torch
import torch.nn as nn
import torch.nn.functional as F


class CLSTokenAndPositionEmbedding(nn.Module):
    """Adds a learned [CLS] token and learned positional embeddings to patch tokens.

    This module prepends a learnable classification token to the patch token
    sequence, adds learned positional embeddings, and applies dropout. When
    fine-tuning/inference at a different resolution (i.e., a different number
    of patches), it resizes the patch positional embeddings (excluding the [CLS]
    position) using 2D interpolation and concatenates the [CLS] positional
    embedding back.

    Note:
        This implementation assumes a square patch grid. Both the original and
        target number of patches must be perfect squares.

    Args:
        num_patches: Number of patch tokens used during training (N).
        embed_dim: Token embedding dimension (D).
        dropout_p: Dropout probability applied after adding positional embeddings.

    Shape:
        Input:  (B, N, D)
        Output: (B, N + 1, D)

    Raises:
        ValueError: If `num_patches` or `new_num_patches` is not a perfect square,
            or if the input embedding dimension does not match `embed_dim`.
    """

    def __init__(self, num_patches: int, embed_dim: int, dropout_p: float):

        super().__init__()

        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.num_patches = num_patches

        # Learnable [CLS] token that is prepended to the patch token sequence.
        # Shape: (1, 1, D)
        self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, self.embed_dim)))

        # Learnable positional embeddings for the [CLS] token and all patch tokens.
        # Shape: (1, N+1, D)
        self.pos_embed = nn.Parameter(
            torch.zeros(size=(1, self.num_patches + 1, self.embed_dim))
        )
        # Dropout applied after adding positional embeddings
        # (as described in the ViT paper).
        self.positional_dropout = nn.Dropout(self.dropout_p)

    @staticmethod
    def _grid_size(num_patches: int) -> tuple[int, int]:
        g = int(num_patches**0.5)
        if g * g != num_patches:
            raise ValueError(f"num_patches must be a perfect square, got {num_patches}")
        return g, g

    def interpolate_positional_embeddings(self, new_num_patches: int) -> torch.Tensor:
        """Interpolates learned positional embeddings for a new patch grid.

        ViT learns positional embeddings for a fixed number of patches. When
        fine-tuning/inference at a different resolution, the patch grid changes,
        so the patch positional embeddings (excluding the [CLS] token) are resized
        using 2D interpolation and concatenated back with the [CLS] embedding.

        Args:
            new_num_patches: Target number of patch tokens (Gh_new * Gw_new).

        Returns:
            Tensor of shape (1, new_num_patches + 1, D).
        """

        if new_num_patches == self.num_patches:
            return self.pos_embed

        D = self.pos_embed.shape[-1]
        h_old, w_old = self._grid_size(num_patches=self.num_patches)
        h_new, w_new = self._grid_size(num_patches=new_num_patches)

        # Discard positional encoding for cls token for interpolation
        cls_pos_embed = self.pos_embed[:, :1, :]  # (1, 1, D)
        patch_pos = self.pos_embed[:, 1:, :]  # (1, N, D)

        # Reshape patch embedding in form of grid for interpolatoin
        patch_pos = patch_pos.reshape(
            shape=(1, h_old, w_old, D)
        )  # (1, h_old, w_old, D)
        patch_pos = patch_pos.permute(0, 3, 1, 2).contiguous()  # (1, D, h_old, w_old)

        # Interpolate in 2D, since input is image.
        # (1, D, h_new, w_new)
        patch_pos = F.interpolate(
            input=patch_pos,
            size=(h_new, w_new),
            mode="bicubic",
            align_corners=False,
        )

        # (1, D, h_new, w_new) -> (1, h_new, w_new, D)
        patch_pos = patch_pos.permute(0, 2, 3, 1).contiguous()
        # (1, h_new, w_new, D) -> (1, new_num_patches, D)
        patch_pos = patch_pos.reshape(1, new_num_patches, D)

        # Add back cls token positional embedding
        # (1, new_num_patches+1, D)
        pos_embed_new = torch.cat([cls_pos_embed, patch_pos], dim=1)

        return pos_embed_new

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # X: Batch size (B), Number of tokens (N), token dimension (D)
        B, N, D = x.shape

        if D != self.embed_dim:
            raise ValueError(f"Expected embed_dim {self.embed_dim}, got {D}")

        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # Add the classification-token to stream of tokens
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)

        # Add positional embeddings to the token sequence.
        # Broadcasting applies the same positional encoding to all samples in the batch.
        # Output shape: (B, N+1, D)
        positional_embds = self.interpolate_positional_embeddings(new_num_patches=N)
        x = x + positional_embds

        # Apply dropout after positional embedding addition
        x = self.positional_dropout(x)
        return x
