import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Image to patch embedding layer for Vision Transformers.

    Splits an input image into non-overlapping patches of size
    `patch_size × patch_size` and projects each patch into an
    `embed_dim`-dimensional embedding using a convolutional projection.
    This is equivalent to flattening each patch and applying a linear
    layer, as described in the ViT paper.

    Args:
        image_size: Expected height and width of the input images.
        in_channels: Number of input image channels.
        patch_size: Size of each square patch.
        embed_dim: Dimension of the patch embeddings.
        strict: If True, validates that input images match `image_size`
            and `in_channels` during the forward pass. If False, only
            enforces divisibility by `patch_size`, allowing variable
            input resolutions (e.g., during fine-tuning).

    Attributes:
        num_patches: Number of patches produced for an image of size
            `image_size × image_size`.

    Shape:
        Input:  (B, C, H, W)
        Output: (B, N, embed_dim), where
            N = (H / patch_size) × (W / patch_size).
    """
        
    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        strict: bool = True,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = int((self.image_size // self.patch_size) ** 2)

        # Equivalent to linear project each patch
        self.proj = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        # Whether to validate input shape/channels in forward().
        # Model can be fine-tuned on higher resolutions, therefore,
        # enfocing fix size images should be optional.
        self.strict = strict

    def set_strict(self, strict: bool) -> None:
        """Enable or disable strict input shape/channel validation."""
        self.strict = strict
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, H, W = x.shape

        if self.strict:
            if H != self.image_size or W != self.image_size:
                raise ValueError(
                    f"Expected input size {(self.image_size, self.image_size)}, got {(H, W)}"
                )
            if C != self.in_channels:
                raise ValueError(
                    f"Expected image with {self.in_channels} channels, but got {C}."
                )
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError("Input height/width must be divisible by patch_size")

        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.reshape(B, self.embed_dim, -1)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)

        return x
