import torch
import torch.nn as nn

from vision_transformer.model.embeddings import CLSTokenAndPositionEmbedding
from vision_transformer.model.mlp_block import (
    FinetuningClassificationHead,
    PreTrainingClassificationHead,
)
from vision_transformer.model.patch_embedding import PatchEmbedding
from vision_transformer.model.transformer_encoder import TransformerEncoder
from vision_transformer.model.transformer_encoder_block import TransformerEncoderBlock


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) model for image classification.

    This module implements the Vision Transformer architecture as described in
    "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale".
    The model splits an image into fixed-size patches, projects them into a
    sequence of embeddings, prepends a learnable [CLS] token, adds learned
    positional embeddings, and processes the sequence with a stack of
    Transformer encoder blocks. The final classification is produced from
    the [CLS] token using either a pre-training or fine-tuning head.

    Architecture:
        Image -> PatchEmbedding
              -> [CLS] token + Positional Embedding
              -> TransformerEncoder (L blocks)
              -> (optional) Final LayerNorm
              -> Classification Head ([CLS] token)

    Args:
        image_size: Height and width of the input images (must be square).
        in_channels: Number of input image channels (e.g., 3 for RGB).
        patch_size: Size of each square image patch.
        embed_dim: Dimension of the token embeddings (D).
        strict: If True, enforces fixed input image size in PatchEmbedding.
        positional_embedding_dropout_p: Dropout probability applied after
            adding positional embeddings.
        num_attention_heads: Number of attention heads in each encoder block.
        mha_attention_dropout_p: Dropout probability applied to attention
            weights in multi-head self-attention.
        mha_proj_dropout_p: Dropout probability applied after the attention
            projection.
        mlp_ratio: Expansion ratio for the hidden dimension of the MLP
            inside each Transformer encoder block.
        mlp_dropout_p: Dropout probability applied in the Transformer MLP.
        num_encoder_blocks: Number of Transformer encoder blocks (L).
        encoder_final_norm: If True, applies a final LayerNorm after the
            encoder stack.
        num_classes: Number of output classes.
        head_type: Type of classification head to use. Must be either
            "pretrain" (MLP head with tanh) or "finetune" (linear head).

    Shape:
        Input:  (B, C, H, W)
        Output: (B, num_classes)

    Raises:
        ValueError: If `image_size` is not divisible by `patch_size`.
        ValueError: If `head_type` is not one of {"pretrain", "finetune"}.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        patch_size: int,
        embed_dim: int,
        strict: bool,
        positional_embedding_dropout_p: float,
        num_attention_heads: int,
        mha_attention_dropout_p: float,
        mha_proj_dropout_p: float,
        mlp_ratio: float,
        mlp_dropout_p: float,
        num_encoder_blocks: int,
        encoder_final_norm: bool,
        num_classes: int,
        head_type: str,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size ({image_size}) must be divisible by "
                f"patch_size ({patch_size}). Vision Transformer requires a "
                "regular grid of non-overlapping patches."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.head_type = head_type.lower()

        grid = image_size // patch_size
        self.num_patches = grid * grid

        # 1) patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            in_channels=in_channels,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            strict=strict,
        )

        # 2) CLS token + positional embedding (with interpolation)
        self.embeddings = CLSTokenAndPositionEmbedding(
            num_patches=self.num_patches,
            embed_dim=self.embed_dim,
            dropout_p=positional_embedding_dropout_p,
        )

        # 3) --- Transformer encoder stack
        final_norm = (
            nn.LayerNorm(normalized_shape=self.embed_dim)
            if encoder_final_norm
            else None
        )

        def encoder_block_fn():
            return TransformerEncoderBlock(
                embed_dim=self.embed_dim,
                num_heads=num_attention_heads,
                mha_attention_dropout_p=mha_attention_dropout_p,
                mha_proj_dropout_p=mha_proj_dropout_p,
                mlp_ratio=mlp_ratio,
                mlp_dropout_p=mlp_dropout_p,
            )

        self.encoder = TransformerEncoder(
            num_encoder_blocks=num_encoder_blocks,
            block_fn=encoder_block_fn,
            final_norm=final_norm,
        )

        # 4) Head
        if self.head_type == "pretrain":
            self.head = PreTrainingClassificationHead(
                input_dim=self.embed_dim, num_classes=self.num_classes
            )
        elif self.head_type == "finetune":
            self.head = FinetuningClassificationHead(
                input_dim=self.embed_dim, num_classes=self.num_classes
            )
        else:
            raise ValueError('head_type must be "pretrain" or "finetune"')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Shape (B, C, H, W)
        x = self.patch_embed(x)  # (B, N, D)
        x = self.embeddings(x)  # (B, N+1, D)
        x = self.encoder(x)  # (B, N+1, D)

        cls_token = x[:, 0]  # (B, D)
        logits = self.head(cls_token)  # (B, num_classes)

        return logits
