from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    num_transformer_blocks: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    dropout: float = 0.1


VIT_B_16 = ViTConfig(
    patch_size=16,
    embed_dim=768,
    num_transformer_blocks=12,
    num_heads=12,
)

VIT_B_32 = ViTConfig(
    patch_size=32,
    embed_dim=768,
    num_transformer_blocks=12,
    num_heads=12,
)
