from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    image_size: int = 224
    in_channels: int = 3
    patch_size: int = 16
    embed_dim: int = 768
    num_attention_heads: int = 12
    num_encoder_blocks: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    encoder_final_norm: bool = True
    positional_embedding_dropout_p: float = 0.1
    mha_attention_dropout_p: float = 0.1
    mha_proj_dropout_p: float = 0.1
    mlp_dropout_p: float = 0.1


VIT_B_16 = ViTConfig(
    patch_size=16,
    embed_dim=768,
    num_encoder_blocks=12,
    num_attention_heads=12,
)

VIT_B_32 = ViTConfig(
    patch_size=32,
    embed_dim=768,
    num_encoder_blocks=12,
    num_attention_heads=12,
)
