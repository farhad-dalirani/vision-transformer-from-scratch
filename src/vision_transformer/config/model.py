from dataclasses import dataclass


@dataclass(frozen=True)
class ViTConfig:
    """Configuration for a Vision Transformer (ViT) model.

    This dataclass defines the architectural hyperparameters used to
    construct a Vision Transformer, including patch embedding, encoder
    depth, attention settings, and classifier head configuration.

    Attributes:
        image_size (int): Input image size (assumed square).
            Defaults to 224.
        in_channels (int): Number of input channels (e.g., 3 for RGB).
            Defaults to 3.
        patch_size (int): Size of each image patch (assumed square).
            The image is divided into non-overlapping patches of size
            `patch_size x patch_size`. Defaults to 16.
        embed_dim (int): Dimension of the token embeddings and
            transformer hidden states. Defaults to 768.
        num_attention_heads (int): Number of attention heads in each
            multi-head self-attention layer. Defaults to 12.
        num_encoder_blocks (int): Number of transformer encoder blocks.
            Defaults to 12.
        mlp_ratio (float): Expansion ratio for the hidden dimension
            inside the MLP block relative to `embed_dim`.
            Defaults to 4.0.
        num_classes (int): Number of output classes for classification.
            Defaults to 1000.
        encoder_final_norm (bool): Whether to apply a final layer
            normalization after the encoder stack. Defaults to True.
        positional_embedding_dropout_p (float): Dropout probability
            applied to the positional embeddings. Defaults to 0.1.
        mha_attention_dropout_p (float): Dropout probability applied
            to attention weights in multi-head attention.
            Defaults to 0.1.
        mha_proj_dropout_p (float): Dropout probability applied to
            the output projection of multi-head attention.
            Defaults to 0.1.
        mlp_dropout_p (float): Dropout probability applied within
            the MLP block. Defaults to 0.1.
    """

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
