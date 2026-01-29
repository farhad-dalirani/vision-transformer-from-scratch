import pytest
import torch
import torch.nn as nn

from vision_transformer.model.transformer_encoder import TransformerEncoder


def test_raises_if_num_encoder_blocks_non_positive():
    with pytest.raises(ValueError):
        _ = TransformerEncoder(num_encoder_blocks=0, block_fn=lambda: nn.Identity())
    with pytest.raises(ValueError):
        _ = TransformerEncoder(num_encoder_blocks=-3, block_fn=lambda: nn.Identity())


def test_output_shape_is_preserved():
    embed_dim = 32
    encoder = TransformerEncoder(
        num_encoder_blocks=3,
        block_fn=lambda: nn.LayerNorm(embed_dim),
        final_norm=None,
    )
    x = torch.randn(2, 10, embed_dim)
    y = encoder(x)
    assert y.shape == x.shape


def test_blocks_are_distinct_no_weight_sharing():
    embed_dim = 16

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    encoder = TransformerEncoder(
        num_encoder_blocks=2, block_fn=DummyBlock, final_norm=None
    )

    # Different module instances
    assert (
        encoder.transformer_encoder_blocks[0]
        is not encoder.transformer_encoder_blocks[1]
    )
    # Different parameter objects (no weight sharing)
    w0 = encoder.transformer_encoder_blocks[0].fc.weight
    w1 = encoder.transformer_encoder_blocks[1].fc.weight
    assert w0 is not w1


def test_final_norm_is_applied_when_provided():
    embed_dim = 8

    class AddOne(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

    encoder = TransformerEncoder(
        num_encoder_blocks=1,
        block_fn=lambda: AddOne(),
        final_norm=nn.Identity(),
    )

    x = torch.zeros(2, 4, embed_dim)
    y = encoder(x)
    # Block adds 1, final_norm is identity
    assert torch.allclose(y, torch.ones_like(y))


def test_final_norm_changes_output_when_not_identity():
    embed_dim = 8

    class AddOne(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

    # With LayerNorm as final norm
    encoder_ln = TransformerEncoder(
        num_encoder_blocks=1,
        block_fn=lambda: AddOne(),
        final_norm=nn.LayerNorm(embed_dim),
    )
    # Without final norm
    encoder_no = TransformerEncoder(
        num_encoder_blocks=1,
        block_fn=lambda: AddOne(),
        final_norm=None,
    )

    x = torch.zeros(2, 4, embed_dim)
    y_ln = encoder_ln(x)
    y_no = encoder_no(x)

    # y_no should be all ones; LayerNorm will transform it (to ~0s for constant inputs)
    assert torch.allclose(y_no, torch.ones_like(y_no))
    assert not torch.allclose(y_ln, y_no)


def test_backward_pass_produces_gradients():
    embed_dim = 16

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(embed_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    encoder = TransformerEncoder(
        num_encoder_blocks=3,
        block_fn=DummyBlock,
        final_norm=nn.LayerNorm(embed_dim),
    )

    x = torch.randn(2, 5, embed_dim, requires_grad=True)
    y = encoder(x).sum()
    y.backward()

    assert x.grad is not None
    # Ensure at least one parameter got gradients
    grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_num_encoder_blocks_matches_length():
    encoder = TransformerEncoder(num_encoder_blocks=4, block_fn=lambda: nn.Identity())
    assert len(encoder.transformer_encoder_blocks) == 4
