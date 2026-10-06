import numpy as np
import pytest
import torch
import torch.nn.functional as F

from booklm.batches import get_batch
from booklm.model import GPT, GPTConfig, causal_attention
from booklm.pretrain import lr_at, train_step

TINY = GPTConfig(vocab_size=50, context=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0)
EXERCISE = pytest.mark.xfail(raises=NotImplementedError, strict=True, reason="exercise")


# ── provided pieces ─────────────────────────────────────────────────────────────────────


def test_batches_are_shifted_by_one_token():
    data = np.arange(100, dtype=np.uint16)
    x, y = get_batch(data, batch_size=4, context=8, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y[:, -1], x[:, -1] + 1)


def test_param_count_and_weight_tying():
    model = GPT(TINY)
    assert model.head.weight is model.tok_emb.weight
    assert model.num_params() == sum(p.numel() for p in model.parameters())


def test_lr_schedule_warms_up_then_decays_to_ten_percent():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 10, 100) == pytest.approx(0.1)


# ── exercise 1: causal_attention ────────────────────────────────────────────────────────


def test_attention_matches_pytorch():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 8, 16) for _ in range(3))
    out, _ = causal_attention(q, k, v)
    assert torch.allclose(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5)


def test_attention_weights_are_causal_and_normalised():
    q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
    _, w = causal_attention(q, k, v)
    assert w.shape == (1, 2, 5, 5)
    assert torch.allclose(w.sum(-1), torch.ones(1, 2, 5))
    assert torch.all(w.triu(diagonal=1) == 0)  # nothing above the diagonal: no future
    assert torch.all(w[..., 0, 0] == 1)  # the first token can only look at itself


def test_future_tokens_cannot_change_the_past():
    model = GPT(TINY).eval()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 2, 3, 9, 9]])
    assert torch.allclose(model(a)[:, :3], model(b)[:, :3], atol=1e-6)


# ── exercise 2: train_step ──────────────────────────────────────────────────────────────


def test_train_step_overfits_one_batch():
    torch.manual_seed(0)
    model = GPT(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, 50, (4, 16))
    y = torch.randint(0, 50, (4, 16))
    first = train_step(model, optimizer, x, y)
    assert first == pytest.approx(np.log(50), rel=0.1)  # untrained ≈ uniform guess
    for _ in range(100):
        last = train_step(model, optimizer, x, y)
    assert last < 0.5 * first
