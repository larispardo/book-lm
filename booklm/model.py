"""A small GPT, written out in full so every matrix is visible.

    token ids (B, T)
      → token embedding + position embedding        (B, T, d)
      → N × Block:  x + Attention(LayerNorm(x))      tokens look at earlier tokens
                    x + MLP(LayerNorm(x))            each token thinks on its own
      → LayerNorm → linear head (tied to embedding) (B, T, vocab)  = logits

B = batch, T = sequence length, d = model width, H = heads, Dh = d / H.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    context: int = 256
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "tiny": GPTConfig(d_model=128, n_layers=2, n_heads=4),  # ~1.5M params, smoke test
    "small": GPTConfig(d_model=256, n_layers=4, n_heads=4),  # ~5M
    "base": GPTConfig(d_model=384, n_layers=6, n_heads=6),  # ~14M
}


# ── YOUR EXERCISE ───────────────────────────────────────────────────────────────────────


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention where each position only sees itself and the past.

    Inputs q, k, v: (B, H, T, Dh). Returns (output (B, H, T, Dh), weights (B, H, T, T)).

    1. scores  = q @ kᵀ / √Dh          how well each query matches each key   (B, H, T, T)
    2. mask    scores[..., i, j] = -inf wherever j > i (no peeking at future tokens)
    3. weights = softmax(scores) over the last dim  → each row sums to 1
    4. output  = weights @ v           a weighted average of the values

    Useful: k.transpose(-2, -1), torch.triu / torch.ones(T, T, dtype=torch.bool),
    tensor.masked_fill(mask, float("-inf")), torch.softmax(x, dim=-1).
    The weights are returned too, so the lab can draw attention maps.
    """
    raise NotImplementedError


# ── building blocks (provided) ─────────────────────────────────────────────────────────


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, "d_model must divide evenly into heads"
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)  # q, k, v in one matmul
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.last_weights: torch.Tensor | None = None  # filled when capture=True

    def forward(self, x: torch.Tensor, capture: bool = False) -> torch.Tensor:
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (B, T, d) -> (B, H, T, Dh): each head works on its own slice of the width
        q, k, v = (t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2) for t in (q, k, v))
        out, weights = causal_attention(q, k, v)
        if capture:
            self.last_weights = weights.detach()
        out = out.transpose(1, 2).contiguous().view(B, T, d)  # heads back side by side
        return self.dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.up = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.down = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.gelu(self.up(x))))


class Block(nn.Module):
    """Pre-norm residual block: normalise, transform, add back to the stream."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, capture: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), capture=capture)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # one row per token id
        self.pos_emb = nn.Embedding(cfg.context, cfg.d_model)  # one row per position
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying: same matrix in and out
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                # Residual branches add up across layers; shrink them so the stream stays stable.
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx: torch.Tensor, capture: bool = False) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.cfg.context, f"sequence of {T} exceeds context {self.cfg.context}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, capture=capture)
        return self.head(self.ln_f(x))  # (B, T, vocab) logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Append tokens one at a time (no KV cache yet: that is stage 3)."""
        for _ in range(max_new_tokens):
            logits = self(idx[:, -self.cfg.context :])[:, -1, :]
            if temperature == 0:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k:
                    kth = torch.topk(logits, top_k).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs.cpu(), 1, generator=generator).to(idx.device)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
