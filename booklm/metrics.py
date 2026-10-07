"""Metrics that stay comparable across tokenizers."""

from __future__ import annotations
import math

# ── YOUR EXERCISE ───────────────────────────────────────────────────────────────────────


def bits_per_byte(loss_nats_per_token: float, tokens: int, n_bytes: int) -> float:
    """Convert a per-token loss into bits per byte of text.

    Our loss is cross-entropy in *nats* (natural log) *per token*. Two changes make it
    comparable across tokenizers:
      1. per token -> per byte: the text has `tokens` tokens spread over `n_bytes` bytes,
         so the total loss of the text is loss × tokens; divide that by n_bytes.
      2. nats -> bits: divide by ln(2)  (math.log(2)).

    Sanity checks: a model guessing uniformly among 256 bytes, with one token per byte, has
    loss ln(256) nats/token and should score exactly 8 bits per byte.
    """
    loss = loss_nats_per_token*tokens / n_bytes
    return loss / math.log(2)
