"""Decoding as an explicit pipeline, so every stage can be shown to a human.

    raw logits
      -> repetition penalty     (push down tokens already in the context)
      -> temperature            (sharpen T<1 / flatten T>1; T=0 means greedy)
      -> truncation filters     (top-k, top-p, min-p decide which tokens may be sampled)
      -> renormalise and sample

Order matches Hugging Face's logits processors: penalty, temperature, then top-k, top-p, min-p.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    min_p: float = 0.0  # 0.0 disables
    repetition_penalty: float = 1.0  # 1.0 disables
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0


@dataclass
class Distribution:
    """Every intermediate view of one next-token distribution (1-D tensors over the vocab)."""

    raw_logits: torch.Tensor
    penalized_logits: torch.Tensor
    raw_probs: torch.Tensor  # softmax(raw_logits): what the model "believes"
    scaled_probs: torch.Tensor  # after penalty and temperature, before truncation
    keep: torch.Tensor  # bool mask of tokens allowed by the filters
    final_probs: torch.Tensor  # what is actually sampled from


def apply_repetition_penalty(
    logits: torch.Tensor, history: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style penalty: divide positive logits, multiply negative ones, for seen tokens."""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    out = logits.clone()
    seen = torch.unique(history)
    vals = out[seen]
    out[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def top_k_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.numel():
        return torch.ones_like(logits, dtype=torch.bool)
    threshold = torch.topk(logits, k).values[-1]
    return logits >= threshold


def top_p_mask(probs: torch.Tensor, p: float) -> torch.Tensor:
    """Smallest set of most-likely tokens whose cumulative probability reaches p."""
    if p >= 1.0:
        return torch.ones_like(probs, dtype=torch.bool)
    sorted_probs, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=0)
    # Keep a token if the mass *before* it is still below p; the top token always survives.
    keep_sorted = (cumulative - sorted_probs) < p
    keep_sorted[0] = True
    keep = torch.zeros_like(keep_sorted)
    keep[order] = keep_sorted
    return keep


def min_p_mask(probs: torch.Tensor, min_p: float) -> torch.Tensor:
    """Keep tokens at least min_p times as likely as the most likely token."""
    if min_p <= 0.0:
        return torch.ones_like(probs, dtype=torch.bool)
    return probs >= min_p * probs.max()


def process(logits: torch.Tensor, params: SamplingParams, history: torch.Tensor) -> Distribution:
    logits = logits.float()
    penalized = apply_repetition_penalty(logits, history, params.repetition_penalty)
    raw_probs = torch.softmax(logits, dim=-1)

    if params.greedy:
        scaled = torch.softmax(penalized, dim=-1)
        keep = torch.zeros_like(scaled, dtype=torch.bool)
        keep[torch.argmax(penalized)] = True
    else:
        scaled = torch.softmax(penalized / params.temperature, dim=-1)
        keep = top_k_mask(penalized, params.top_k)
        keep &= top_p_mask(scaled, params.top_p)
        keep &= min_p_mask(scaled, params.min_p)

    final = torch.where(keep, scaled, torch.zeros_like(scaled))
    final = final / final.sum()
    return Distribution(logits, penalized, raw_probs, scaled, keep, final)


def sample(dist: Distribution, generator: torch.Generator | None = None) -> int:
    return int(torch.multinomial(dist.final_probs, 1, generator=generator).item())


def entropy_bits(probs: torch.Tensor) -> float:
    nz = probs[probs > 0]
    return float(-(nz * torch.log2(nz)).sum())
