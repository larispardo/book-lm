"""Shared pieces for both training routes: batching, evaluation, LR schedule, model cards."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def load_split(data_dir: Path, split: str) -> torch.Tensor:
    return torch.load(data_dir / f"{split}.pt").long()


def get_batch(
    data: torch.Tensor, batch_size: int, ctx: int, device: torch.device, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(len(data) - ctx - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i : i + ctx] for i in starts])
    y = torch.stack([data[i + 1 : i + ctx + 1] for i in starts])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model, data: torch.Tensor, batch_size: int, ctx: int, device, iters: int = 20
) -> float:
    """Mean next-token cross-entropy over a fixed set of validation windows."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, batch_size, ctx, device, gen)
        logits = model(input_ids=x).logits
        losses.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten()).item())
    model.train()
    return sum(losses) / len(losses)


def cosine_lr(step: int, max_lr: float, min_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@dataclass
class ModelCard:
    name: str
    kind: str  # "scratch" | "pretrained" | "finetuned"
    description: str
    base_model: str | None
    total_params: int
    trainable_params: int
    train_tokens_seen: int = 0
    val_loss: float | None = None
    history: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    created_at: float = field(default_factory=time.time)

    @property
    def val_ppl(self) -> float | None:
        return math.exp(self.val_loss) if self.val_loss is not None else None

    def save(self, model_dir: Path) -> None:
        payload = asdict(self) | {"val_ppl": self.val_ppl}
        (model_dir / "card.json").write_text(json.dumps(payload, indent=2))


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
