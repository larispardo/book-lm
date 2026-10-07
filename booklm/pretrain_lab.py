"""Analysis helpers behind the pretraining lab panel. No Streamlit in here."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from booklm.metrics import bits_per_byte
from booklm.model import GPT, GPTConfig
from booklm.text_codec import DEFAULT

MODELS = Path("artifacts/models")
TOKENS = Path("artifacts/data/tokens")


@dataclass
class Run:
    name: str
    header: dict
    train: pd.DataFrame  # step, train_loss, lr, elapsed
    evals: pd.DataFrame  # step, val_loss, sample, best

    @property
    def best(self) -> pd.Series | None:
        return None if self.evals.empty else self.evals.loc[self.evals["val_loss"].idxmin()]

    def epochs(self, steps: pd.Series) -> pd.Series:
        return steps * self.header["tokens_per_step"] / self.header["train_tokens"]

    @property
    def tokenizer(self) -> str:
        return self.header.get("tokenizer", DEFAULT)

    def best_bits_per_byte(self, tokens_dir: Path = TOKENS) -> float | None:
        """Best val loss in bits per byte: comparable across tokenizers.

        Raises NotImplementedError until the metrics.bits_per_byte exercise is done.
        """
        if self.best is None:
            return None
        val = json.loads((tokens_dir / self.tokenizer / "meta.json").read_text())["val"]
        n_bytes = round(val["tokens"] * val["bytes_per_token"])
        return bits_per_byte(float(self.best.val_loss), val["tokens"], n_bytes)


def list_runs(models_dir: Path = MODELS) -> list[str]:
    return sorted(p.parent.name for p in models_dir.glob("*/log.jsonl"))


def load_run(run_dir: Path) -> Run:
    header, train, evals = None, [], []
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "header" in rec:
            header = rec["header"]
        elif "val_loss" in rec:
            evals.append({**rec, "best": rec.get("best", False)})
        elif "train_loss" in rec:
            train.append(rec)
    return Run(
        name=run_dir.name,
        header=header or _header_from_checkpoint(run_dir),
        train=pd.DataFrame(train, columns=["step", "train_loss", "lr", "elapsed"]),
        evals=pd.DataFrame(evals, columns=["step", "val_loss", "sample", "best"]),
    )


def _header_from_checkpoint(run_dir: Path, batch_size: int = 32) -> dict:
    """Runs logged before headers existed: rebuild what we can (default batch size)."""
    ckpt = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    cfg, tokenizer = ckpt["config"], ckpt.get("tokenizer", DEFAULT)
    train_tokens = (TOKENS / tokenizer / "train.bin").stat().st_size // 2  # uint16: 2 bytes
    return {
        "run": run_dir.name,
        "tokenizer": tokenizer,
        "preset": run_dir.name.removeprefix("holmes-gpt-"),
        "config": cfg,
        "params": GPT(GPTConfig(**cfg)).num_params(),
        "batch_size": batch_size,
        "tokens_per_step": batch_size * cfg["context"],
        "train_tokens": train_tokens,
    }


def load_model(run_dir: Path, device: str = "cpu") -> tuple[GPT, int]:
    """The best checkpoint of a run, ready for inference. Returns (model, step)."""
    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["step"]


@torch.no_grad()
def attention_maps(model: GPT, ids: list[int]) -> torch.Tensor:
    """Attention weights for one sequence: (layers, heads, T, T)."""
    model(torch.tensor([ids]), capture=True)
    return torch.stack([block.attn.last_weights[0] for block in model.blocks])


def perplexity(loss: float) -> float:
    return math.exp(loss)
