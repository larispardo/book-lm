"""Load registered models and run step-by-step generation that reports every distribution."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from booklm.sampling import Distribution, SamplingParams, entropy_bits, process, sample

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    id: int
    token: str
    logit: float
    raw_prob: float
    scaled_prob: float
    final_prob: float
    kept: bool


@dataclass
class Step:
    index: int
    token_id: int
    token: str
    raw_prob: float  # model's own probability for the chosen token
    final_prob: float  # probability under the processed distribution it was drawn from
    rank: int  # rank of the chosen token under the raw distribution (0 = argmax)
    entropy_bits: float  # uncertainty of the raw distribution
    kept_count: int  # tokens that survived truncation
    kept_mass: float  # raw probability mass of the survivors
    candidates: list[Candidate] = field(default_factory=list)


@dataclass
class LoadedModel:
    name: str
    card: dict
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    lock: threading.Lock = field(default_factory=threading.Lock)

    def token_str(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]


def discover(models_dir: Path) -> list[str]:
    return sorted(p.parent.name for p in models_dir.glob("*/card.json"))


def load_model(models_dir: Path, name: str, device: torch.device) -> LoadedModel:
    path = models_dir / name
    card = json.loads((path / "card.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).to(device).eval()
    log.info("loaded model=%s params=%s device=%s", name, card.get("total_params"), device)
    return LoadedModel(name=name, card=card, model=model, tokenizer=tokenizer, device=device)


def describe(lm: LoadedModel, dist: Distribution, chosen: int, index: int, top_n: int) -> Step:
    """Summarise one distribution: top-n candidates by (penalised) logit, plus the chosen token."""
    top = torch.topk(dist.penalized_logits, top_n).indices.tolist()
    if chosen not in top:
        top.append(chosen)
    raw_rank = int((dist.raw_logits > dist.raw_logits[chosen]).sum())
    candidates = [
        Candidate(
            id=i,
            token=lm.token_str(i),
            logit=float(dist.raw_logits[i]),
            raw_prob=float(dist.raw_probs[i]),
            scaled_prob=float(dist.scaled_probs[i]),
            final_prob=float(dist.final_probs[i]),
            kept=bool(dist.keep[i]),
        )
        for i in top
    ]
    return Step(
        index=index,
        token_id=chosen,
        token=lm.token_str(chosen),
        raw_prob=float(dist.raw_probs[chosen]),
        final_prob=float(dist.final_probs[chosen]),
        rank=raw_rank,
        entropy_bits=entropy_bits(dist.raw_probs),
        kept_count=int(dist.keep.sum()),
        kept_mass=float(dist.raw_probs[dist.keep].sum()),
        candidates=candidates,
    )


@torch.no_grad()
def generate(
    lm: LoadedModel,
    prompt_ids: list[int],
    params: SamplingParams,
    max_new_tokens: int,
    top_n: int = 10,
    stop_at_eos: bool = True,
) -> Iterator[Step]:
    """Yield one Step per generated token. Holds the model lock for the whole generation."""
    if not prompt_ids:
        prompt_ids = [lm.tokenizer.bos_token_id or lm.tokenizer.eos_token_id]
    gen = torch.Generator().manual_seed(params.seed) if params.seed is not None else None
    history = torch.tensor(prompt_ids, dtype=torch.long)
    with lm.lock:
        input_ids = history.unsqueeze(0).to(lm.device)
        past = None
        for index in range(max_new_tokens):
            out = lm.model(input_ids=input_ids, past_key_values=past, use_cache=True)
            past = out.past_key_values
            # Sampling happens on CPU so a seed gives the same text on any device.
            dist = process(out.logits[0, -1].float().cpu(), params, history)
            chosen = sample(dist, gen)
            yield describe(lm, dist, chosen, index, top_n)
            if stop_at_eos and chosen == lm.tokenizer.eos_token_id:
                return
            history = torch.cat([history, torch.tensor([chosen])])
            input_ids = torch.tensor([[chosen]], device=lm.device)


@torch.no_grad()
def next_distribution(
    lm: LoadedModel, prompt_ids: list[int], params: SamplingParams, top_n: int = 10
) -> Step:
    """Inspect the next-token distribution after the prompt without committing to a continuation."""
    with closing(generate(lm, prompt_ids, params, 1, top_n, stop_at_eos=False)) as steps:
        return next(steps)


class Registry:
    def __init__(self, models_dir: Path, device: torch.device):
        self.models_dir = models_dir
        self.device = device
        self.models: dict[str, LoadedModel] = {}

    def load(self, names: list[str] | None = None) -> None:
        for name in names or discover(self.models_dir):
            start = time.time()
            self.models[name] = load_model(self.models_dir, name, self.device)
            log.info("model=%s ready in %.1fs", name, time.time() - start)

    def get(self, name: str) -> LoadedModel:
        return self.models[name]
