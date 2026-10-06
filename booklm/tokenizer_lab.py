"""Analysis helpers behind the tokenization lab panels. No Streamlit in here."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from booklm.tokenize_own import ByteBPE, iter_merges, merge

ARTIFACTS = Path("artifacts")
# Reference model width used to translate vocab size into embedding parameters.
D_MODEL = 384
SMOLLM2_ID = "HuggingFaceTB/SmolLM2-135M"

# Sample texts chosen to stress tokenizers differently.
SAMPLES = {
    "Classic quote": 'Holmes looked at me and said, "Elementary, my dear Watson."',
    "Rare Victorian words": (
        "The commissionnaire handed me a telegram from Northumberland, marked most "
        "confidential, concerning the disappearance of the ambassador's correspondence."
    ),
    "Names and numbers": (
        "Inspector Lestrade met Mycroft and Irene Adler at 221B Baker Street at "
        "half-past nine on the 4th of March, 1881."
    ),
    "Modern English": (
        "I deployed the Kubernetes cluster from my smartphone while streaming a podcast "
        "about cryptocurrency."
    ),
    "Spanish": "Holmes miró por la ventana y dijo que el asesino seguía en la ciudad.",
    "Code": "def solve(case):\n    return min(case.suspects, key=lambda s: s.alibi)",
}


def load_split(split: str, artifacts: Path = ARTIFACTS) -> str:
    return (artifacts / "data" / "clean" / f"{split}.txt").read_text(encoding="utf-8")


def random_paragraph(text: str, min_chars: int = 120, max_chars: int = 500) -> str:
    """A random mid-length paragraph, e.g. from the validation split."""
    import random

    paragraphs = [p for p in text.split("\n\n") if min_chars <= len(p) <= max_chars]
    return random.choice(paragraphs)


def load_meta(name: str, artifacts: Path = ARTIFACTS) -> dict:
    return json.loads((artifacts / "data" / "tokens" / name / "meta.json").read_text())


# ── training with a record of every greedy choice ───────────────────────────────────────


@dataclass
class TrainingRun:
    merges: list[tuple[int, int]]
    counts: list[int]  # how often each chosen pair occurred at the moment it was chosen


def train_with_counts(text: str, num_merges: int) -> TrainingRun:
    run = TrainingRun([], [])
    for pair, count, _ in iter_merges(text):
        if len(run.merges) == num_merges:
            break
        run.merges.append(pair)
        run.counts.append(count)
    return run


def candidates_at_step(text: str, step: int, top: int = 10) -> list[tuple[tuple[int, int], int]]:
    """Replay training up to merge `step` and return the pairs it chose between."""
    for i, (_, _, pair_counts) in enumerate(iter_merges(text)):
        if i == step:
            return heapq.nlargest(top, pair_counts.items(), key=lambda kv: kv[1])
    return []


# ── representation ───────────────────────────────────────────────────────────────────


def visible(piece: str) -> str:
    """Make whitespace visible: spaces as ·, newlines as ↵."""
    return piece.replace(" ", "·").replace("\n", "↵")


def pieces(bpe: ByteBPE, text: str) -> list[tuple[int, str]]:
    return [(i, bpe.decode([i])) for i in bpe.encode(text)]


def tokens_at_step(merges: list[tuple[int, int]], step: int, text: str) -> list[tuple[int, str]]:
    """How `text` tokenizes if training had stopped after `step` merges."""
    return pieces(ByteBPE(merges[:step]), text)


def merge_trace(bpe: ByteBPE, text: str) -> list[dict]:
    """Every merge applied while encoding one chunk of text, in order."""
    ids = list(text.encode("utf-8"))
    trace = [{"rank": None, "pair": None, "tokens": [bpe.decode([i]) for i in ids]}]
    while len(ids) > 1:
        rank, pair = min(
            (bpe.ranks.get(p, float("inf")), p) for p in zip(ids, ids[1:], strict=False)
        )
        if rank == float("inf"):
            break
        ids = merge(ids, pair, 256 + rank)
        trace.append(
            {
                "rank": rank,
                "pair": (bpe.decode([pair[0]]), bpe.decode([pair[1]])),
                "tokens": [bpe.decode([i]) for i in ids],
            }
        )
    return trace


_CHIP_COLORS = ("blue", "orange", "green", "violet")
_MARKDOWN_SPECIAL = set("\\`*_{}[]()#+-.!|$~<>:")


def chips_markdown(token_strings: list[str]) -> str:
    """Render tokens as coloured chips with Streamlit's :color-background[...] markdown."""
    out = []
    for i, tok in enumerate(token_strings):
        text = "".join("\\" + c if c in _MARKDOWN_SPECIAL else c for c in visible(tok))
        out.append(f":{_CHIP_COLORS[i % len(_CHIP_COLORS)]}-background[{text or '∅'}]")
    return " ".join(out)


# ── comparing vocabulary sizes ───────────────────────────────────────────────────────────


@dataclass
class VariantStats:
    vocab_size: int
    merges_learned: int
    train_bytes_per_token: float
    val_bytes_per_token: float
    embedding_params: int

    @property
    def generalization_gap(self) -> float:
        """How much worse val compresses than train, in percent."""
        return 100 * (1 - self.val_bytes_per_token / self.train_bytes_per_token)


def train_variant(train_text: str, vocab_size: int) -> ByteBPE:
    return ByteBPE(train_with_counts(train_text, vocab_size - 256 - 1).merges)


def variant_stats(bpe: ByteBPE, train_text: str, val_text: str) -> VariantStats:
    def bytes_per_token(text: str) -> float:
        return len(text.encode("utf-8")) / len(bpe.encode(text))

    return VariantStats(
        vocab_size=bpe.vocab_size,
        merges_learned=len(bpe.merges),
        train_bytes_per_token=bytes_per_token(train_text),
        val_bytes_per_token=bytes_per_token(val_text),
        embedding_params=bpe.vocab_size * D_MODEL,
    )


# ── SmolLM2 ──────────────────────────────────────────────────────────────────────────────


@cache
def smollm2():
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(SMOLLM2_ID)


def smollm2_pieces(text: str) -> list[dict]:
    """Each token as id, raw vocab entry (GPT-2 byte alphabet, e.g. 'Ġthe') and real text."""
    tok = smollm2()
    enc = tok.encode(text, add_special_tokens=False)
    return [
        {"id": i, "vocab_entry": raw, "text": tok.decode([i])}
        for i, raw in zip(enc.ids, enc.tokens, strict=True)
    ]


# ── exercise ─────────────────────────────────────────────────────────────────────────────


def merge_tree(bpe: ByteBPE, token_id: int) -> dict:
    """How a token was built, as a nested tree. YOUR EXERCISE.

    Expected shape, for ' the' (id 262 = 256 + 257, where 256 = ' ' + 't', 257 = 'h' + 'e'):

        {"id": 262, "token": " the", "children": [
            {"id": 256, "token": " t", "children": [
                {"id": 32, "token": " ", "children": []},
                {"id": 116, "token": "t", "children": []}]},
            {"id": 257, "token": "he", "children": [...]}]}

    Hints: ids below 256 are raw bytes (leaves). For a merged id, `bpe.merges[id - 256]`
    gives the two ids it was glued from. Recursion does the rest.
    """
    raise NotImplementedError
