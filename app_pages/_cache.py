"""Cached wrappers around booklm.tokenizer_lab, shared by the tokenization pages.

BPE is greedy and deterministic, so a smaller vocabulary is always a prefix of a bigger
one. We train once, as far as the data allows, and slice that run for every size.
"""

import streamlit as st

from booklm import tokenizer_lab as lab
from booklm.tokenize_own import ByteBPE

DEFAULT_VOCAB = 8192
SPECIALS = 1  # <|endoftext|>


@st.cache_data(show_spinner=False)
def split(name: str) -> str:
    return lab.load_split(name)


@st.cache_resource(show_spinner="Training BPE until no pair repeats…")
def full_run() -> lab.TrainingRun:
    return lab.train_with_counts(split("train"), num_merges=10**9)


def max_vocab() -> int:
    return 256 + len(full_run().merges) + SPECIALS


def training_run(vocab_size: int) -> lab.TrainingRun:
    run, n = full_run(), vocab_size - 256 - SPECIALS
    return lab.TrainingRun(run.merges[:n], run.counts[:n])


@st.cache_resource(max_entries=16)
def tokenizer(vocab_size: int) -> ByteBPE:
    return ByteBPE(training_run(vocab_size).merges)


@st.cache_data(show_spinner="Encoding both splits…", max_entries=16)
def stats(vocab_size: int) -> lab.VariantStats:
    return lab.variant_stats(tokenizer(vocab_size), split("train"), split("val"))


@st.cache_data(show_spinner="Replaying training…", max_entries=64)
def candidates(step: int) -> list[tuple[tuple[int, int], int]]:
    return lab.candidates_at_step(split("train"), step)


@st.cache_resource(show_spinner="Loading SmolLM2 tokenizer…")
def smollm2():
    return lab.smollm2()


@st.cache_data(show_spinner=False)
def word_counts() -> dict[str, int]:
    return dict(lab.word_counts(split("train")))


@st.cache_data(show_spinner=False)
def frequency_words() -> list[tuple[str, int]]:
    return lab.words_by_frequency(word_counts())


# ── pretraining ──────────────────────────────────────────────────────────────────────────


def _run_dir(name: str):
    from booklm.pretrain_lab import MODELS

    return MODELS / name


def run_log(name: str):
    """Logs grow while a run trains, so the cache key includes the file's mtime."""
    return _run_log(name, (_run_dir(name) / "log.jsonl").stat().st_mtime)


@st.cache_data(show_spinner=False, max_entries=32)
def _run_log(name: str, mtime: float):
    from booklm import pretrain_lab

    return pretrain_lab.load_run(_run_dir(name))


def gpt_model(name: str):
    """Reload when a run saves a better checkpoint (mtime changes)."""
    return _gpt_model(name, (_run_dir(name) / "model.pt").stat().st_mtime)


@st.cache_resource(show_spinner="Loading checkpoint…", max_entries=4)
def _gpt_model(name: str, mtime: float):
    from booklm import pretrain_lab

    return pretrain_lab.load_model(_run_dir(name))
