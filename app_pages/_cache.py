"""Cached wrappers around booklm.tokenizer_lab, shared by the tokenization pages."""

import streamlit as st

from booklm import tokenizer_lab as lab
from booklm.tokenize_own import ByteBPE

DEFAULT_VOCAB = 8192


@st.cache_data(show_spinner=False)
def split(name: str) -> str:
    return lab.load_split(name)


@st.cache_resource(show_spinner="Training BPE…", max_entries=8)
def training_run(vocab_size: int) -> lab.TrainingRun:
    return lab.train_with_counts(split("train"), vocab_size - 256 - 1)


@st.cache_resource(max_entries=8)
def tokenizer(vocab_size: int) -> ByteBPE:
    return ByteBPE(training_run(vocab_size).merges)


@st.cache_data(show_spinner="Encoding both splits…", max_entries=8)
def stats(vocab_size: int) -> lab.VariantStats:
    return lab.variant_stats(tokenizer(vocab_size), split("train"), split("val"))


@st.cache_data(show_spinner="Replaying training…", max_entries=64)
def candidates(step: int) -> list[tuple[tuple[int, int], int]]:
    return lab.candidates_at_step(split("train"), step)


@st.cache_resource(show_spinner="Loading SmolLM2 tokenizer…")
def smollm2():
    return lab.smollm2()
