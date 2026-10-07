import math

import pytest
import torch

from booklm.generation import generate_chars
from booklm.metrics import bits_per_byte
from booklm.model import GPT, GPTConfig
from booklm.text_codec import Codec
from booklm.tokenize_own import ByteBPE, train_bpe


@pytest.fixture(scope="module")
def codec():
    bpe = ByteBPE(train_bpe("Holmes and Watson walked to Baker Street. " * 20, num_merges=30))
    return Codec("toy", bpe.vocab_size, bpe.special["<|endoftext|>"], bpe.encode, bpe.decode)


@pytest.fixture(scope="module")
def model(codec):
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=codec.vocab_size, context=32, d_model=32, n_layers=1, n_heads=2)
    return GPT(cfg).eval()


def test_generation_respects_the_character_budget(model, codec):
    text = generate_chars(model, codec, "Holmes", max_chars=50, seed=1)
    assert 0 < len(text) <= 50


def test_generation_is_reproducible_with_a_seed(model, codec):
    a = generate_chars(model, codec, "Holmes", max_chars=40, seed=7)
    b = generate_chars(model, codec, "Holmes", max_chars=40, seed=7)
    assert a == b


# Exercise: implement metrics.bits_per_byte, then delete this marker.
def test_bits_per_byte():
    assert bits_per_byte(math.log(256), tokens=100, n_bytes=100) == pytest.approx(8.0)
    assert bits_per_byte(math.log(2), tokens=1, n_bytes=1) == pytest.approx(1.0)
    # Same total information spread over twice the bytes -> half the bits per byte.
    assert bits_per_byte(2.0, tokens=50, n_bytes=200) == pytest.approx(0.5 / math.log(2))
    # Our small model: 4.16 nats/token on the 8k tokenizer's val split.
    assert bits_per_byte(4.16, tokens=44_351, n_bytes=175_604) == pytest.approx(1.516, abs=1e-3)
