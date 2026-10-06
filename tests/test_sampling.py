import pytest
import torch

from booklm.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    entropy_bits,
    min_p_mask,
    process,
    sample,
    top_k_mask,
    top_p_mask,
)

LOGITS = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0, -1.0])
NO_HISTORY = torch.tensor([], dtype=torch.long)


def test_greedy_keeps_only_argmax():
    dist = process(LOGITS, SamplingParams(temperature=0), NO_HISTORY)
    assert dist.keep.tolist() == [True, False, False, False, False, False]
    assert sample(dist) == 0


def test_lower_temperature_sharpens():
    cold = process(LOGITS, SamplingParams(temperature=0.5), NO_HISTORY).final_probs
    hot = process(LOGITS, SamplingParams(temperature=2.0), NO_HISTORY).final_probs
    assert cold[0] > hot[0]
    assert entropy_bits(cold) < entropy_bits(hot)


def test_temperature_does_not_change_raw_probs():
    dist = process(LOGITS, SamplingParams(temperature=2.0), NO_HISTORY)
    assert torch.allclose(dist.raw_probs, torch.softmax(LOGITS, -1))


def test_top_k_keeps_exactly_k():
    assert top_k_mask(LOGITS, 3).tolist() == [True, True, True, False, False, False]
    assert top_k_mask(LOGITS, 0).all()


def test_top_p_keeps_smallest_prefix_reaching_p():
    probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
    assert top_p_mask(probs, 0.5).tolist() == [True, False, False, False]
    assert top_p_mask(probs, 0.8).tolist() == [True, True, False, False]
    assert top_p_mask(probs, 0.81).tolist() == [True, True, True, False]
    assert top_p_mask(probs, 0.01).tolist() == [True, False, False, False]


def test_min_p_is_relative_to_top_token():
    probs = torch.tensor([0.6, 0.3, 0.07, 0.03])
    assert min_p_mask(probs, 0.1).tolist() == [True, True, True, False]


def test_repetition_penalty_pushes_seen_tokens_down():
    logits = torch.tensor([2.0, -2.0, 1.0])
    out = apply_repetition_penalty(logits, torch.tensor([0, 1]), 2.0)
    assert out.tolist() == [1.0, -4.0, 1.0]


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(),
        SamplingParams(temperature=0.7, top_k=2),
        SamplingParams(top_p=0.3),
        SamplingParams(min_p=0.5, repetition_penalty=1.3),
    ],
)
def test_final_distribution_is_normalised_and_respects_mask(params):
    dist = process(LOGITS, params, torch.tensor([0]))
    assert dist.final_probs.sum().item() == pytest.approx(1.0)
    assert (dist.final_probs[~dist.keep] == 0).all()


def test_seed_makes_sampling_reproducible():
    dist = process(LOGITS, SamplingParams(temperature=1.5), NO_HISTORY)
    draws = [sample(dist, torch.Generator().manual_seed(7)) for _ in range(3)]
    assert len(set(draws)) == 1
