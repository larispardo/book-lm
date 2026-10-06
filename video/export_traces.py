"""Run the trained models once and dump everything the manim showcase animates.

Writes video/data/trace.json and video/data/logits.npy (full-vocab logits for one step, so the
video can recompute softmax at any temperature exactly instead of faking it).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from booklm.config import get_settings, resolve_device
from booklm.engine import Registry, generate
from booklm.sampling import SamplingParams

OUT = Path(__file__).parent / "data"
PROMPT = "Holmes looked at me and said"
FOCUS = "holmes-lora"
ORDER = ["holmes-scratch", "smollm2-base", "holmes-lora"]
BALANCED = SamplingParams(temperature=0.8, top_p=0.95, seed=11)


def run(lm, ids, params, n, top_n=8):
    return [asdict(s) for s in generate(lm, ids, params, n, top_n)]


def main() -> None:
    settings = get_settings()
    registry = Registry(settings.models_dir, resolve_device(settings.device))
    registry.load([m for m in ORDER if (settings.models_dir / m / "card.json").exists()])
    lm = registry.get(FOCUS)
    ids = lm.encode(PROMPT)
    OUT.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        logits = lm.model(input_ids=torch.tensor([ids], device=lm.device)).logits[0, -1]
    np.save(OUT / "logits.npy", logits.float().cpu().numpy())

    walk = run(lm, ids, BALANCED, 14)
    # Branch at the most uncertain early step: force the runner-up token and continue greedily.
    fork = max(walk[:6], key=lambda s: s["entropy_bits"])
    alt = next(c for c in fork["candidates"] if c["id"] != fork["token_id"])
    prefix = ids + [s["token_id"] for s in walk[: fork["index"]]]
    greedy = SamplingParams(temperature=0)
    branches = {
        "fork_index": fork["index"],
        "original": run(lm, prefix + [fork["token_id"]], greedy, 10),
        "alternative": run(lm, prefix + [alt["id"]], greedy, 10),
        "original_token": fork["token"],
        "alternative_token": alt["token"],
        "original_prob": fork["candidates"][0]["raw_prob"],
        "alternative_prob": alt["raw_prob"],
    }

    compare = {
        name: "".join(s["token"] for s in run(registry.get(name), ids, BALANCED, 40, 1))
        for name in registry.models
    }
    cards = {name: registry.get(name).card for name in registry.models}
    trace = {
        "prompt": PROMPT,
        "prompt_tokens": [lm.token_str(i) for i in ids],
        "focus_model": FOCUS,
        "vocab_size": len(lm.tokenizer),
        "token_strs": {str(i): lm.token_str(i) for i in torch.topk(logits, 200).indices.tolist()},
        "walk": walk,
        "branches": branches,
        "compare": compare,
        "cards": cards,
        "corpus": json.loads((settings.data_dir / "meta.json").read_text()),
    }
    (OUT / "trace.json").write_text(json.dumps(trace, indent=1))
    print(f"wrote {OUT}/trace.json")
    for name, text in compare.items():
        print(f"\n[{name}] {PROMPT}{text}")


if __name__ == "__main__":
    main()
