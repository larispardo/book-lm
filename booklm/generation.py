"""Generate a fixed amount of *text*, not a fixed number of tokens.

Comparing models with different tokenizers on "40 tokens each" is unfair: 40 tokens of a
1k vocabulary is ~100 characters, of SmolLM2's 49k vocabulary ~170. A character budget
gives every model the same amount of text to prove itself.
"""

from __future__ import annotations

import torch

from booklm.model import GPT
from booklm.text_codec import Codec


@torch.no_grad()
def generate_chars(
    model: GPT,
    codec: Codec,
    prompt: str,
    max_chars: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int = 0,
) -> str:
    """Sample until the continuation reaches `max_chars` characters (or <|endoftext|>).

    Returns only the continuation, cut to `max_chars`. A fixed seed makes it reproducible.
    """
    device = next(model.parameters()).device
    gen = torch.Generator().manual_seed(seed)
    ids = codec.encode(prompt)
    start = len(codec.decode(ids))
    text = ""
    for _ in range(4 * max_chars):  # safety cap: a token is at least one byte
        context = torch.tensor([ids[-model.cfg.context :]], device=device)
        logits = model(context)[0, -1].float().cpu()
        if temperature == 0:
            nxt = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=gen))
        if nxt == codec.eot_id:
            break
        ids.append(nxt)
        text = codec.decode(ids)[start:]
        if len(text) >= max_chars:
            break
    return text[:max_chars]
