"""One interface over every tokenizer we train with: ours (any vocab size) or SmolLM2's.

codec = load_codec("holmes-bpe-8192")   # artifacts/tokenizers/holmes-bpe-8192
codec = load_codec("smollm2")           # SmolLM2's pretrained tokenizer
ids = codec.encode(text); codec.decode(ids); codec.vocab_size; codec.eot_id
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from booklm.tokenize_own import EOT, ByteBPE

SMOLLM2_ID = "HuggingFaceTB/SmolLM2-135M"
DEFAULT = "holmes-bpe-8192"


@dataclass
class Codec:
    name: str
    vocab_size: int
    eot_id: int
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]


def load_codec(name: str, artifacts: Path = Path("artifacts")) -> Codec:
    if name == "smollm2":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_pretrained(SMOLLM2_ID)
        return Codec(
            name=name,
            vocab_size=tok.get_vocab_size(),
            eot_id=tok.token_to_id(EOT),
            encode=lambda text: tok.encode(text, add_special_tokens=False).ids,
            decode=lambda ids: tok.decode(ids, skip_special_tokens=False),
        )
    bpe = ByteBPE.load(artifacts / "tokenizers" / name)
    return Codec(
        name=name,
        vocab_size=bpe.vocab_size,
        eot_id=bpe.special[EOT],
        encode=bpe.encode,
        decode=bpe.decode,
    )
