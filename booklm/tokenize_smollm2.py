"""Step 2: tokenize the same cleaned splits with SmolLM2's pretrained tokenizer.

The finetuning step starts from SmolLM2-135M, whose embedding table is tied to this exact
vocabulary (49,152 tokens). A model can only read ids from the tokenizer it was trained
with, so here we don't train anything: we load it and encode.

Reads the cleaned text written by step 1, so both tokenizers see identical data.

    python -m booklm.tokenize_smollm2

Outputs: artifacts/data/tokens/smollm2/{train,val}.bin + meta.json
"""

from __future__ import annotations

import argparse
import json
from array import array
from pathlib import Path

from tokenizers import Tokenizer

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
EOT = "<|endoftext|>"  # step 1 separates books with this; SmolLM2 has the same special token


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode cleaned splits with SmolLM2.")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    tokenizer = Tokenizer.from_pretrained(MODEL_ID)
    eot_id = tokenizer.token_to_id(EOT)
    vocab_size = tokenizer.get_vocab_size()
    assert eot_id is not None and vocab_size < 2**16, "need <|endoftext|> and a uint16 vocab"

    out_dir = args.out / "data" / "tokens" / "smollm2"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": MODEL_ID, "vocab_size": vocab_size, "dtype": "uint16"}
    for split in ("train", "val"):
        text = (args.out / "data" / "clean" / f"{split}.txt").read_text(encoding="utf-8")
        ids: list[int] = []
        for i, doc in enumerate(text.split(f"\n\n{EOT}\n\n")):
            if i:
                ids.append(eot_id)
            ids.extend(tokenizer.encode(doc, add_special_tokens=False).ids)
        with open(out_dir / f"{split}.bin", "wb") as f:
            array("H", ids).tofile(f)
        n_bytes = len(text.encode("utf-8"))
        meta[split] = {
            "chars": len(text),
            "tokens": len(ids),
            "bytes_per_token": n_bytes / len(ids),
        }
        print(f"{split:>5}: {len(ids):>9,} tokens  ({n_bytes / len(ids):.2f} bytes/token)")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    sample = 'Holmes looked at me and said, "Elementary, my dear Watson."'
    pieces = [tokenizer.decode([i]) for i in tokenizer.encode(sample).ids]
    print(f"\nSmolLM2 splits {len(pieces)} tokens:", " | ".join(pieces))


if __name__ == "__main__":
    main()
