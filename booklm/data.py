"""Download the Sherlock Holmes canon from Project Gutenberg, clean it, and tokenize it.

All nine books are public domain in the United States. Output:
  artifacts/data/corpus.txt     cleaned text, books separated by blank lines
  artifacts/data/train.pt       int32 token ids (first 95%)
  artifacts/data/val.pt         int32 token ids (last 5%)
  artifacts/data/meta.json      token counts and provenance
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer

from booklm.config import BASE_MODEL_ID, get_settings

BOOKS = {
    244: "A Study in Scarlet",
    2097: "The Sign of the Four",
    1661: "The Adventures of Sherlock Holmes",
    834: "The Memoirs of Sherlock Holmes",
    2852: "The Hound of the Baskervilles",
    108: "The Return of Sherlock Holmes",
    3289: "The Valley of Fear",
    2350: "His Last Bow",
    69700: "The Case-Book of Sherlock Holmes",
}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
VAL_FRACTION = 0.05

_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def strip_gutenberg(raw: str) -> str:
    """Keep only the book body between the Gutenberg START and END markers."""
    start = _START.search(raw)
    end = _END.search(raw)
    body = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    body = body.replace("\r\n", "\n").replace("_", "")
    # Unwrap hard-wrapped lines inside paragraphs; keep paragraph breaks.
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body)]
    return "\n\n".join(p for p in paragraphs if p)


def download(book_id: int, cache_dir: Path) -> str:
    path = cache_dir / f"pg{book_id}.txt"
    if not path.exists():
        with urllib.request.urlopen(URL.format(id=book_id), timeout=60) as resp:
            path.write_bytes(resp.read())
    return path.read_text(encoding="utf-8-sig")


def build(data_dir: Path) -> dict:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    for book_id, title in BOOKS.items():
        texts.append(strip_gutenberg(download(book_id, raw_dir)))
        print(f"  {title}: {len(texts[-1]):,} chars")
    corpus = "\n\n\n".join(texts)
    (data_dir / "corpus.txt").write_text(corpus, encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    ids = torch.tensor(tokenizer(corpus)["input_ids"], dtype=torch.int32)
    split = int(len(ids) * (1 - VAL_FRACTION))
    torch.save(ids[:split].clone(), data_dir / "train.pt")
    torch.save(ids[split:].clone(), data_dir / "val.pt")

    meta = {
        "source": "Project Gutenberg (public domain)",
        "books": BOOKS,
        "tokenizer": BASE_MODEL_ID,
        "chars": len(corpus),
        "tokens": len(ids),
        "train_tokens": split,
        "val_tokens": len(ids) - split,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=get_settings().data_dir)
    args = parser.parse_args()
    meta = build(args.data_dir)
    print(f"corpus: {meta['chars']:,} chars, {meta['tokens']:,} tokens")


if __name__ == "__main__":
    main()
