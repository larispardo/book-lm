"""Step 1: turn raw Gutenberg files into token ids with a tokenizer we train ourselves.

Four stages, in this order:

  1. normalize   strip Gutenberg boilerplate, unify unicode and quotes, unwrap hard line breaks
  2. partition   per book, the last 5% (by characters) goes to validation
  3. train BPE   byte-level BPE learned on the *train split only*, so no validation text
                 influences the vocabulary (that would be a small leak)
  4. encode      both splits -> uint16 token ids, books separated by <|endoftext|>

    python -m booklm.download            # once, fetches raw files
    python -m booklm.tokenize_own        # ~1-2 min

Outputs (under artifacts/):
  data/clean/{train,val}.txt                cleaned text, shared with step 2
  tokenizers/holmes-bpe-<V>/tokenizer.json  merges + settings, enough to encode/decode
  tokenizers/holmes-bpe-<V>/vocab.txt       every token, human readable: open it!
  data/tokens/holmes-bpe-<V>/{train,val}.bin + meta.json
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
import time
import unicodedata
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from booklm.download import BOOKS

EOT = "<|endoftext|>"  # document separator; never produced by BPE merges

# ── 1. normalize ──────────────────────────────────────────────────────────────────────

_START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", re.I)
_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", re.I)
# Curly and straight quotes mean the same thing; unifying them saves vocabulary slots.
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def normalize(raw: str) -> str:
    start, end = _START.search(raw), _END.search(raw)
    text = raw[start.end() if start else 0 : end.start() if end else len(raw)]
    # NFKC folds compatibility characters ("…" -> "...", ligatures -> letters).
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = text.replace("_", "")  # _word_ marks italics in Gutenberg plain text
    # Lines are hard-wrapped at ~70 chars. Blank lines separate paragraphs: keep those,
    # join everything else with single spaces.
    paragraphs = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paragraphs if p)


# ── 2. partition ──────────────────────────────────────────────────────────────────────


def split_book(text: str, val_fraction: float) -> tuple[str, str]:
    """Hold out the end of each book, cut on a paragraph boundary.

    A contiguous tail (rather than random paragraphs) keeps neighbouring sentences from
    landing on both sides of the split. Doing it per book puts every book in validation.
    """
    paragraphs = text.split("\n\n")
    budget = len(text) * (1 - val_fraction)
    seen = 0
    for i, p in enumerate(paragraphs):
        seen += len(p) + 2
        if seen >= budget:
            return "\n\n".join(paragraphs[: i + 1]), "\n\n".join(paragraphs[i + 1 :])
    return text, ""


# ── 3. byte-level BPE ─────────────────────────────────────────────────────────────────

# Pre-tokenisation: cut text into chunks (words with their leading space, numbers,
# punctuation runs, whitespace) so merges never cross word boundaries. Lossless: the
# chunks concatenate back to the original text. Same idea as GPT-2's pattern.
PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
_CHUNK = re.compile(PATTERN)


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair`, left to right, with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def train_bpe(text: str, num_merges: int) -> list[tuple[int, int]]:
    """Learn merges: repeatedly fuse the most frequent adjacent pair of tokens.

    Start: every chunk is a list of bytes (ids 0-255). Each merge creates id 256, 257, ...
    Speed trick: work on *unique* chunks weighted by frequency (~30k instead of ~1M), and
    only update the chunks that contain the merged pair. A heap finds the top pair fast;
    entries that went stale are skipped when popped.
    """
    counts = Counter(c for doc in text.split(EOT) for c in _CHUNK.findall(doc))
    words = [list(chunk.encode("utf-8")) for chunk in counts]
    freqs = list(counts.values())

    pair_counts: Counter[tuple[int, int]] = Counter()
    where: defaultdict[tuple[int, int], set[int]] = defaultdict(set)
    for wi, (word, freq) in enumerate(zip(words, freqs, strict=True)):
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] += freq
            where[pair].add(wi)
    heap = [(-count, pair) for pair, count in pair_counts.items()]
    heapq.heapify(heap)

    merges: list[tuple[int, int]] = []
    while len(merges) < num_merges and heap:
        neg_count, pair = heapq.heappop(heap)
        if pair_counts.get(pair, 0) != -neg_count:
            continue  # stale entry: this pair's count changed since it was pushed
        if -neg_count < 2:
            break  # nothing left that occurs more than once
        new_id = 256 + len(merges)
        merges.append(pair)
        changed = set()
        for wi in where.pop(pair):
            word, freq = words[wi], freqs[wi]
            for p in zip(word, word[1:], strict=False):
                pair_counts[p] -= freq
                changed.add(p)
            word = words[wi] = merge(word, pair, new_id)
            for p in zip(word, word[1:], strict=False):
                pair_counts[p] += freq
                where[p].add(wi)
                changed.add(p)
        for p in changed:
            if pair_counts[p] > 0:
                heapq.heappush(heap, (-pair_counts[p], p))
            else:
                del pair_counts[p]
    return merges


class ByteBPE:
    """Encode/decode with learned merges. Ids: 0-255 bytes, then merges, then specials."""

    def __init__(self, merges: list[tuple[int, int]], special_tokens: list[str] = (EOT,)):
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab = {i: bytes([i]) for i in range(256)}
        for rank, (a, b) in enumerate(self.merges):
            self.vocab[256 + rank] = self.vocab[a] + self.vocab[b]
        first_special = 256 + len(self.merges)
        self.special = {tok: first_special + i for i, tok in enumerate(special_tokens)}
        self.special_by_id = {i: tok for tok, i in self.special.items()}
        self._split_special = re.compile("(" + "|".join(map(re.escape, self.special)) + ")")
        self._cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.special)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        # Apply merges in the order they were learned: always the lowest-rank pair first.
        while len(ids) > 1:
            rank, pair = min(
                (self.ranks.get(p, float("inf")), p) for p in zip(ids, ids[1:], strict=False)
            )
            if rank == float("inf"):
                break
            ids = merge(ids, pair, 256 + rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._split_special.split(text):
            if piece in self.special:
                ids.append(self.special[piece])
            else:
                for chunk in _CHUNK.findall(piece):
                    ids.extend(self._encode_chunk(chunk))
        return ids

    def token_bytes(self, token_id: int) -> bytes:
        if token_id in self.special_by_id:
            return self.special_by_id[token_id].encode()
        return self.vocab[token_id]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors="replace")

    def save(self, directory: Path, **extra) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        config = {
            "type": "byte-level-bpe",
            "vocab_size": self.vocab_size,
            "pattern": PATTERN,
            "special_tokens": self.special,
            "merges": self.merges,
            **extra,
        }
        (directory / "tokenizer.json").write_text(json.dumps(config))
        with open(directory / "vocab.txt", "w", encoding="utf-8") as f:
            f.write("id\ttoken\tmerged_from\n")
            for i in range(self.vocab_size):
                text = self.token_bytes(i).decode("utf-8", errors="replace")
                parents = self.merges[i - 256] if 256 <= i < 256 + len(self.merges) else ""
                f.write(f"{i}\t{text!r}\t{parents}\n")

    @classmethod
    def load(cls, directory: Path) -> ByteBPE:
        config = json.loads((directory / "tokenizer.json").read_text())
        specials = sorted(config["special_tokens"], key=config["special_tokens"].get)
        return cls(config["merges"], specials)


# ── 4. encode + run everything ────────────────────────────────────────────────────────


def write_ids(path: Path, ids: list[int]) -> None:
    """uint16 is enough while vocab < 65,536 and halves the file vs int32."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        array("H", ids).tofile(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize, partition, train BPE, encode.")
    parser.add_argument("--raw-dir", type=Path, default=Path("artifacts/data/raw"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()

    # 1 + 2
    splits: dict[str, list[str]] = {"train": [], "val": []}
    for book_id, title in BOOKS.items():
        raw = (args.raw_dir / f"pg{book_id}.txt").read_text(encoding="utf-8-sig")
        train, val = split_book(normalize(raw), args.val_fraction)
        splits["train"].append(train)
        splits["val"].append(val)
        print(f"  {title:<36} train {len(train):>9,} chars   val {len(val):>7,} chars")
    clean_dir = args.out / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    for split, docs in splits.items():
        texts[split] = f"\n\n{EOT}\n\n".join(docs)
        (clean_dir / f"{split}.txt").write_text(texts[split], encoding="utf-8")

    # 3
    start = time.time()
    merges = train_bpe(texts["train"], num_merges=args.vocab_size - 256 - 1)
    tokenizer = ByteBPE(merges)
    print(f"trained {len(merges):,} merges in {time.time() - start:.0f}s")
    name = f"holmes-bpe-{tokenizer.vocab_size}"
    tokenizer.save(args.out / "tokenizers" / name, trained_on="data/clean/train.txt")

    # 4
    meta = {"tokenizer": name, "vocab_size": tokenizer.vocab_size, "dtype": "uint16"}
    for split, text in texts.items():
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text, "round trip failed"
        write_ids(args.out / "data" / "tokens" / name / f"{split}.bin", ids)
        n_bytes = len(text.encode("utf-8"))
        meta[split] = {
            "chars": len(text),
            "tokens": len(ids),
            "bytes_per_token": n_bytes / len(ids),
        }
        print(f"{split:>5}: {len(ids):>9,} tokens  ({n_bytes / len(ids):.2f} bytes/token)")
    (args.out / "data" / "tokens" / name / "meta.json").write_text(json.dumps(meta, indent=2))

    longest = sorted(range(256, 256 + len(merges)), key=lambda i: -len(tokenizer.vocab[i]))[:12]
    print("longest tokens:", ", ".join(repr(tokenizer.decode([i])) for i in longest))


if __name__ == "__main__":
    main()
