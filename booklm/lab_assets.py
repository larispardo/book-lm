"""Ship the lab's prebuilt files (data, tokenizer, checkpoints) with a GitHub release.

`artifacts/` is gitignored, so a fresh clone (e.g. Streamlit Community Cloud) has no data.
The app calls `ensure()` on start: if the files are missing it downloads one archive from
the release and unpacks it.

    python -m booklm.lab_assets pack       # build dist/lab-artifacts.tar.gz from artifacts/
    gh release create lab-v1 dist/lab-artifacts.tar.gz
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

TAG = "lab-v1"
URL = f"https://github.com/larispardo/book-lm/releases/download/{TAG}/lab-artifacts.tar.gz"
MARKER = Path("artifacts/data/clean/train.txt")

# Everything the lab panels read; raw downloads and older experiments stay out.
PATTERNS = [
    "artifacts/data/clean/*.txt",
    "artifacts/data/tokens/*/*",
    "artifacts/tokenizers/holmes-bpe-8192/*",
    "artifacts/models/holmes-gpt-*/model.pt",
    "artifacts/models/holmes-gpt-*/log.jsonl",
]


def pack(out: Path = Path("dist/lab-artifacts.tar.gz"), root: Path = Path(".")) -> list[Path]:
    files = sorted({f for pattern in PATTERNS for f in root.glob(pattern) if f.is_file()})
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=str(f.relative_to(root)))
    return files


def ensure(root: Path = Path("."), url: str = URL) -> bool:
    """Download and unpack the lab files if they are missing. Returns True if it fetched."""
    if (root / MARKER).exists():
        return False
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp:
        with urllib.request.urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, tmp)
        tmp.flush()
        with tarfile.open(tmp.name, "r:gz") as tar:
            # "data" filter refuses absolute paths, links outside root and device files.
            tar.extractall(root, filter="data")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack or fetch the lab artifacts.")
    parser.add_argument("command", choices=["pack", "fetch"])
    args = parser.parse_args()
    if args.command == "pack":
        files = pack()
        size = Path("dist/lab-artifacts.tar.gz").stat().st_size
        print(f"packed {len(files)} files into dist/lab-artifacts.tar.gz ({size / 1e6:.1f} MB)")
    else:
        print("fetched" if ensure() else "already present")


if __name__ == "__main__":
    main()
