"""Download the Sherlock Holmes canon from Project Gutenberg.

Polite by design: it identifies itself, waits between requests, retries with backoff,
and never re-downloads a file already on disk. Files are saved exactly as served;
cleaning is a separate step, so it can be redone without hitting the network again.

    python -m booklm.download                 # all nine books -> artifacts/data/raw/
    python -m booklm.download --only 244 1661
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

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
USER_AGENT = "book-lm/0.1 (+https://github.com/larispardo/book-lm)"
DEFAULT_OUT = Path("artifacts/data/raw")

Fetcher = Callable[[str], bytes]


def fetch(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    """GET a URL, retrying transient failures (network errors, 429, 5xx) with backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            if err.code != 429 and err.code < 500 or attempt == retries:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def looks_like_book(text: str) -> bool:
    """A real Gutenberg ebook has START and END markers around the body."""
    return "*** START OF" in text and "*** END OF" in text


def download_book(
    book_id: int, out_dir: Path, fetcher: Fetcher = fetch, force: bool = False
) -> tuple[Path, bool]:
    """Save one book to out_dir/pg{id}.txt. Returns (path, downloaded_now)."""
    path = out_dir / f"pg{book_id}.txt"
    if path.exists() and not force:
        return path, False
    data = fetcher(URL.format(id=book_id))
    if not looks_like_book(data.decode("utf-8-sig", errors="replace")):
        raise ValueError(f"book {book_id}: response has no Gutenberg START/END markers")
    # Write to a temp file and rename, so an interrupted run never leaves a half file behind.
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path, True


def download_all(
    out_dir: Path,
    ids: list[int] | None = None,
    fetcher: Fetcher = fetch,
    delay: float = 2.0,
    force: bool = False,
) -> list[dict]:
    """Download the requested books and write manifest.json describing what is on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for book_id in ids or list(BOOKS):
        path, fresh = download_book(book_id, out_dir, fetcher, force)
        data = path.read_bytes()
        manifest.append(
            {
                "id": book_id,
                "title": BOOKS.get(book_id, "?"),
                "url": URL.format(id=book_id),
                "file": path.name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        print(f"{'downloaded' if fresh else 'cached    '}  {book_id:>5}  {BOOKS.get(book_id, '?')}")
        if fresh and delay:
            time.sleep(delay)  # be gentle with Gutenberg's servers
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Sherlock Holmes canon.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", type=int, nargs="+", help="Gutenberg ids (default: all nine)")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds between downloads")
    parser.add_argument("--force", action="store_true", help="re-download files already on disk")
    args = parser.parse_args()
    manifest = download_all(args.out, args.only, delay=args.delay, force=args.force)
    total = sum(entry["bytes"] for entry in manifest)
    print(f"{len(manifest)} books, {total / 1e6:.1f} MB in {args.out}")


if __name__ == "__main__":
    main()
