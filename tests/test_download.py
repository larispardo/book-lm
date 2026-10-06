import json

import pytest

from booklm.download import download_all, download_book

BOOK = (
    b"\xef\xbb\xbfheader\n"
    b"*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n"
    b"*** END OF THE PROJECT GUTENBERG EBOOK X ***\n"
)


class FakeFetcher:
    def __init__(self, payload: bytes = BOOK):
        self.payload = payload
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.payload


def test_downloads_once_then_uses_cache(tmp_path):
    fetcher = FakeFetcher()
    _, fresh = download_book(244, tmp_path, fetcher)
    _, again = download_book(244, tmp_path, fetcher)
    assert (fresh, again) == (True, False)
    assert fetcher.urls == ["https://www.gutenberg.org/cache/epub/244/pg244.txt"]


def test_force_redownloads(tmp_path):
    fetcher = FakeFetcher()
    download_book(244, tmp_path, fetcher)
    download_book(244, tmp_path, fetcher, force=True)
    assert len(fetcher.urls) == 2


def test_rejects_pages_that_are_not_books(tmp_path):
    with pytest.raises(ValueError, match="START/END"):
        download_book(244, tmp_path, FakeFetcher(b"<html>rate limited</html>"))
    assert list(tmp_path.iterdir()) == []  # nothing half-written left behind


def test_manifest_records_every_book(tmp_path):
    manifest = download_all(tmp_path, ids=[244, 1661], fetcher=FakeFetcher(), delay=0)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    assert [m["id"] for m in manifest] == [244, 1661]
    assert all(len(m["sha256"]) == 64 and m["bytes"] == len(BOOK) for m in manifest)
