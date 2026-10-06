import re

import pytest

from booklm.tokenize_own import (
    _CHUNK,
    EOT,
    ByteBPE,
    merge,
    normalize,
    split_book,
    train_bpe,
)

RAW = (
    "Project Gutenberg licence blah\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK THE SIGN OF FOUR ***\n\n"
    "“You have been in Afghanistan,\n"
    "I perceive,” said _Holmes_.\n\n\n"
    "‘Twas brillig…\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK THE SIGN OF FOUR ***\n"
    "more licence\n"
)


def test_normalize_strips_boilerplate_unwraps_and_unifies_quotes():
    assert normalize(RAW) == (
        '"You have been in Afghanistan, I perceive," said Holmes.\n\n\'Twas brillig...'
    )


def test_split_book_holds_out_the_tail_on_paragraph_boundaries():
    text = "\n\n".join(f"paragraph {i:02d}" for i in range(20))
    train, val = split_book(text, 0.1)
    assert train + "\n\n" + val == text
    assert val.startswith("paragraph 1") and val.count("paragraph") == 2


@pytest.mark.parametrize("text", ["Hello  world!\n\n'Tis 1895 -- _odd_ café", "  \t x\n"])
def test_pretokenizer_is_lossless(text):
    assert "".join(_CHUNK.findall(text)) == text


def test_merge_replaces_non_overlapping_pairs_left_to_right():
    assert merge([1, 1, 1, 2], (1, 1), 9) == [9, 1, 2]


def test_bpe_learns_frequent_pairs_first():
    merges = train_bpe("aaab " * 50 + "ab", num_merges=3)
    assert merges[0] == (ord("a"), ord("a"))


def test_round_trip_and_special_token():
    text = "Holmes and Watson. " * 30
    merges = train_bpe(text, num_merges=40)
    assert len(merges) < 40  # stops early once no pair occurs twice
    bpe = ByteBPE(merges)
    doc = f"Holmes said{EOT}Watson replied: naïve? 🙂"
    ids = bpe.encode(doc)
    assert bpe.decode(ids) == doc
    assert ids.count(bpe.special[EOT]) == 1
    assert bpe.vocab_size == 256 + len(merges) + 1
    assert len(bpe.encode(" Holmes")) == 1  # frequent word became a single token


def test_save_and_load_give_identical_encoding(tmp_path):
    bpe = ByteBPE(train_bpe("the cat sat on the mat " * 20, num_merges=20))
    bpe.save(tmp_path)
    again = ByteBPE.load(tmp_path)
    assert again.encode("the mat") == bpe.encode("the mat")
    assert re.search(r"^256\t", (tmp_path / "vocab.txt").read_text(), re.M)
