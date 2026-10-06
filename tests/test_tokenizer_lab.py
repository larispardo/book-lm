import pytest

from booklm import tokenizer_lab as lab
from booklm.tokenize_own import ByteBPE, train_bpe

TEXT = "the cat sat on the mat with the hat " * 20


@pytest.fixture(scope="module")
def bpe():
    return ByteBPE(train_bpe(TEXT, num_merges=30))


def test_train_with_counts_matches_train_bpe_and_counts_decrease():
    run = lab.train_with_counts(TEXT, 30)
    assert run.merges == train_bpe(TEXT, 30)
    assert run.counts == sorted(run.counts, reverse=True)


def test_candidates_include_the_pair_that_gets_chosen():
    run = lab.train_with_counts(TEXT, 5)
    for step in range(5):
        pairs = [pair for pair, _ in lab.candidates_at_step(TEXT, step)]
        assert run.merges[step] in pairs


def test_tokens_at_step_zero_are_bytes_and_shrink_with_merges(bpe):
    assert len(lab.tokens_at_step(bpe.merges, 0, " the hat")) == len(" the hat")
    assert len(lab.tokens_at_step(bpe.merges, 30, " the hat")) < len(" the hat")


def test_merge_trace_ends_at_the_encoding(bpe):
    trace = lab.merge_trace(bpe, " the")
    assert trace[0]["tokens"] == [" ", "t", "h", "e"]
    assert trace[-1]["tokens"] == [bpe.decode([i]) for i in bpe.encode(" the")]


def test_chips_escape_markdown_and_show_spaces():
    md = lab.chips_markdown([" said", "[x]*"])
    assert md == ":blue-background[·said] :orange-background[\\[x\\]\\*]"


def test_variant_stats(bpe):
    s = lab.variant_stats(bpe, TEXT, "the cat")
    assert s.vocab_size == bpe.vocab_size
    assert s.embedding_params == bpe.vocab_size * lab.D_MODEL
    assert s.train_bytes_per_token > 1


# Exercise: implement lab.merge_tree, then delete this marker.
def test_merge_tree(bpe):
    the = bpe.encode(" the")[0]
    tree = lab.merge_tree(bpe, the)
    assert tree["token"] == " the"

    def leaves(node):
        return [node["token"]] if not node["children"] else sum(map(leaves, node["children"]), [])

    assert leaves(tree) == [" ", "t", "h", "e"]
    assert lab.merge_tree(bpe, 65) == {"id": 65, "token": "A", "children": []}


def test_smaller_vocab_is_a_prefix_of_a_bigger_one():
    """BPE is greedy and deterministic: train once, cut anywhere."""
    assert train_bpe(TEXT, 10) == train_bpe(TEXT, 30)[:10]


def test_random_paragraph_respects_length_bounds():
    text = "\n\n".join(["short", "x" * 200, "y" * 900])
    assert lab.random_paragraph(text) == "x" * 200


def test_words_by_frequency_spans_the_bands():
    counts = lab.word_counts(" common" * 50 + " middling" * 10 + " rarity")
    picked = lab.words_by_frequency(counts, bands=(50, 10, 1), per_band=1, min_len=6)
    assert picked == [(" common", 50), (" middling", 10), (" rarity", 1)]
