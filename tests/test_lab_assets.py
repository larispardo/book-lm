from booklm import lab_assets


def test_pack_then_ensure_round_trip(tmp_path):
    src = tmp_path / "src"
    for rel in [
        "artifacts/data/clean/train.txt",
        "artifacts/tokenizers/holmes-bpe-8192/tokenizer.json",
        "artifacts/models/holmes-gpt-tiny/model.pt",
        "artifacts/data/raw/pg1661.txt",  # must NOT be shipped
    ]:
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(rel)
    archive = tmp_path / "lab.tar.gz"
    files = lab_assets.pack(archive, root=src)
    assert len(files) == 3

    dst = tmp_path / "dst"
    dst.mkdir()
    assert lab_assets.ensure(dst, url=archive.as_uri()) is True
    assert (dst / "artifacts/models/holmes-gpt-tiny/model.pt").read_text().endswith("model.pt")
    assert not (dst / "artifacts/data/raw").exists()
    assert lab_assets.ensure(dst, url="http://unused.invalid") is False  # already present
