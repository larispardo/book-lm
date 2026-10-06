import json

import torch

from booklm import pretrain_lab as lab
from booklm.model import GPT, GPTConfig

CFG = GPTConfig(vocab_size=50, context=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0)


def make_run(tmp_path, header=True):
    run = tmp_path / "holmes-gpt-test"
    run.mkdir()
    lines = [
        {"step": 0, "val_loss": 3.9, "sample": "a", "best": True},
        {"step": 0, "train_loss": 3.9, "lr": 1e-4, "elapsed": 0.1},
        {"step": 10, "val_loss": 3.0, "sample": "b", "best": True},
        {"step": 20, "val_loss": 3.2, "sample": "c"},
    ]
    if header:
        lines.insert(0, {"header": {"tokens_per_step": 100, "train_tokens": 1000}})
    (run / "log.jsonl").write_text("\n".join(json.dumps(r) for r in lines))
    model = GPT(CFG)
    torch.save({"config": CFG.to_dict(), "model": model.state_dict(), "step": 10}, run / "model.pt")
    return run


def test_load_run_splits_records_and_finds_best(tmp_path):
    run = lab.load_run(make_run(tmp_path))
    assert len(run.train) == 1 and len(run.evals) == 3
    assert run.best.step == 10
    assert list(run.epochs(run.evals["step"])) == [0.0, 1.0, 2.0]
    assert lab.list_runs(tmp_path) == ["holmes-gpt-test"]


def test_attention_maps_shape_and_causality(tmp_path):
    model, step = lab.load_model(make_run(tmp_path))
    maps = lab.attention_maps(model, [1, 2, 3, 4, 5])
    assert step == 10
    assert maps.shape == (CFG.n_layers, CFG.n_heads, 5, 5)
    assert torch.all(maps.triu(diagonal=1) == 0)
