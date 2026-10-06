import json

import pytest
import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from booklm.config import BASE_MODEL_ID, Settings
from booklm.engine import Registry


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """A registry with one tiny random model; behaviour, not quality, is under test."""
    root = tmp_path_factory.mktemp("artifacts")
    model_dir = root / "models" / "tiny"
    model_dir.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    LlamaForCausalLM(config).save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    card = {"name": "tiny", "kind": "scratch", "description": "test", "total_params": 1}
    (model_dir / "card.json").write_text(json.dumps(card))
    return root


@pytest.fixture(scope="session")
def settings(artifacts):
    return Settings(artifacts_dir=artifacts, device="cpu", max_new_tokens=20)


@pytest.fixture(scope="session")
def registry(settings):
    reg = Registry(settings.models_dir, torch.device("cpu"))
    reg.load()
    return reg
