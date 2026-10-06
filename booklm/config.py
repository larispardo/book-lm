"""Runtime configuration, read from environment variables prefixed with BOOKLM_."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

# Every model shares one tokenizer so token-level comparisons line up across models.
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-135M"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOOKLM_", env_file=".env", extra="ignore")

    artifacts_dir: Path = Path("artifacts")
    # Comma-separated model names to load at startup; empty means every model in the registry.
    models: str = ""
    device: str = "auto"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_prompt_tokens: int = 512
    max_new_tokens: int = 200
    max_top_n: int = 50

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def data_dir(self) -> Path:
        return self.artifacts_dir / "data"

    def model_names(self) -> list[str]:
        return [m.strip() for m in self.models.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
