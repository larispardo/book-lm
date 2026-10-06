"""Feed token ids to the model as (input, target) windows.

A training example is any `context`-long slice of the token stream. The target is the same
slice shifted one token to the right, so at every position the model learns to predict the
next token:

    tokens:  [ Holmes  looked  at  me  and ]
    input:   [ Holmes  looked  at  me ]
    target:  [ looked  at      me  and ]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_tokens(path: Path) -> np.ndarray:
    """Memory-map a uint16 .bin file: slices are read from disk on demand."""
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    context: int,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random windows: x is (batch, context) token ids, y is x shifted by one."""
    starts = torch.randint(len(data) - context - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[s : s + context].astype(np.int64)) for s in starts])
    y = torch.stack(
        [torch.from_numpy(data[s + 1 : s + 1 + context].astype(np.int64)) for s in starts]
    )
    return x.to(device), y.to(device)
