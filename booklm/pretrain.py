"""Stage 2: pretrain our GPT from random weights on the Holmes tokens.

    python -m booklm.pretrain --preset tiny --steps 300      # smoke test, ~1 min
    python -m booklm.pretrain --preset small                 # the real run

Keeps the Mac awake on its own (caffeinate) while training. Writes, per run:
  artifacts/models/<name>/model.pt     best checkpoint by validation loss (+ config)
  artifacts/models/<name>/log.jsonl    one line per log/eval step: losses, lr, samples
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from booklm.batches import get_batch, load_tokens
from booklm.model import GPT, PRESETS
from booklm.tokenize_own import ByteBPE

TOKENIZER = "holmes-bpe-8192"
SAMPLE_PROMPT = "Holmes"


# ── YOUR EXERCISE ───────────────────────────────────────────────────────────────────────


def train_step(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip: float = 1.0,
) -> float:
    """One step of learning on one batch. Returns the loss as a Python float.

    1. logits = model(x)                                  (B, T, vocab)
    2. loss   = cross-entropy between logits and targets y (B, T)
              F.cross_entropy wants (N, vocab) and (N,): flatten B and T together
    3. clear old gradients, backpropagate the loss
    4. clip the gradient norm to `grad_clip` (torch.nn.utils.clip_grad_norm_)
    5. let the optimizer update the weights
    """
    # Step 1: forward pass
    logits = model(x)  # (B, T, vocab)

    # Step 2: compute loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    # Step 3: clear old gradients and backpropagate the loss
    optimizer.zero_grad()
    loss.backward()

    # Step 4: clip the gradient norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Step 5: update the weights
    optimizer.step()

    return loss.item()


# ── provided ───────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model: GPT, data, batch_size: int, batches: int, device) -> float:
    """Mean loss over a fixed set of windows, so evaluations are comparable over time."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    total = 0.0
    for _ in range(batches):
        x, y = get_batch(data, batch_size, model.cfg.context, device, gen)
        logits = model(x)
        total += F.cross_entropy(logits.flatten(0, 1), y.flatten()).item()
    model.train()
    return total / batches


def lr_at(step: int, max_lr: float, warmup: int, total: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def keep_awake() -> None:
    """On macOS, stop the machine sleeping until this process exits."""
    if shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def main() -> None:
    p = argparse.ArgumentParser(description="Pretrain the GPT on Holmes tokens.")
    p.add_argument("--preset", choices=PRESETS, default="small")
    p.add_argument("--name", help="run name (default: holmes-gpt-<preset>)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = p.parse_args()

    keep_awake()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    cfg = PRESETS[args.preset]
    tokens = args.artifacts / "data" / "tokens" / TOKENIZER
    train, val = load_tokens(tokens / "train.bin"), load_tokens(tokens / "val.bin")
    tokenizer = ByteBPE.load(args.artifacts / "tokenizers" / TOKENIZER)
    assert tokenizer.vocab_size == cfg.vocab_size

    model = GPT(cfg).to(device)
    # Weight decay on matrices only; biases and LayerNorm gains are left alone.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr,
        betas=(0.9, 0.95),
    )

    out = args.artifacts / "models" / (args.name or f"holmes-gpt-{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")
    tokens_per_step = args.batch_size * cfg.context
    header = {
        "run": out.name,
        "preset": args.preset,
        "config": cfg.to_dict(),
        "params": model.num_params(),
        "batch_size": args.batch_size,
        "tokens_per_step": tokens_per_step,
        "train_tokens": len(train),
        "steps": args.steps,
        "lr": args.lr,
        "device": str(device),
    }
    log.write(json.dumps({"header": header}) + "\n")
    print(
        f"{out.name}: {model.num_params() / 1e6:.2f}M params on {device}, "
        f"{len(train):,} train tokens, {tokens_per_step:,} tokens/step, "
        f"{args.steps * tokens_per_step / len(train):.1f} epochs"
    )

    gen = torch.Generator().manual_seed(args.seed)
    prompt = torch.tensor([tokenizer.encode(SAMPLE_PROMPT)], device=device)
    best, start = float("inf"), time.time()
    for step in range(args.steps + 1):
        lr = lr_at(step, args.lr, args.warmup, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val, args.batch_size, 20, device)
            sample_ids = model.generate(prompt, 40, temperature=0.8, top_k=50, generator=gen)
            sample = tokenizer.decode(sample_ids[0].tolist())
            record = {"step": step, "val_loss": val_loss, "sample": sample}
            if val_loss < best:
                best = val_loss
                torch.save(
                    {"config": cfg.to_dict(), "model": model.state_dict(), "step": step},
                    out / "model.pt",
                )
                record["best"] = True
            log.write(json.dumps(record) + "\n")
            log.flush()
            mark = "*" if record.get("best") else " "
            print(f"step {step:>5}  val {val_loss:.3f} {mark}  {sample!r}")
            if step == args.steps:
                break

        x, y = get_batch(train, args.batch_size, cfg.context, device, gen)
        loss = train_step(model, optimizer, x, y)
        if step % args.log_every == 0:
            elapsed = time.time() - start
            log.write(
                json.dumps({"step": step, "train_loss": loss, "lr": lr, "elapsed": elapsed}) + "\n"
            )

    log.close()
    print(
        f"best val loss {best:.3f} (perplexity {math.exp(best):.1f}) in {time.time() - start:.0f}s"
    )


if __name__ == "__main__":
    main()
