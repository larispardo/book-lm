"""Route B: LoRA-finetune pretrained SmolLM2-135M on the corpus, then merge the adapters.

Also registers the untouched base model, so the explorer can show all three side by side:
  holmes-scratch  -> learned English from the books alone
  smollm2-base    -> general English, knows nothing special about Holmes' voice
  holmes-lora     -> general English, steered toward Conan Doyle
"""

from __future__ import annotations

import argparse
import time

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from booklm.config import BASE_MODEL_ID, get_settings, resolve_device
from booklm.training import (
    ModelCard,
    cosine_lr,
    count_params,
    evaluate,
    get_batch,
    load_split,
)


def register_base(device, val, batch_size: int, ctx: int) -> None:
    """Save the pretrained model unchanged, with its validation loss on our corpus as a baseline."""
    settings = get_settings()
    out_dir = settings.models_dir / "smollm2-base"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID).to(device)
    total, _ = count_params(model)
    card = ModelCard(
        name="smollm2-base",
        kind="pretrained",
        description="SmolLM2-135M as released: pretrained on ~2T tokens of web, code and books.",
        base_model=BASE_MODEL_ID,
        total_params=total,
        trainable_params=0,
        val_loss=evaluate(model, val, batch_size, ctx, device),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(BASE_MODEL_ID).save_pretrained(out_dir)
    card.save(out_dir)
    print(f"saved {out_dir} val_loss={card.val_loss:.3f} ppl={card.val_ppl:.1f}")


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="holmes-lora")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(settings.device)
    train = load_split(settings.data_dir, "train")
    val = load_split(settings.data_dir, "val")
    register_base(device, val, args.batch_size, args.ctx)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID).to(device)
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora)
    total, trainable = count_params(model)
    print(f"device={device} trainable={trainable / 1e6:.2f}M / {total / 1e6:.1f}M")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    gen = torch.Generator().manual_seed(args.seed)
    card = ModelCard(
        name=args.name,
        kind="finetuned",
        description=(
            f"SmolLM2-135M + LoRA (r={args.rank}) on the Sherlock Holmes canon, adapters merged. "
            f"Trained {trainable / 1e6:.1f}M of {total / 1e6:.0f}M params."
        ),
        base_model=BASE_MODEL_ID,
        total_params=total,
        trainable_params=trainable,
    )
    best_state, best = None, float("inf")
    start = time.time()
    model.train()
    bar = tqdm(range(args.steps + 1), desc="lora")
    for step in bar:
        if step % args.eval_every == 0:
            val_loss = evaluate(model, val, args.batch_size, args.ctx, device)
            card.history.append({"step": step, "train_loss": None, "val_loss": val_loss})
            if val_loss < best:
                best = val_loss
                card.val_loss = val_loss
                card.train_tokens_seen = step * args.batch_size * args.ctx
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k
                }
            bar.set_postfix(val=f"{val_loss:.3f}", best=f"{best:.3f}")
        if step == args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, args.lr, args.lr / 10, 30, args.steps)
        x, y = get_batch(train, args.batch_size, args.ctx, device, gen)
        logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            card.history.append({"step": step, "train_loss": loss.item(), "val_loss": None})

    model.load_state_dict(best_state, strict=False)
    # Merging folds B·A into W: the result is a plain model, same size and speed as the base.
    merged = model.merge_and_unload()
    out_dir = settings.models_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    card.elapsed_s = time.time() - start
    card.save(out_dir)
    print(f"saved {out_dir} best val_loss={best:.3f} ppl={card.val_ppl:.1f}")


if __name__ == "__main__":
    main()
