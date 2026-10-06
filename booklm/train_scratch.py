"""Route A: pretrain a tiny Llama-architecture model from random weights on the corpus only.

It shares SmolLM2's tokenizer so its predictions line up token-for-token with the other models.
Whatever it knows about English, it learned from ~0.8M tokens of Conan Doyle.
"""

from __future__ import annotations

import argparse
import time

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from booklm.config import BASE_MODEL_ID, get_settings, resolve_device
from booklm.training import (
    ModelCard,
    cosine_lr,
    count_params,
    evaluate,
    get_batch,
    load_split,
)


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="holmes-scratch")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(settings.device)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden,
        intermediate_size=args.hidden * 8 // 3,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.heads,
        max_position_embeddings=1024,
        attention_dropout=0.1,
        tie_word_embeddings=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config).to(device)
    total, trainable = count_params(model)
    print(f"device={device} params={total / 1e6:.1f}M")

    train = load_split(settings.data_dir, "train")
    val = load_split(settings.data_dir, "val")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(args.seed)
    out_dir = settings.models_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    card = ModelCard(
        name=args.name,
        kind="scratch",
        description=(
            f"{total / 1e6:.0f}M-param Llama trained from random init "
            "on the Sherlock Holmes canon only."
        ),
        base_model=None,
        total_params=total,
        trainable_params=trainable,
    )
    best = float("inf")
    start = time.time()
    model.train()
    bar = tqdm(range(args.steps + 1), desc="scratch")
    for step in bar:
        if step % args.eval_every == 0:
            val_loss = evaluate(model, val, args.batch_size, args.ctx, device)
            card.history.append({"step": step, "train_loss": None, "val_loss": val_loss})
            if val_loss < best:  # keep the best checkpoint: one book series overfits quickly
                best = val_loss
                card.val_loss = val_loss
                card.train_tokens_seen = step * args.batch_size * args.ctx
                model.save_pretrained(out_dir)
            bar.set_postfix(val=f"{val_loss:.3f}", best=f"{best:.3f}")
        if step == args.steps:
            break
        for group in opt.param_groups:
            group["lr"] = cosine_lr(step, args.lr, args.lr / 10, 100, args.steps)
        x, y = get_batch(train, args.batch_size, args.ctx, device, gen)
        logits = model(input_ids=x).logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 20 == 0:
            card.history.append({"step": step, "train_loss": loss.item(), "val_loss": None})

    card.elapsed_s = time.time() - start
    tokenizer.save_pretrained(out_dir)
    card.save(out_dir)
    print(f"saved {out_dir} best val_loss={best:.3f} ppl={card.val_ppl:.1f}")


if __name__ == "__main__":
    main()
