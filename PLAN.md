# book-lm plan

Build a small language model on the Sherlock Holmes canon, one stage at a time, while reading
Chip Huyen's *AI Engineering*. Each stage adds code to the `booklm/` package **and** a panel to
the lab UI, so every concept can be seen and poked at, not just run.

## How we work

- Guided pairing: decisions and hands-on parts are yours; Claude explains, scaffolds, reviews.
- Logic lives in `booklm/` (tested); the lab UI (`streamlit_app.py`, `app_pages/`) only calls it.
- One branch and a few small commits per stage.
- The lab UI is an instrument, not a product: clarity over polish.
- Infrastructure and hosting come last, once the pipeline exists.
- `solutions` branch: a complete reference implementation to peek at when stuck.

## Roadmap

| # | Stage | Code | Lab panel | Book | Status |
|---|---|---|---|---|---|
| 0 | Data | `download.py` | — | 8 | done |
| 1 | Tokenization | `tokenize_own.py`, `tokenize_smollm2.py` | Our BPE (step by step, vocab size, browser), SmolLM2, side by side | 2 | done |
| 2 | Tiny GPT from scratch | `model.py`, `batches.py`, `pretrain.py` on `holmes-bpe-8192` | loss curves, samples over training, attention maps | 2 | in progress |
| 3 | Decoding | logits, temperature, top-k/p, min-p, KV cache | live distribution + sampling knobs | 2 | |
| 4 | Finetuning | LoRA by hand, then `peft`, on SmolLM2-135M | scratch vs base vs finetuned | 7 | |
| 5 | Evaluation | perplexity, bits per byte, LLM-as-judge | eval dashboard | 3–4 | |
| 6 | Preference tuning | DPO pairs + training | before/after | 2, 7 | |
| 7 | Serving and hosting | FastAPI, streaming, Docker, metrics, HF Spaces | — | 9–10 | |

## Experiments backlog

- Blocked validation split: random ~2k-char chunks with a purge gap, instead of each book's tail.
- Case markers (`<cap>` + lowercase) vs plain case: does pooling variants help a tiny model?
- Vocab size sweep (1k–32k): compression vs embedding cost vs model loss.
- Show partial-UTF-8 vocab entries with `errors="backslashreplace"` in `vocab.txt`.
- Unigram LM tokenizer (SentencePiece) vs BPE on the same split.

## Decisions so far

- Corpus: the nine Holmes books, Project Gutenberg, public domain.
- Normalization: NFKC, curly quotes → straight, `_italics_` markers removed, lines unwrapped.
- Split: last 5% of each book (by characters) is validation; the tokenizer trains on train only.
- Own tokenizer: byte-level BPE, 8,192 ids (256 bytes + 7,935 merges + `<|endoftext|>`).
- Finetuning route uses SmolLM2's tokenizer (49,152 ids), on the same cleaned splits.
- Pretraining: hand-written GPT (pre-norm blocks, learned positions, tied embeddings), AdamW,
  warmup + cosine LR, best checkpoint by val loss, MPS with caffeinate. Presets: tiny 1.5M,
  small 5.3M, base 13.9M params.
- Hand-written core pieces so far: `merge_tree` (stage 1); `causal_attention`, `train_step`
  (stage 2).
