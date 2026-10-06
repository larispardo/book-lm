# BookLM: a glass-box language model

Train small language models on one fictional universe (the Sherlock Holmes canon) and explore
**every stage of decoding** in the browser: raw logits, probabilities, temperature, top-k, top-p,
min-p, repetition penalty, branching, and model-vs-model comparison. Includes a Manim video built
from real model outputs.

Everything is open: MIT code, Apache-2.0 base model ([SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)),
public-domain corpus ([Project Gutenberg](https://www.gutenberg.org)).

## The three models

All three share SmolLM2's tokenizer, so their predictions line up token for token.

| model | how it was made | what it shows |
|---|---|---|
| `holmes-scratch` | tiny Llama, random init, trained on the books only | what pretraining on ~0.8M tokens can learn |
| `smollm2-base` | SmolLM2-135M as released (~2T tokens of web, code, books) | general English, no Holmes voice |
| `holmes-lora` | SmolLM2-135M + LoRA on the books, adapters merged | pretrained knowledge steered toward Conan Doyle |

Results on held-out Holmes text are in [Results](#results).

## Quickstart

```bash
brew install cairo pango        # only needed for the Manim video
make setup                      # venv + deps (uv)
make train                      # download corpus, train both routes (~25 min on Apple M-series)
make serve                      # http://localhost:8000
make video QUALITY=h            # media/videos/showcase/1080p60/Showcase.mp4
```

Container (serves models already trained into `./artifacts/models`):

```bash
make docker-build && make docker-up             # http://localhost:8000
docker compose --profile obs up -d              # + Prometheus on :9090
```

## Architecture

```
booklm/
  data.py           Gutenberg download, header/footer stripping, tokenisation, train/val split
  train_scratch.py  route A: LlamaForCausalLM from random weights
  train_lora.py     route B: LoRA on SmolLM2-135M, merge, also registers the untouched base
  training.py       batching, eval, cosine LR, model cards (card.json next to each model)
  sampling.py       decoding as explicit stages: penalty -> temperature -> top-k/top-p/min-p -> sample
  engine.py         model registry, KV-cached step-by-step generation reporting each distribution
  server/app.py     FastAPI: API, SSE streaming, health, metrics; serves the UI
  server/static/    vanilla JS UI, no build step, no CDN
video/
  export_traces.py  runs the models once, dumps traces + full-vocab logits
  showcase.py       Manim scene animating those traces
```

Model registry = `artifacts/models/<name>/` with HF weights, tokenizer and `card.json`
(params, tokens seen, validation loss, training history). The server loads every directory
with a `card.json`, or the subset in `BOOKLM_MODELS`.

## API

| method | path | purpose |
|---|---|---|
| GET | `/healthz` | liveness |
| GET | `/readyz` | readiness, 503 until models are loaded |
| GET | `/metrics` | Prometheus: request counts, latency, generated tokens |
| GET | `/api/models` | model cards |
| POST | `/api/tokenize` | how the tokenizer splits text |
| POST | `/api/step` | next-token distribution after a prompt, nothing committed |
| POST | `/api/generate` | generate; JSON, or Server-Sent Events with `"stream": true` |

```bash
curl -s localhost:8000/api/step -H 'content-type: application/json' -d '{
  "model": "holmes-lora", "prompt": "Holmes looked at me and said",
  "sampling": {"temperature": 0.8, "top_p": 0.95, "seed": 1}, "top_n": 5}' | jq .step
```

Each step returns, for the top-n candidates: `logit`, `raw_prob` (model belief),
`scaled_prob` (after penalty and temperature), `final_prob` (after truncation, what is sampled
from) and `kept`; plus entropy, the chosen token's rank, and how many tokens survived truncation.
`prefix_ids` appends exact token ids after the prompt, which is how the UI branches from any
past step without re-tokenising.

## Production-ish choices

- **Bounded inputs**: prompt length, `max_new_tokens`, `top_n` and every sampling knob validated (413/422).
- **Probes**: `/healthz` for liveness, `/readyz` only after models load; Docker `HEALTHCHECK` uses it.
- **Observability**: Prometheus metrics, optional Prometheus service via compose profile.
- **Container**: CPU-only torch, non-root user, models mounted read-only, `HF_HUB_OFFLINE=1`
  (each model directory carries its tokenizer, so no network at runtime).
- **Concurrency**: one lock per model; different models generate in parallel. A client
  disconnecting mid-stream releases the lock.
- **Reproducibility**: sampling runs on CPU with a seeded generator, so a seed gives the same
  text on CPU, MPS or CUDA.
- **CI**: `.github/workflows/ci.yml` lints, tests against a tiny random model, builds the image.
  It runs when `book-lm/` is the repository root.

### Why not vLLM / llama.cpp / Ollama?

They are the right choice for serving throughput, and these models are plain HF Llama
checkpoints, so they load there unchanged. But they expose at most the top-N log-probs of the
final distribution. A glass box needs the raw logits and every intermediate stage before
sampling, so this project runs its own small decoding loop.

## Results

_Filled in from `card.json` after training._

## Limitations

- ~0.87M tokens is tiny. The scratch model learns names, punctuation and rhythm, not plot.
- Validation text is the last 5% of the corpus (the end of *The Case-Book*), so it is a fair
  held-out sample, but from the same author.
- CPU inference in Docker is fine for a demo, not for many concurrent users.
