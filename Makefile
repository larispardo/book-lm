PY := .venv/bin/python
QUALITY ?= l  # manim quality: l (480p15), m (720p30), h (1080p60)

.PHONY: setup data train-scratch train-lora train serve test lint docker-build docker-up video clean

setup:            ## create venv and install everything (manim needs: brew install cairo pango)
	uv venv -p 3.11 .venv
	uv pip install -p $(PY) -e ".[dev,video]"

data:             ## download + tokenize the Sherlock Holmes canon
	$(PY) -m booklm.data

train-scratch:    ## route A: tiny Llama from random init (~15 min on Apple M-series)
	$(PY) -m booklm.train_scratch

train-lora:       ## route B: LoRA on SmolLM2-135M, also registers the untouched base
	$(PY) -m booklm.train_lora

train: data train-scratch train-lora

serve:            ## run API + UI on http://localhost:8000
	$(PY) -m booklm.server.app

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check booklm tests video
	.venv/bin/ruff format --check booklm tests video

docker-build:
	docker compose build

docker-up:        ## serve trained models from ./artifacts in a container
	docker compose up -d && docker compose ps

video:            ## export real model traces, then render the manim showcase
	$(PY) video/export_traces.py
	.venv/bin/manim -q$(strip $(QUALITY)) video/showcase.py Showcase

clean:
	rm -rf media .pytest_cache .ruff_cache
