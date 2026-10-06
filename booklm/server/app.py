"""HTTP API and static UI for the glass-box explorer.

Endpoints
  GET  /healthz          liveness
  GET  /readyz           readiness (503 until models are loaded)
  GET  /metrics          Prometheus metrics
  GET  /api/models       model cards
  POST /api/tokenize     how a model's tokenizer splits text
  POST /api/step         next-token distribution after a prompt (nothing committed)
  POST /api/generate     generate tokens; JSON, or Server-Sent Events when stream=true
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from booklm.config import Settings, get_settings, resolve_device
from booklm.engine import LoadedModel, Registry, generate, next_distribution
from booklm.sampling import SamplingParams

log = logging.getLogger("booklm")
STATIC_DIR = Path(__file__).parent / "static"

REQUESTS = Counter("booklm_requests_total", "API requests", ["endpoint", "model", "status"])
TOKENS = Counter("booklm_generated_tokens_total", "Generated tokens", ["model"])
LATENCY = Histogram("booklm_request_seconds", "Request latency", ["endpoint"])


class Sampling(BaseModel):
    temperature: float = Field(1.0, ge=0.0, le=5.0, description="0 = greedy")
    top_k: int = Field(0, ge=0, le=50_000, description="0 = off")
    top_p: float = Field(1.0, gt=0.0, le=1.0, description="1 = off")
    min_p: float = Field(0.0, ge=0.0, lt=1.0, description="0 = off")
    repetition_penalty: float = Field(1.0, ge=1.0, le=3.0, description="1 = off")
    seed: int | None = Field(None, ge=0, le=2**31 - 1)

    def to_params(self) -> SamplingParams:
        return SamplingParams(**self.model_dump())


class TokenizeRequest(BaseModel):
    model: str
    text: str = Field(..., max_length=20_000)


class StepRequest(BaseModel):
    model: str
    prompt: str = Field("", max_length=20_000)
    # Exact token ids appended after the tokenised prompt; used to branch from a past step
    # without re-tokenising text (which can split differently).
    prefix_ids: list[int] = Field(default_factory=list, max_length=2_000)
    sampling: Sampling = Field(default_factory=Sampling)
    top_n: int = Field(10, ge=1)


class GenerateRequest(StepRequest):
    max_new_tokens: int = Field(60, ge=1)
    stream: bool = False


def create_app(settings: Settings | None = None, registry: Registry | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if registry is None:
            reg = Registry(settings.models_dir, resolve_device(settings.device))
            reg.load(settings.model_names())
            app.state.registry = reg
        else:
            app.state.registry = registry
        if not app.state.registry.models:
            log.warning("no models found in %s; run training first", settings.models_dir)
        yield

    app = FastAPI(title="BookLM glass-box explorer", version="0.1.0", lifespan=lifespan)

    def get_model(request: Request, name: str) -> LoadedModel:
        try:
            return request.app.state.registry.get(name)
        except KeyError:
            raise HTTPException(404, f"unknown model {name!r}") from None

    def prompt_ids(lm: LoadedModel, req: StepRequest) -> list[int]:
        vocab = len(lm.tokenizer)
        if any(not 0 <= i < vocab for i in req.prefix_ids):
            raise HTTPException(422, "prefix_ids contains ids outside the vocabulary")
        ids = lm.encode(req.prompt) + req.prefix_ids
        if len(ids) > settings.max_prompt_tokens:
            limit = settings.max_prompt_tokens
            raise HTTPException(413, f"prompt is {len(ids)} tokens; max {limit}")
        return ids

    def check_limits(req: StepRequest) -> None:
        if req.top_n > settings.max_top_n:
            raise HTTPException(422, f"top_n must be <= {settings.max_top_n}")
        if isinstance(req, GenerateRequest) and req.max_new_tokens > settings.max_new_tokens:
            raise HTTPException(422, f"max_new_tokens must be <= {settings.max_new_tokens}")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            LATENCY.labels(request.url.path).observe(time.perf_counter() - start)
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(request: Request):
        models = list(request.app.state.registry.models)
        status = 200 if models else 503
        return JSONResponse({"ready": bool(models), "models": models}, status_code=status)

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/models")
    def list_models(request: Request):
        return [lm.card for lm in request.app.state.registry.models.values()]

    @app.post("/api/tokenize")
    def tokenize(req: TokenizeRequest, request: Request):
        lm = get_model(request, req.model)
        ids = lm.encode(req.text)
        REQUESTS.labels("tokenize", req.model, "200").inc()
        return {"ids": ids, "tokens": [lm.token_str(i) for i in ids]}

    @app.post("/api/step")
    def step(req: StepRequest, request: Request):
        lm = get_model(request, req.model)
        check_limits(req)
        ids = prompt_ids(lm, req)
        result = next_distribution(lm, ids, req.sampling.to_params(), req.top_n)
        REQUESTS.labels("step", req.model, "200").inc()
        return {"prompt_tokens": len(ids), "step": asdict(result)}

    @app.post("/api/generate")
    def generate_route(req: GenerateRequest, request: Request):
        lm = get_model(request, req.model)
        check_limits(req)
        ids = prompt_ids(lm, req)
        params = req.sampling.to_params()
        steps = generate(lm, ids, params, req.max_new_tokens, req.top_n)
        REQUESTS.labels("generate", req.model, "200").inc()

        if not req.stream:
            start = time.perf_counter()
            out = [asdict(s) for s in steps]
            TOKENS.labels(req.model).inc(len(out))
            return {
                "model": req.model,
                "prompt_tokens": len(ids),
                "text": "".join(s["token"] for s in out),
                "steps": out,
                "elapsed_s": time.perf_counter() - start,
            }

        def events():
            start, count = time.perf_counter(), 0
            try:
                for s in steps:
                    count += 1
                    yield f"event: token\ndata: {json.dumps(asdict(s))}\n\n"
                elapsed = time.perf_counter() - start
                done = {"tokens": count, "elapsed_s": elapsed, "prompt_tokens": len(ids)}
                yield f"event: done\ndata: {json.dumps(done)}\n\n"
            finally:
                steps.close()  # releases the model lock if the client disconnects mid-stream
                TOKENS.labels(req.model).inc(count)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
    return app


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
