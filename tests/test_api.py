import json

import pytest
from fastapi.testclient import TestClient

from booklm.server.app import create_app

SAMPLING = {"temperature": 0.9, "top_k": 20, "seed": 3}


@pytest.fixture(scope="module")
def client(settings, registry):
    with TestClient(create_app(settings, registry)) as c:
        yield c


def test_health_and_readiness(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["models"] == ["tiny"]


def test_models_and_metrics(client):
    assert [m["name"] for m in client.get("/api/models").json()] == ["tiny"]
    assert "booklm_requests_total" in client.get("/metrics").text


def test_tokenize_round_trips(client):
    body = client.post("/api/tokenize", json={"model": "tiny", "text": "Elementary, Watson"}).json()
    assert "".join(body["tokens"]) == "Elementary, Watson"


def test_step_reports_full_distribution(client):
    body = client.post(
        "/api/step", json={"model": "tiny", "prompt": "Holmes", "sampling": SAMPLING, "top_n": 5}
    ).json()
    step = body["step"]
    assert body["prompt_tokens"] >= 1
    assert step["kept_count"] == 20
    assert 5 <= len(step["candidates"]) <= 6
    assert step["token_id"] in [c["id"] for c in step["candidates"]]


def test_generate_is_deterministic_with_seed(client):
    req = {"model": "tiny", "prompt": "Holmes", "sampling": SAMPLING, "max_new_tokens": 8}
    a = client.post("/api/generate", json=req).json()
    b = client.post("/api/generate", json=req).json()
    assert a["text"] == b["text"]
    assert len(a["steps"]) == 8


def test_prefix_ids_force_a_branch(client):
    req = {"model": "tiny", "prompt": "Holmes", "sampling": SAMPLING, "max_new_tokens": 3}
    first = client.post("/api/generate", json=req).json()["steps"]
    branch = client.post("/api/step", json={**req, "prefix_ids": [first[0]["token_id"]]}).json()
    assert branch["prompt_tokens"] == client.post("/api/step", json=req).json()["prompt_tokens"] + 1


def test_generate_streams_server_sent_events(client):
    req = {"model": "tiny", "prompt": "Holmes", "sampling": SAMPLING, "max_new_tokens": 4}
    with client.stream("POST", "/api/generate", json={**req, "stream": True}) as resp:
        events = [line for line in resp.iter_lines() if line.startswith("event:")]
        assert resp.headers["content-type"].startswith("text/event-stream")
    assert events == ["event: token"] * 4 + ["event: done"]


def test_stream_releases_model_lock(client, registry):
    req = {"model": "tiny", "prompt": "Holmes", "max_new_tokens": 2, "stream": True}
    with client.stream("POST", "/api/generate", json=req) as resp:
        list(resp.iter_lines())
    assert not registry.get("tiny").lock.locked()


@pytest.mark.parametrize(
    "patch, status",
    [
        ({"model": "nope"}, 404),
        ({"sampling": {"top_p": 0}}, 422),
        ({"sampling": {"temperature": 9}}, 422),
        ({"max_new_tokens": 999}, 422),
        ({"top_n": 999}, 422),
        ({"prefix_ids": [10**9]}, 422),
        ({"prompt": "word " * 2000}, 413),
    ],
)
def test_rejects_bad_requests(client, patch, status):
    req = {"model": "tiny", "prompt": "Holmes", "max_new_tokens": 2} | patch
    resp = client.post("/api/generate", json=req)
    assert resp.status_code == status, json.dumps(resp.json())
