"""Unit and integration tests for exact and semantic caching."""

import httpx
import pytest

from gateway.main import create_app
from gateway.upstream.client import AsyncUpstreamClient
from tests.conftest import MultiAppTransport


@pytest.mark.asyncio
async def test_cache_miss_then_hit(client):
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Deterministic question for cache testing"}],
        "temperature": 0.0,
        "stream": False,
    }

    # 1. First request: Cache Miss
    resp1 = await client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cache") == "MISS"

    # 2. Second request: Cache Hit
    resp2 = await client.post("/v1/chat/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"

    data1 = resp1.json()
    data2 = resp2.json()
    assert data1["choices"][0]["message"]["content"] == data2["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_streaming_cache_hit(client):
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Deterministic streaming cache prompt"}],
        "temperature": 0.0,
        "stream": True,
    }

    # First streaming request populates cache
    resp1 = await client.post("/v1/chat/completions", json=payload)
    assert resp1.status_code == 200
    assert resp1.headers.get("X-Cache") == "MISS"
    _ = resp1.text

    # Second streaming request served from cache instantly
    resp2 = await client.post("/v1/chat/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"
    body2 = resp2.text
    assert "data: " in body2
    assert "[DONE]" in body2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variant",
    [
        {"max_tokens": 2},
        {"stop": [" is"]},
        {"n": 2},
        {"presence_penalty": 1.0},
        {"frequency_penalty": 1.0},
        {"messages": [{"role": "user", "content": "Generation params cache prompt", "name": "alice"}]},
    ],
    ids=["max_tokens", "stop", "n", "presence_penalty", "frequency_penalty", "message_name"],
)
async def test_cache_key_distinguishes_generation_params(client, variant):
    base = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Generation params cache prompt"}],
        "temperature": 0.0,
        "stream": False,
    }

    resp1 = await client.post("/v1/chat/completions", json={**base, **variant})
    assert resp1.status_code == 200

    resp2 = await client.post("/v1/chat/completions", json=base)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") != "HIT", f"request differing in {variant} must not replay the cached response"

    resp3 = await client.post("/v1/chat/completions", json=base)
    assert resp3.headers.get("X-Cache") == "HIT", "an identical request still hits the cache"


@pytest.mark.asyncio
async def test_cache_max_size_zero_serves_requests_without_caching(mock_apps, test_settings):
    primary_app, fallback_app = mock_apps
    upstream = AsyncUpstreamClient(transport=MultiAppTransport(primary_app, fallback_app))
    app = create_app(settings=test_settings.model_copy(update={"cache_max_size": 0}), client=upstream)
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Zero-size cache prompt"}],
        "temperature": 0.0,
        "stream": False,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post("/v1/chat/completions", json=payload)
        assert resp1.status_code == 200, "an uncacheable response is still delivered"
        assert resp1.json()["choices"][0]["message"]["content"]

        resp2 = await client.post("/v1/chat/completions", json=payload)
        assert resp2.status_code == 200
        assert resp2.headers.get("X-Cache") != "HIT", "a zero-size cache stores nothing"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_disabled_cache_sends_no_cache_header(mock_apps, test_settings, stream):
    primary_app, fallback_app = mock_apps
    upstream = AsyncUpstreamClient(transport=MultiAppTransport(primary_app, fallback_app))
    app = create_app(settings=test_settings.model_copy(update={"cache_enabled": False}), client=upstream)
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Disabled cache prompt"}],
        "temperature": 0.0,
        "stream": stream,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200
        assert "X-Cache" not in resp.headers, "a disabled cache reports no cache status"
