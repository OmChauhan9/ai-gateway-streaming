"""Unit and integration tests for exact and semantic caching."""

import pytest


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
    assert resp1.headers.get("X-Cache") is None

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
    _ = resp1.text

    # Second streaming request served from cache instantly
    resp2 = await client.post("/v1/chat/completions", json=payload)
    assert resp2.status_code == 200
    assert resp2.headers.get("X-Cache") == "HIT"
    body2 = resp2.text
    assert "data: " in body2
    assert "[DONE]" in body2
