"""Unit and integration tests for exact and semantic caching."""

import json

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
        {"seed": 7},
        {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]},
        {"response_format": {"type": "json_object"}},
    ],
    ids=["max_tokens", "stop", "n", "presence_penalty", "frequency_penalty", "message_name", "seed", "tools", "response_format"],
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


SHAPED_REPLY = {
    "id": "x",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
            },
            "logprobs": {"content": [{"token": "Let", "logprob": -0.1, "top_logprobs": []}]},
            "finish_reason": "tool_calls",
        },
        {"index": 1, "message": {"role": "assistant", "content": "Second choice."}, "finish_reason": "stop"},
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}
SHAPED_STREAM = (
    b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
    b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n'
    b"data: [DONE]\n\n"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stream, fields",
    [
        (False, {"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]}),
        (False, {"logprobs": True}),
        (False, {"n": 2}),
        (True, {"stream_options": {"include_usage": True}}),
    ],
    ids=["tools", "logprobs", "n", "stream_options"],
)
async def test_response_the_cache_cannot_replay_comes_from_the_upstream_every_time(test_settings, stream, fields):
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if stream:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=SHAPED_STREAM)
        return httpx.Response(200, json=SHAPED_REPLY)

    app = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "weather?"}], "stream": stream, **fields}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/v1/chat/completions", json=payload)
        second = await client.post("/v1/chat/completions", json=payload)

    assert first.status_code == second.status_code == 200, second.text
    assert second.headers.get("X-Cache") != "HIT", f"a text-only replay would drop what {sorted(fields)} shaped"
    assert second.text == first.text, "the repeat gets the upstream's full response"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cache_key_distinguishes_tool_calls(client):
    def tool_loop_turn(city: str) -> dict:
        return {
            "model": "llama3-8b",
            "messages": [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "w", "arguments": f"{{\"city\": \"{city}\"}}"}}],
                },
                {"role": "tool", "content": "18C", "tool_call_id": "call_1"},
            ],
            "stream": False,
        }

    resp1 = await client.post("/v1/chat/completions", json=tool_loop_turn("Paris"))
    assert resp1.status_code == 200, resp1.text

    resp2 = await client.post("/v1/chat/completions", json=tool_loop_turn("Rome"))
    assert resp2.status_code == 200, resp2.text
    assert resp2.headers.get("X-Cache") != "HIT", "a different tool call must not replay the cached response"

    resp3 = await client.post("/v1/chat/completions", json=tool_loop_turn("Rome"))
    assert resp3.headers.get("X-Cache") == "HIT", "an identical tool-loop turn still hits the cache"


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


def _finish_reasons(resp: httpx.Response, stream: bool) -> list:
    if not stream:
        return [choice["finish_reason"] for choice in resp.json()["choices"]]
    events = [line[len("data: "):] for line in resp.text.splitlines() if line.startswith("data: ")]
    return [
        choice["finish_reason"]
        for event in events
        if event != "[DONE]"
        for choice in json.loads(event)["choices"]
        if choice.get("finish_reason") is not None
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "stop"])
async def test_cache_hit_reports_the_upstream_finish_reason(test_settings, stream, finish_reason):
    def upstream(request: httpx.Request) -> httpx.Response:
        if stream:
            body = (
                b'data: {"choices":[{"index":0,"delta":{"content":"The answer is"}}]}\n\n'
                + f'data: {{"choices":[{{"index":0,"delta":{{}},"finish_reason":"{finish_reason}"}}]}}\n\n'.encode()
                + b"data: [DONE]\n\n"
            )
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "The answer is"}, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
        )

    app = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "cut me off"}], "max_tokens": 2, "stream": stream}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = await client.post("/v1/chat/completions", json=payload)
        second = await client.post("/v1/chat/completions", json=payload)

    assert first.headers.get("X-Cache") == "MISS"
    assert second.headers.get("X-Cache") == "HIT", second.text
    assert _finish_reasons(second, stream) == [finish_reason], "a replay must not report a cut-off answer as complete"


@pytest.mark.asyncio
async def test_overwriting_a_cached_request_in_a_full_cache_keeps_the_other_entries():
    from gateway.cache.semantic_cache import SemanticCache
    from gateway.schemas.openai import ChatCompletionRequest

    cache = SemanticCache(ttl_seconds=60, max_size=2)
    a, b, c = (ChatCompletionRequest(model="m", messages=[{"role": "user", "content": x}]) for x in "abc")
    await cache.set(a, "A", "stop")
    await cache.set(b, "B", "stop")
    await cache.set(b, "B2", "stop")

    assert (await cache.get(a) or ("",))[0] == "A", "re-storing an existing request evicts nothing"
    assert (await cache.get(b))[0] == "B2", "the overwrite stores the new answer"

    await cache.set(c, "C", "stop")
    assert await cache.get(a) is None and await cache.get(c) is not None, "a new request in a full cache evicts the oldest"
