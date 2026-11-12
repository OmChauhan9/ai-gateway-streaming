"""Unit and integration tests for Token Leaky-Bucket Rate Limiter."""

import asyncio
import re
from types import SimpleNamespace

import httpx
import pytest
from gateway.limiter.tokenizer_estimator import estimate_prompt_tokens, estimate_request_tokens, estimate_text_tokens
from gateway.limiter import leaky_bucket
from gateway.limiter.leaky_bucket import TokenLeakyBucketLimiter, RateLimitExceeded
from gateway.schemas.openai import ChatCompletionRequest, ChatMessage


def test_token_estimation():
    # Empty
    assert estimate_text_tokens("") == 0
    # Short
    assert estimate_text_tokens("hello") == 2
    # Longer text: ~20 chars -> ~5-6 tokens
    assert 4 <= estimate_text_tokens("This is a quick sentence.") <= 7

    messages = [
        ChatMessage(role="user", content="What is the speed of light?"),
    ]
    prompt_tokens = estimate_prompt_tokens(messages)
    # Framing overhead + user content + prime
    assert 8 <= prompt_tokens <= 18


def test_token_estimate_for_a_null_content_tool_call_message_counts_its_arguments():
    def tool_call_turn(arguments: str) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "w", "arguments": arguments}}],
        )

    small = estimate_prompt_tokens([tool_call_turn("{}")])
    large = estimate_prompt_tokens([tool_call_turn("{}" + " " * 400)])

    assert small > 0
    assert 95 <= large - small <= 105, "400 more characters of arguments cost about 100 tokens"


def test_text_parts_are_estimated_like_the_same_text_as_a_string():
    as_parts = ChatMessage(role="user", content=[{"type": "text", "text": "What is the speed of light?"}])
    as_string = ChatMessage(role="user", content="What is the speed of light?")

    assert estimate_prompt_tokens([as_parts]) == estimate_prompt_tokens([as_string])


@pytest.mark.asyncio
async def test_leaky_bucket_basic_acquire():
    limiter = TokenLeakyBucketLimiter(capacity_tokens=500, leak_rate_tokens_per_second=100.0)

    # First acquisition should succeed
    admitted, retry_after = await limiter.acquire("client_a", 200)
    assert admitted is True
    assert retry_after == 0.0

    status = await limiter.get_status("client_a")
    assert status["level"] == 200.0
    assert status["remaining"] == 300.0


@pytest.mark.asyncio
async def test_leaky_bucket_overflow_and_retry_after():
    limiter = TokenLeakyBucketLimiter(capacity_tokens=300, leak_rate_tokens_per_second=100.0)

    # Acquire up to 250
    await limiter.acquire("client_b", 250)

    # Request 100 more (250 + 100 = 350 > 300) -> should exceed
    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.acquire("client_b", 100)

    err = exc_info.value
    assert err.capacity == 300
    assert err.requested == 100
    # 350 - 300 = 50 tokens overflow / 100 tokens/sec = 0.5s retry after
    assert 0.45 <= err.retry_after <= 0.55


@pytest.mark.asyncio
async def test_leaky_bucket_drains_over_time():
    limiter = TokenLeakyBucketLimiter(capacity_tokens=200, leak_rate_tokens_per_second=500.0)

    # Fill bucket completely
    await limiter.acquire("client_c", 200)

    # Wait 0.15s (should leak 0.15 * 500 = 75 tokens)
    await asyncio.sleep(0.15)

    status = await limiter.get_status("client_c")
    assert status["level"] < 150.0

    # Now an acquisition of 50 should succeed
    admitted, _ = await limiter.acquire("client_c", 50)
    assert admitted is True


@pytest.mark.asyncio
async def test_cleanup_prunes_buckets_that_drained_and_went_idle(monkeypatch):
    limiter = TokenLeakyBucketLimiter(
        capacity_tokens=50, leak_rate_tokens_per_second=10.0, cleanup_interval_seconds=0
    )
    for i in range(100):
        await limiter.acquire(f"idle_{i}", 5)

    two_hours_later = leaky_bucket.time.monotonic() + 7200
    monkeypatch.setattr(leaky_bucket, "time", SimpleNamespace(monotonic=lambda: two_hours_later))
    await limiter.acquire("active", 5)

    assert set(limiter._buckets) == {"active"}, "drained buckets idle for 2h must be pruned"


@pytest.mark.asyncio
async def test_completion_tokens_beyond_capacity_are_owed_until_they_drain():
    limiter = TokenLeakyBucketLimiter(capacity_tokens=1000, leak_rate_tokens_per_second=100.0)
    await limiter.acquire("debtor", 10)
    await limiter.release_or_adjust("debtor", 10, 5010)

    with pytest.raises(RateLimitExceeded) as exc_info:
        await limiter.acquire("debtor", 10)

    # 5010 + 10 - 1000 = 4020 tokens over capacity, at 100 tokens/s
    assert 40.1 <= exc_info.value.retry_after <= 40.25, "the wait covers the whole debt, not only the capacity"


@pytest.mark.asyncio
async def test_cleanup_keeps_a_bucket_that_is_still_in_debt(monkeypatch):
    limiter = TokenLeakyBucketLimiter(
        capacity_tokens=1000, leak_rate_tokens_per_second=0.5, cleanup_interval_seconds=0
    )
    await limiter.acquire("debtor", 10)
    await limiter.release_or_adjust("debtor", 10, 10010)

    two_hours_later = leaky_bucket.time.monotonic() + 7200
    monkeypatch.setattr(leaky_bucket, "time", SimpleNamespace(monotonic=lambda: two_hours_later))
    await limiter.acquire("active", 5)

    with pytest.raises(RateLimitExceeded):
        await limiter.acquire("debtor", 10)


@pytest.mark.asyncio
async def test_gateway_429_endpoint_rejection(client):
    ctx = client._transport.app.state.ctx
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=50, leak_rate_tokens_per_second=10.0)
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "x" * 120}],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test-key-429"}

    first = await client.post("/v1/chat/completions", json=payload, headers=headers)
    assert first.status_code == 200, "a prompt that fits the empty bucket is admitted"

    resp = await client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    error = resp.json()["error"]
    assert error["type"] == "tokens_per_second_exceeded"
    assert error["retry_after_seconds"] > 0, "the wait hint stays in the OpenAI-shaped body"


@pytest.mark.asyncio
async def test_gateway_413_when_prompt_exceeds_bucket_capacity(client):
    ctx = client._transport.app.state.ctx
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=50, leak_rate_tokens_per_second=10.0)
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Tell me about deep learning. " * 30}],
        "stream": False,
    }

    resp = await client.post(
        "/v1/chat/completions", json=payload, headers={"Authorization": "Bearer test-key-413"}
    )
    assert resp.status_code == 413, "a prompt larger than capacity can never be admitted"
    assert "Retry-After" not in resp.headers, "waiting cannot help, so no Retry-After"
    assert resp.json()["error"]["type"] == "request_exceeds_token_capacity"


LONG_TEXT = "Tell me about deep learning. " * 30


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fields",
    [
        {"messages": [{"role": "user", "content": LONG_TEXT}]},
        {
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "description": LONG_TEXT, "parameters": {"type": "object"}}}],
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": LONG_TEXT}}]},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": LONG_TEXT}}]},
            ]
        },
        {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "annotation": LONG_TEXT}]}]},
    ],
    ids=["in_content", "in_top_level_field", "in_message_field", "in_null_content_tool_calls", "in_text_part_field"],
)
async def test_forwarded_fields_count_toward_the_token_capacity(client, fields):
    ctx = client._transport.app.state.ctx
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=50, leak_rate_tokens_per_second=10.0)

    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "llama3-8b", "stream": False, **fields},
        headers={"Authorization": "Bearer test-key-forwarded-413"},
    )

    assert resp.status_code == 413, "text the upstream receives is budgeted wherever in the request it sits"


@pytest.mark.asyncio
async def test_non_streaming_request_charges_upstream_completion_tokens(client):
    ctx = client._transport.app.state.ctx
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=100000, leak_rate_tokens_per_second=1e-9)
    messages = [{"role": "user", "content": "hi"}]
    headers = {"Authorization": "Bearer test-key-reconcile"}

    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "llama3-8b", "messages": messages, "stream": False},
        headers=headers,
    )
    assert resp.status_code == 200
    completion_tokens = resp.json()["usage"]["completion_tokens"]
    assert completion_tokens > 0

    status = (await client.get("/v1/rate-limiter/status", headers=headers)).json()["bucket"]
    expected = estimate_prompt_tokens([ChatMessage(**m) for m in messages]) + completion_tokens
    assert status["level"] == expected, "bucket holds the prompt estimate plus upstream completion tokens"


PRIMARY, FALLBACK = "mock-groq-primary", "mock-together-fallback"


def _tokens_counted(metrics_text: str, model: str) -> float:
    return sum(
        float(value)
        for value in re.findall(rf'^llm_tokens_total\{{model="{model}",.*\}} (\S+)$', metrics_text, re.MULTILINE)
    )


def _upstream(primary: int, fallback: int, stream: bool):
    def reply(request: httpx.Request) -> httpx.Response:
        code = fallback if ":8002" in str(request.url) else primary
        if code != 200:
            return httpx.Response(code, json={"error": {"message": f"upstream {code}", "type": "upstream"}})
        if stream:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b'data: {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}\n\n'
                b"data: [DONE]\n\n",
            )
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    return reply


async def _post_and_measure(settings, primary, fallback, tripped, payload, held=0):
    """POST once, with `held` tokens already in the client's bucket.

    Returns the response, the bucket level before and after, and the llm_tokens_total it moved.
    """
    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    gateway = create_app(
        settings=settings.model_copy(update={"cache_enabled": False}),
        client=AsyncUpstreamClient(transport=httpx.MockTransport(_upstream(primary, fallback, payload["stream"]))),
    )
    ctx = gateway.state.ctx
    ctx.limiter.leak_rate = 1e-9
    for name in tripped:
        breaker = ctx.registry.get_provider(name).circuit_breaker
        breaker.recovery_timeout = 60.0
        await breaker.trip_manually()
    if held:
        await ctx.limiter.acquire("charge-client", held)
    auth = {"Authorization": "Bearer charge-client"}
    transport = httpx.ASGITransport(app=gateway, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        level_before = (await c.get("/v1/rate-limiter/status", headers=auth)).json()["bucket"]["level"]
        tokens_before = _tokens_counted((await c.get("/metrics")).text, payload["model"])
        resp = await c.post("/v1/chat/completions", json=payload, headers=auth)
        level = (await c.get("/v1/rate-limiter/status", headers=auth)).json()["bucket"]["level"]
        tokens = _tokens_counted((await c.get("/metrics")).text, payload["model"]) - tokens_before
    return resp, level_before, level, tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize(
    "primary, fallback, tripped, fallback_name, http_status, error_type",
    [
        (200, 200, (PRIMARY, FALLBACK), FALLBACK, 503, "all_providers_unavailable"),
        (503, 200, (FALLBACK,), FALLBACK, 503, "all_providers_unavailable"),
        (503, 503, (), FALLBACK, 503, "all_providers_unavailable"),
        (401, 200, (FALLBACK,), FALLBACK, 502, "upstream_auth_failed"),
        (503, 401, (), FALLBACK, 502, "upstream_auth_failed"),
        (200, 503, (PRIMARY,), FALLBACK, 502, "upstream_error"),
        (200, 401, (PRIMARY,), FALLBACK, 502, "upstream_auth_failed"),
        (503, 200, (), "", 502, "upstream_error"),
        (401, 200, (), "", 502, "upstream_auth_failed"),
    ],
    ids=[
        "all-circuits-open",
        "primary-503-fallback-open",
        "both-503",
        "primary-401-fallback-open",
        "primary-503-fallback-401",
        "primary-open-fallback-503",
        "primary-open-fallback-401",
        "no-fallback-503",
        "no-fallback-401",
    ],
)
async def test_request_the_gateway_cannot_serve_costs_the_client_no_tokens(
    test_settings, stream, primary, fallback, tripped, fallback_name, http_status, error_type
):
    settings = test_settings.model_copy(update={"fallback_provider_name": fallback_name})
    payload = {"model": "charge-unserved", "messages": [{"role": "user", "content": "unserved"}], "stream": stream}

    resp, level_before, level, tokens = await _post_and_measure(settings, primary, fallback, tripped, payload, held=100)

    assert resp.status_code == http_status, resp.text
    assert resp.json()["error"]["type"] == error_type
    assert level == level_before, "the prompt charge is refunded once, leaving other charges in place"
    assert level - level_before == tokens, "the limiter charge matches llm_tokens_total"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("primary, fallback", [(400, 200), (503, 400)], ids=["primary-400", "fallback-400"])
async def test_relayed_client_error_keeps_the_prompt_charge(test_settings, stream, primary, fallback):
    payload = {"model": "charge-relayed", "messages": [{"role": "user", "content": "bad request"}], "stream": stream}

    resp, _, level, _ = await _post_and_measure(test_settings, primary, fallback, (), payload)

    assert resp.status_code == 400, resp.text
    assert level == estimate_request_tokens(ChatCompletionRequest(**payload)), "a client fault keeps the prompt charge"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("http_status", [413, 429])
async def test_limiter_rejection_charges_nothing(test_settings, stream, http_status):
    settings = test_settings.model_copy(update={"rate_limiter_capacity_tokens": 10})
    prompt = "word " * 50 if http_status == 413 else "hi"
    payload = {"model": "charge-limited", "messages": [{"role": "user", "content": prompt}], "stream": stream}

    held = 10 if http_status == 429 else 0
    resp, level_before, level, _ = await _post_and_measure(settings, 200, 200, (), payload, held=held)

    assert resp.status_code == http_status, resp.text
    assert level == level_before, "a request the limiter rejects adds nothing to the bucket"
