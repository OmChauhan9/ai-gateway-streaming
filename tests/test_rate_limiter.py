"""Unit and integration tests for Token Leaky-Bucket Rate Limiter."""

import asyncio
import pytest
from gateway.limiter.tokenizer_estimator import estimate_prompt_tokens, estimate_text_tokens
from gateway.limiter.leaky_bucket import TokenLeakyBucketLimiter, RateLimitExceeded
from gateway.schemas.openai import ChatMessage


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
async def test_gateway_429_endpoint_rejection(client):
    # Set small capacity on test app limiter to trigger 429
    gateway = client._transport.app  # FastAPI app
    ctx = gateway.state.ctx
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=50, leak_rate_tokens_per_second=10.0)

    # Send large prompt that exceeds 50 tokens
    large_prompt = "Tell me about deep learning. " * 30
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": large_prompt}],
        "stream": False,
    }
    headers = {"Authorization": "Bearer test-key-429"}

    resp = await client.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 429
    assert "Retry-After" in resp.headers
    body = resp.json()
    assert "detail" in body
    assert body["detail"]["error"]["type"] == "tokens_per_second_exceeded"
