"""Integration tests for Observability, Prometheus metrics, and OpenTelemetry instrumentation."""

import re

import pytest

from gateway.limiter.leaky_bucket import TokenLeakyBucketLimiter


@pytest.mark.asyncio
async def test_prometheus_metrics_scrape_endpoint(client):
    # Perform a request to populate metrics
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Telemetry test prompt"}],
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200

    # Scrape /metrics
    metrics_resp = await client.get("/metrics")
    assert metrics_resp.status_code == 200
    metrics_text = metrics_resp.text

    # Verify standard GenAI metrics are tracked
    assert "llm_requests_total" in metrics_text
    assert "llm_ttft_seconds" in metrics_text
    assert "llm_tpot_seconds" in metrics_text
    assert "llm_tokens_total" in metrics_text
    assert "circuit_breaker_state" in metrics_text


@pytest.mark.asyncio
async def test_health_endpoint(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert "providers" in data
    assert "mock-groq-primary" in data["providers"]


def _rate_limit_exceeded_total(metrics_text: str) -> float:
    match = re.search(r"^llm_rate_limit_exceeded_total (\S+)$", metrics_text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


@pytest.mark.asyncio
async def test_rate_limit_metric_counts_rejections_without_exposing_credentials(client):
    ctx = client._transport.app.state.ctx
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=1, leak_rate_tokens_per_second=0.001)
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    tokens = ["sk-telemetry-secret-alpha", "sk-telemetry-secret-beta"]

    before = _rate_limit_exceeded_total((await client.get("/metrics")).text)
    for token in tokens:
        resp = await client.post(
            "/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {token}"}
        )
        assert resp.status_code == 429, "request should be rate limited"
    metrics_text = (await client.get("/metrics")).text

    for token in tokens:
        assert token not in metrics_text, "bearer token must not appear on /metrics"
    assert _rate_limit_exceeded_total(metrics_text) - before == len(tokens), (
        "unlabeled counter should count each rate-limited request once"
    )
