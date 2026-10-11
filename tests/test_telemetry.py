"""Integration tests for Observability, Prometheus metrics, and OpenTelemetry instrumentation."""

import pytest


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
