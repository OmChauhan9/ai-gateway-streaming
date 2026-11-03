"""Integration tests for Observability, Prometheus metrics, and OpenTelemetry instrumentation."""

import asyncio
import json
import re

import httpx
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
    ctx.limiter = TokenLeakyBucketLimiter(capacity_tokens=10, leak_rate_tokens_per_second=0.001)
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    tokens = ["sk-telemetry-secret-alpha", "sk-telemetry-secret-beta"]

    before = _rate_limit_exceeded_total((await client.get("/metrics")).text)
    for token in tokens:
        headers = {"Authorization": f"Bearer {token}"}
        first = await client.post("/v1/chat/completions", json=payload, headers=headers)
        assert first.status_code == 200, "the first small prompt fits the bucket"
        resp = await client.post("/v1/chat/completions", json=payload, headers=headers)
        assert resp.status_code == 429, "the repeat overflows the bucket and is rate limited"
    metrics_text = (await client.get("/metrics")).text

    for token in tokens:
        assert token not in metrics_text, "bearer token must not appear on /metrics"
    assert _rate_limit_exceeded_total(metrics_text) - before == len(tokens), (
        "unlabeled counter should count each rate-limited request once"
    )


def _provider_metric(metrics_text: str, name: str, provider: str):
    match = re.search(rf'^{name}{{provider="{provider}"}} (\S+)$', metrics_text, re.MULTILINE)
    return float(match.group(1)) if match else None


def _series_sum(metrics_text: str, metric: str, provider: str) -> float:
    pattern = rf'^{metric}_sum\{{[^}}]*provider="{re.escape(provider)}"[^}}]*\}} (\S+)$'
    match = re.search(pattern, metrics_text, re.MULTILINE)
    assert match, f"{metric} has a series for {provider}"
    return float(match.group(1))


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
async def test_breaker_metrics_count_one_trip_per_opening(test_settings, stream):
    import asyncio
    import httpx
    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    async def upstream(request: httpx.Request) -> httpx.Response:
        if ":8002" in str(request.url):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        await asyncio.sleep(0.1)
        return httpx.Response(503, json={"error": {"message": "down"}})

    gateway = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    gateway.state.ctx.registry.get_provider("mock-groq-primary").circuit_breaker.recovery_timeout = 60.0
    transport = httpx.ASGITransport(app=gateway, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        before = _provider_metric((await c.get("/metrics")).text, "circuit_breaker_trips_total", "mock-groq-primary") or 0.0
        await asyncio.gather(*(
            c.post("/v1/chat/completions", json={
                "model": "llama3-8b",
                "messages": [{"role": "user", "content": f"outage {i}"}],
                "stream": stream,
            })
            for i in range(10)
        ))
        metrics_text = (await c.get("/metrics")).text

    trips = _provider_metric(metrics_text, "circuit_breaker_trips_total", "mock-groq-primary") or 0.0
    assert trips - before == 1, "10 failures in flight open the breaker once"
    assert _provider_metric(metrics_text, "circuit_breaker_state", "mock-groq-primary") == 2.0, "gauge reads OPEN"


@pytest.mark.asyncio
async def test_breaker_metrics_and_health_follow_the_cooldown(client):
    import asyncio

    primary_cb = client._transport.app.state.ctx.registry.get_provider("mock-groq-primary").circuit_breaker
    before = _provider_metric((await client.get("/metrics")).text, "circuit_breaker_trips_total", "mock-groq-primary") or 0.0

    await client.post("/v1/circuit-breakers/mock-groq-primary/trip")
    metrics_text = (await client.get("/metrics")).text
    trips = _provider_metric(metrics_text, "circuit_breaker_trips_total", "mock-groq-primary") or 0.0
    assert trips - before == 1, "a manual trip counts"
    assert _provider_metric(metrics_text, "circuit_breaker_state", "mock-groq-primary") == 2.0

    await asyncio.sleep(primary_cb.recovery_timeout + 0.05)
    metrics_text = (await client.get("/metrics")).text
    health = (await client.get("/health")).json()
    breakers = (await client.get("/v1/circuit-breakers")).json()

    assert _provider_metric(metrics_text, "circuit_breaker_state", "mock-groq-primary") == 1.0, "gauge reads HALF-OPEN after the cooldown"
    assert health["providers"]["mock-groq-primary"]["circuit"]["state"] == "HALF-OPEN", "/health reports the state after the cooldown"
    assert breakers["mock-groq-primary"]["circuit"]["state"] == "HALF-OPEN"
    await primary_cb.reset_manually()


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
async def test_latency_after_failover_counts_from_gateway_ingress(test_settings, stream):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    primary_stall_s = 0.25
    fallback = f"fallback-ingress-{'stream' if stream else 'nonstream'}"
    settings = test_settings.model_copy(update={"fallback_provider_name": fallback, "cache_enabled": False})

    async def upstream(request: httpx.Request) -> httpx.Response:
        if ":8001" in str(request.url):
            await asyncio.sleep(primary_stall_s)
            return httpx.Response(503, json={"error": {"message": "primary down"}})
        if stream:
            chunk = {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}]}
            done = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            body = f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(done)}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, text=body)
        return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})

    gateway = create_app(settings=settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    spans = InMemorySpanExporter()
    gateway.state.ctx.otel.tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "ingress clock"}], "stream": stream}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as c:
        resp = await c.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("X-Gateway-Provider") == fallback
        metrics_text = (await c.get("/metrics")).text

    latency_s = _series_sum(metrics_text, "llm_gateway_latency_seconds", fallback)
    assert latency_s >= primary_stall_s, "recorded latency includes the time spent on the failed primary"
    if stream:
        ttft_s = _series_sum(metrics_text, "llm_ttft_seconds", fallback)
        assert ttft_s >= primary_stall_s, "recorded TTFT includes the time spent on the failed primary"
        [span] = [s for s in spans.get_finished_spans() if s.name == "chat llama3-8b"]
        assert span.attributes["gen_ai.server.ttft"] >= primary_stall_s * 1000, "span TTFT counts from ingress"
    spans.shutdown()
