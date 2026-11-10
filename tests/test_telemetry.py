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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize(
    "primary_open, fallback_name, primary_status, fallback_status, expected_status",
    [
        (True, "mock-together-fallback", 503, 503, 502),
        (True, "mock-together-fallback", 503, 401, 502),
        (False, "", 503, 503, 502),
        (False, "mock-together-fallback", 400, 200, 400),
        (False, "mock-together-fallback", 503, 400, 400),
    ],
    ids=["open-primary-fallback-503", "open-primary-fallback-401", "no-fallback-503", "primary-400", "fallback-400"],
)
async def test_failed_request_ends_its_span_with_error_status(
    test_settings, stream, primary_open, fallback_name, primary_status, fallback_status, expected_status
):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    settings = test_settings.model_copy(update={"fallback_provider_name": fallback_name, "cache_enabled": False})

    def upstream(request: httpx.Request) -> httpx.Response:
        code = fallback_status if ":8002" in str(request.url) else primary_status
        if code == 200:
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        return httpx.Response(code, json={"error": {"message": f"upstream said {code}"}})

    gateway = create_app(settings=settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    gateway.state.ctx.registry.get_provider("mock-groq-primary").circuit_breaker.recovery_timeout = 60.0
    spans = InMemorySpanExporter()
    gateway.state.ctx.otel.tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "span on failure"}], "stream": stream}
    transport = httpx.ASGITransport(app=gateway, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        if primary_open:
            await c.post("/v1/circuit-breakers/mock-groq-primary/trip")
        resp = await c.post("/v1/chat/completions", json=payload)
    finished = [s for s in spans.get_finished_spans() if s.name == "chat llama3-8b"]
    spans.shutdown()

    assert resp.status_code == expected_status, resp.text
    assert len(finished) == 1, "the request's span is ended once"
    assert finished[0].status.status_code == StatusCode.ERROR, "the span records the failure"


PRIMARY, FALLBACK = "mock-groq-primary", "mock-together-fallback"


def _sse_upstream(*, drop: bool) -> httpx.Response:
    async def body():
        yield b'data: {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": null}]}\n\n'
        if drop:
            raise httpx.ReadError("upstream socket reset")
        yield b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\ndata: [DONE]\n\n'

    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body())


def _upstream_replying(status: int, stream: bool) -> httpx.Response:
    if status != 200:
        return httpx.Response(status, json={"error": {"message": f"upstream {status}"}})
    if stream:
        return _sse_upstream(drop=False)
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]})


def _request_counts(metrics_text: str) -> dict:
    return {
        labels: float(value)
        for labels, value in re.findall(r"^llm_requests_total\{(.*)\} (\S+)$", metrics_text, re.MULTILINE)
    }


async def _post_and_count(settings, upstream, payload, tripped=(), warm_up=False):
    """POST once and return the response and the llm_requests_total series it moved."""
    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    gateway = create_app(
        settings=settings.model_copy(update={"cache_enabled": False}),
        client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)),
    )
    for name in tripped:
        await gateway.state.ctx.registry.get_provider(name).circuit_breaker.trip_manually()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as c:
        if warm_up:
            assert (await c.post("/v1/chat/completions", json=payload)).status_code == 200, "the warm-up is served"
        before = _request_counts((await c.get("/metrics")).text)
        resp = await c.post("/v1/chat/completions", json=payload)
        after = _request_counts((await c.get("/metrics")).text)
    return resp, {k: v - before.get(k, 0.0) for k, v in after.items() if v != before.get(k, 0.0)}


def _series(model: str, provider: str, status: str, stream: bool) -> str:
    return f'model="{model}",provider="{provider}",status="{status}",stream="{str(stream).lower()}"'


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize(
    "primary, fallback, tripped, http_status, counted_provider",
    [
        (200, 200, (), 200, PRIMARY),
        (503, 503, (), 503, FALLBACK),
        (200, 503, (PRIMARY,), 502, FALLBACK),
        (401, 401, (), 502, FALLBACK),
        (200, 200, (PRIMARY, FALLBACK), 503, "none"),
        (400, 200, (), 400, PRIMARY),
        (503, 400, (), 400, FALLBACK),
    ],
    ids=["ok", "both-down", "fallback-only-down", "auth-rejected", "all-circuits-open", "primary-400", "fallback-400"],
)
async def test_each_chat_response_is_counted_once_under_the_status_the_client_got(
    test_settings, stream, primary, fallback, tripped, http_status, counted_provider
):
    def upstream(request: httpx.Request) -> httpx.Response:
        return _upstream_replying(primary if ":8001" in str(request.url) else fallback, stream)

    payload = {"model": "count-outcomes", "messages": [{"role": "user", "content": "hi"}], "stream": stream}
    resp, increments = await _post_and_count(test_settings, upstream, payload, tripped=tripped)

    assert resp.status_code == http_status, resp.text
    assert increments == {_series("count-outcomes", counted_provider, str(http_status), stream): 1.0}, (
        "exactly one increment, labeled with the status the client received"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
async def test_request_cancelled_before_response_ends_its_span_with_error_status(test_settings, stream):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode

    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    upstream_called = asyncio.Event()

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_called.set()
        await asyncio.sleep(10)

    settings = test_settings.model_copy(update={"cache_enabled": False})
    gateway = create_app(settings=settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    spans = InMemorySpanExporter()
    gateway.state.ctx.otel.tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "cancelled"}], "stream": stream}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as c:
        request = asyncio.create_task(c.post("/v1/chat/completions", json=payload))
        await upstream_called.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    finished = [s for s in spans.get_finished_spans() if s.name == "chat llama3-8b"]
    spans.shutdown()

    assert len(finished) == 1, "the cancelled request's span is ended once"
    assert finished[0].status.status_code == StatusCode.ERROR, "a cancelled request is not recorded as a success"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("http_status", [413, 429])
async def test_requests_the_limiter_rejects_are_counted_without_a_provider(test_settings, stream, http_status):
    settings = test_settings.model_copy(update={
        "rate_limiter_capacity_tokens": 10,
        "rate_limiter_leak_rate_tokens_per_second": 0.001,
    })
    prompt = "word " * 50 if http_status == 413 else "hi"
    payload = {"model": "count-limited", "messages": [{"role": "user", "content": prompt}], "stream": stream}
    resp, increments = await _post_and_count(
        settings, lambda request: _upstream_replying(200, stream), payload, warm_up=http_status == 429
    )

    assert resp.status_code == http_status, resp.text
    assert increments == {_series("count-limited", "none", str(http_status), stream): 1.0}, (
        "a limiter rejection is counted once, with no provider"
    )


@pytest.mark.asyncio
async def test_stream_that_fails_after_headers_is_counted_as_502(test_settings):
    payload = {"model": "count-dropped", "messages": [{"role": "user", "content": "hi"}], "stream": True}
    resp, increments = await _post_and_count(test_settings, lambda request: _sse_upstream(drop=True), payload)

    assert resp.status_code == 200 and '"code": 502' in resp.text, "headers went out, then an error event with code 502"
    assert increments == {_series("count-dropped", PRIMARY, "502", True): 1.0}, (
        "the failed stream is counted once, as a 502 and not a 200"
    )
