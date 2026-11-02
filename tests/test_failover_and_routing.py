"""Integration tests for Provider Routing and Transparent Failover."""

import pytest
from gateway.circuit_breaker.state_machine import CircuitState


@pytest.mark.asyncio
async def test_normal_routing_to_primary(client):
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.headers.get("X-Gateway-Provider") == "mock-groq-primary"
    assert resp.headers.get("X-Gateway-Fallback") == "false"
    assert "text/event-stream" in resp.headers.get("Content-Type", "")

    # Read stream chunks
    body = resp.text
    assert "data: " in body
    assert "[DONE]" in body


@pytest.mark.asyncio
async def test_transparent_failover_on_primary_503(client):
    gateway = client._transport.app
    primary_app = gateway.state.primary_mock_app
    fallback_app = gateway.state.fallback_mock_app

    # Inject 503 on primary provider
    async with httpx_client_for(primary_app) as adm:
        await adm.post("/admin/fault-injection", json={"fail_with_status": 503})

    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Failover test prompt"}],
        "stream": True,
    }

    resp = await client.post("/v1/chat/completions", json=payload)
    # The client MUST experience 200 OK without dropped connection!
    assert resp.status_code == 200
    # Provider header should reflect fallback was used!
    assert resp.headers.get("X-Gateway-Provider") == "mock-together-fallback"
    assert resp.headers.get("X-Gateway-Fallback") == "true"

    body = resp.text
    assert "data: " in body
    assert "mock-together-fallback" in body

    # Reset primary fault
    async with httpx_client_for(primary_app) as adm:
        await adm.post("/admin/reset")


@pytest.mark.asyncio
async def test_circuit_trips_to_open_after_threshold(client):
    gateway = client._transport.app
    primary_app = gateway.state.primary_mock_app
    fallback_app = gateway.state.fallback_mock_app
    ctx = gateway.state.ctx
    primary_cb = ctx.registry.get_provider("mock-groq-primary").circuit_breaker
    primary_cb.recovery_timeout = 60.0
    primary_hits = 0

    @primary_app.middleware("http")
    async def count_primary_hits(request, call_next):
        nonlocal primary_hits
        if request.url.path.endswith("/chat/completions"):
            primary_hits += 1
        return await call_next(request)

    # Inject 500 error on primary
    async with httpx_client_for(primary_app) as adm:
        await adm.post("/admin/fault-injection", json={"fail_with_status": 500})

    # Issue requests equal to failure_threshold (3) with unique prompt to avoid cache
    for i in range(3):
        payload = {
            "model": "llama3-8b",
            "messages": [{"role": "user", "content": f"Trigger circuit trip {i}"}],
            "stream": True,
        }
        resp = await client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200  # all successfully rescued by fallback

    # Primary circuit breaker should now be OPEN!
    assert primary_cb.state == CircuitState.OPEN

    # Next request with fresh prompt should directly divert to fallback without hitting primary
    next_payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "New prompt when circuit is open"}],
        "stream": True,
    }
    hits_before = primary_hits
    resp = await client.post("/v1/chat/completions", json=next_payload)
    assert resp.status_code == 200
    assert resp.headers.get("X-Gateway-Provider") == "mock-together-fallback"
    assert primary_hits == hits_before, "OPEN circuit kept the request off the primary"

    # Reset
    await primary_cb.reset_manually()
    async with httpx_client_for(primary_app) as adm:
        await adm.post("/admin/reset")


@pytest.mark.asyncio
async def test_admin_manual_trip_endpoint(client):
    gateway = client._transport.app
    ctx = gateway.state.ctx
    primary_cb = ctx.registry.get_provider("mock-groq-primary").circuit_breaker

    # Trip primary
    trip_resp = await client.post("/v1/circuit-breakers/mock-groq-primary/trip")
    assert trip_resp.status_code == 200
    assert primary_cb.state == CircuitState.OPEN

    # Request immediately diverts to fallback
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "Trip check"}], "stream": False}
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert resp.headers.get("X-Gateway-Provider") == "mock-together-fallback"

    # Reset
    reset_resp = await client.post("/v1/circuit-breakers/mock-groq-primary/reset")
    assert reset_resp.status_code == 200
    assert primary_cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("primary_status, fallback_status", [(400, 200), (503, 400)], ids=["primary-400", "fallback-400"])
async def test_upstream_client_error_is_relayed_without_failover(test_settings, stream, primary_status, fallback_status):
    import httpx
    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    upstream_error = {"error": {"message": "maximum context length exceeded", "type": "invalid_request_error"}}
    hits = {"mock-groq-primary": 0, "mock-together-fallback": 0}

    def upstream(request: httpx.Request) -> httpx.Response:
        name = "mock-together-fallback" if ":8002" in str(request.url) else "mock-groq-primary"
        hits[name] += 1
        code = fallback_status if name == "mock-together-fallback" else primary_status
        if code == 200:
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
        if code >= 500:
            return httpx.Response(code, json={"error": {"message": "unavailable"}})
        return httpx.Response(code, json=upstream_error)

    gateway = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    ctx = gateway.state.ctx
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "too long"}], "stream": stream}
    transport = httpx.ASGITransport(app=gateway, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 400, resp.text
    assert resp.json() == upstream_error, "upstream error body relayed"
    rejecting = "mock-groq-primary" if primary_status == 400 else "mock-together-fallback"
    metrics = await ctx.registry.get_provider(rejecting).circuit_breaker.get_metrics()
    assert metrics["recent_failures_count"] == 0, "client error is not a provider failure"
    if primary_status == 400:
        assert hits["mock-together-fallback"] == 0, "no failover on a client error"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("fallback_fault", ["open", "503"])
async def test_primary_failure_without_usable_fallback_returns_503(client, stream, fallback_fault):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    gateway = client._transport.app
    ctx = gateway.state.ctx
    primary_cb = ctx.registry.get_provider("mock-groq-primary").circuit_breaker
    fallback_cb = ctx.registry.get_provider("mock-together-fallback").circuit_breaker
    spans = InMemorySpanExporter()
    ctx.otel.tracer_provider.add_span_processor(SimpleSpanProcessor(spans))

    async with httpx_client_for(gateway.state.primary_mock_app) as adm:
        await adm.post("/admin/fault-injection", json={"fail_with_status": 503})
    if fallback_fault == "open":
        fallback_cb.recovery_timeout = 60.0
        await client.post("/v1/circuit-breakers/mock-together-fallback/trip")
    else:
        async with httpx_client_for(gateway.state.fallback_mock_app) as adm:
            await adm.post("/admin/fault-injection", json={"fail_with_status": 503})

    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": f"No usable fallback {stream} {fallback_fault}"}],
        "stream": stream,
    }
    resp = await client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 503, resp.text
    assert resp.json()["detail"]["error"]["type"] == "all_providers_unavailable"
    primary_failures = (await primary_cb.get_metrics())["recent_failures_count"]
    assert primary_failures == 1, "primary failure recorded exactly once"
    fallback_failures = (await fallback_cb.get_metrics())["recent_failures_count"]
    assert fallback_failures == (0 if fallback_fault == "open" else 1), "fallback failure recorded once, only if contacted"
    ended = [s for s in spans.get_finished_spans() if s.name == "chat llama3-8b"]
    assert len(ended) == 1, "inference span ended"
    spans.shutdown()


@pytest.mark.asyncio
async def test_nonstream_non_json_200_fails_over_to_fallback(test_settings):
    import httpx
    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    def upstream(request: httpx.Request) -> httpx.Response:
        if ":8002" in str(request.url):
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "from fallback"}}]})
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text="<html><body>Bad gateway</body></html>")

    gateway = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    primary_cb = gateway.state.ctx.registry.get_provider("mock-groq-primary").circuit_breaker
    payload = {"model": "llama3-8b", "messages": [{"role": "user", "content": "malformed primary"}], "stream": False}
    transport = httpx.ASGITransport(app=gateway, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200, resp.text
    assert resp.headers.get("X-Gateway-Provider") == "mock-together-fallback"
    assert resp.json()["choices"][0]["message"]["content"] == "from fallback"
    assert (await primary_cb.get_metrics())["recent_failures_count"] == 1, "malformed body counts as a primary failure"


def httpx_client_for(app):
    """Helper to communicate with an in-memory FastAPI app directly."""
    import httpx
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")
