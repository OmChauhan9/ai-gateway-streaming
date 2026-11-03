"""Integration tests for handling upstream mid-stream disconnection edge cases."""

import asyncio
import re
import time
import httpx
import pytest
from gateway.stream.interceptor import telemetry_stream_interceptor


@pytest.mark.asyncio
async def test_midstream_drop_through_gateway_yields_error_frame_from_primary(client, gateway_app):
    primary = httpx.ASGITransport(app=gateway_app.state.primary_mock_app)
    async with httpx.AsyncClient(transport=primary, base_url="http://primary") as admin:
        await admin.post("/admin/fault-injection", json={"fail_midstream": True})

    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Mid-stream drop through the gateway"}],
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)

    assert resp.status_code == 200
    assert resp.headers["X-Gateway-Provider"] == "mock-groq-primary", "headers were already sent by the primary"
    assert resp.headers["X-Gateway-Fallback"] == "false", "a drop after headers cannot fail over"
    assert "Hello" in resp.text, "tokens sent before the drop reach the client"
    assert "event: error" in resp.text, "the drop surfaces as an SSE error frame"
    assert "upstream_connection_broken" in resp.text


@pytest.mark.asyncio
async def test_midstream_drop_interceptor_emits_error_frame():
    """Verify that when an upstream byte generator breaks mid-stream,
    the telemetry stream interceptor intercepts the exception and yields
    a standardized SSE error frame followed by [DONE].
    """
    async def faulty_upstream_stream():
        # First chunk succeeds
        yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
        # Second chunk succeeds
        yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n'
        # Mid-stream crash!
        raise ConnectionResetError("Remote server closed TCP connection abruptly")

    error_called = False
    caught_exception = None

    async def on_error(exc: Exception):
        nonlocal error_called, caught_exception
        error_called = True
        caught_exception = exc

    chunks = []
    async for chunk in telemetry_stream_interceptor(
        upstream_stream=faulty_upstream_stream(),
        request_id="test-drop-123",
        provider_name="groq",
        model_name="llama3-8b",
        t_start=time.perf_counter(),
        on_stream_error=on_error,
    ):
        chunks.append(chunk)

    full_output = b"".join(chunks).decode("utf-8")

    # 1. Verify initial chunks arrived
    assert "Hello" in full_output
    assert "world" in full_output

    # 2. Verify error callback was executed
    assert error_called is True
    assert isinstance(caught_exception, ConnectionResetError)

    # 3. Verify standardized SSE error frame was injected
    assert "event: error" in full_output
    assert "upstream_connection_broken" in full_output
    assert "Remote server closed TCP connection abruptly" in full_output
    assert "data: [DONE]" in full_output


@pytest.mark.asyncio
async def test_error_frame_starts_a_fresh_event_after_upstream_dies_mid_event():
    complete_event = b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'

    async def upstream_dies_mid_event():
        yield complete_event
        yield b'data: {"choices": [{"delta": {"conte'
        raise ConnectionResetError("peer reset")

    output = b""
    async for chunk in telemetry_stream_interceptor(
        upstream_stream=upstream_dies_mid_event(),
        request_id="test-drop-partial",
        provider_name="groq",
        model_name="llama3-8b",
        t_start=time.perf_counter(),
    ):
        output += chunk

    assert output.startswith(complete_event), "the complete event is forwarded"
    assert output[len(complete_event):].startswith(b"event: error\n"), (
        "the incomplete event is withheld so the error frame opens a new event"
    )


CONTENT_EVENT = b'data: {"choices": [{"index": 0, "delta": {"content": "The answer is"}, "finish_reason": null}]}\n\n'
FINISH_EVENT = b'data: {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}\n\n'


async def _intercept_ending(ending: bytes):
    async def upstream():
        yield CONTENT_EVENT
        if ending:
            yield ending

    completed, errors, output = [], [], b""

    async def on_complete(telemetry):
        completed.append(telemetry)

    async def on_error(exc):
        errors.append(exc)

    async for chunk in telemetry_stream_interceptor(
        upstream_stream=upstream(),
        request_id="test-ending",
        provider_name="groq",
        model_name="llama3-8b",
        t_start=time.perf_counter(),
        on_stream_complete=on_complete,
        on_stream_error=on_error,
    ):
        output += chunk
    return output, completed, errors


@pytest.mark.asyncio
async def test_stream_with_finish_reason_completes_without_done():
    output, completed, errors = await _intercept_ending(FINISH_EVENT)

    assert output == CONTENT_EVENT + FINISH_EVENT, "a finished stream is forwarded unchanged"
    assert len(completed) == 1 and not errors, "a finish_reason marks the stream complete"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending, forwarded",
    [
        (b"", CONTENT_EVENT),
        (b"data: [DONE]\n\n", CONTENT_EVENT + b"data: [DONE]\n\n"),
        (b'data: {"choices": [{"delta": {"conte', CONTENT_EVENT),
    ],
    ids=["eof-after-content", "done-without-finish-reason", "eof-mid-event"],
)
async def test_stream_that_ends_without_finish_reason_is_a_midstream_failure(ending, forwarded):
    output, completed, errors = await _intercept_ending(ending)

    assert not completed, "a stream that never finished is not reported complete"
    assert len(errors) == 1, "a stream that never finished is reported as a mid-stream error"
    assert output.startswith(forwarded + b"event: error\n"), (
        "complete events are forwarded, then the error frame opens a new event"
    )
    assert b"upstream_connection_broken" in output


@pytest.mark.asyncio
async def test_truncated_stream_is_not_cached_and_counts_as_provider_failure(test_settings):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode
    from gateway.main import create_app
    from gateway.upstream.client import AsyncUpstreamClient

    upstream_calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=CONTENT_EVENT)

    gateway = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    ctx = gateway.state.ctx
    spans = InMemorySpanExporter()
    ctx.otel.tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Truncated upstream stream"}],
        "stream": True,
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as c:
        first = await c.post("/v1/chat/completions", json=payload)
        second = await c.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert first.content.startswith(CONTENT_EVENT + b"event: error\n"), "the client is told the stream broke"
    assert second.headers.get("X-Cache") == "MISS", "the truncated text is not replayed from cache"
    assert len(upstream_calls) == 2
    primary_cb = ctx.registry.get_provider("mock-groq-primary").circuit_breaker
    assert (await primary_cb.get_metrics())["recent_failures_count"] == 2, "each truncated stream is a provider failure"
    ended = [s for s in spans.get_finished_spans() if s.name == "chat llama3-8b"]
    assert [s.status.status_code for s in ended] == [StatusCode.ERROR] * 2, "each span ends with the error"
    spans.shutdown()


def _disconnect_reporting_interceptor(upstream, disconnects):
    async def on_disconnect(completion_tokens):
        disconnects.append(completion_tokens)

    return telemetry_stream_interceptor(
        upstream_stream=upstream,
        request_id="test-disconnect",
        provider_name="groq",
        model_name="llama3-8b",
        t_start=time.perf_counter(),
        on_client_disconnect=on_disconnect,
    )


@pytest.mark.asyncio
async def test_cancel_while_awaiting_upstream_reports_streamed_tokens_and_propagates():
    async def stalled_upstream():
        yield CONTENT_EVENT
        await asyncio.Event().wait()

    disconnects, received = [], []
    stream = _disconnect_reporting_interceptor(stalled_upstream(), disconnects)

    async def consume():
        async for chunk in stream:
            received.append(chunk)

    task = asyncio.create_task(consume())
    while not received:
        await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert disconnects == [1], "the one streamed token is reported once"


@pytest.mark.asyncio
async def test_close_while_parked_at_yield_reports_streamed_tokens():
    disconnects = []

    async def upstream():
        yield CONTENT_EVENT + CONTENT_EVENT + FINISH_EVENT

    stream = _disconnect_reporting_interceptor(upstream(), disconnects)
    assert await stream.__anext__() == CONTENT_EVENT
    await stream.aclose()

    assert disconnects == [1], "the one forwarded token is reported once"


def _requests_total(metrics_text: str, status: str) -> float:
    labels = f'model="llama3-8b",provider="mock-groq-primary",status="{status}",stream="true"'
    match = re.search(rf"^llm_requests_total\{{{labels}\}} (\S+)$", metrics_text, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_is_counted_and_reconciled(client, gateway_app):
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import StatusCode
    from gateway.limiter.tokenizer_estimator import estimate_prompt_tokens
    from gateway.schemas.openai import ChatMessage

    ctx = gateway_app.state.ctx
    ctx.limiter.leak_rate = 1e-9
    spans = InMemorySpanExporter()
    ctx.otel.tracer_provider.add_span_processor(SimpleSpanProcessor(spans))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=gateway_app.state.primary_mock_app), base_url="http://primary"
    ) as admin:
        await admin.post("/admin/fault-injection", json={"added_tpot_delay_ms": 200})
    primary_cb = ctx.registry.get_provider("mock-groq-primary").circuit_breaker
    await client.post("/v1/circuit-breakers/mock-groq-primary/trip")
    await asyncio.sleep(0.25)

    messages = [{"role": "user", "content": "Client leaves after the first token"}]
    payload = {"model": "llama3-8b", "messages": messages, "stream": True}
    auth = {"Authorization": "Bearer disconnect-client"}
    before = (await client.get("/metrics")).text
    async with client.stream("POST", "/v1/chat/completions", json=payload, headers=auth) as resp:
        request_id = resp.headers["X-Gateway-Request-Id"]
        async for chunk in resp.aiter_raw():
            if b'"content"' in chunk:
                break
    after = (await client.get("/metrics")).text

    assert _requests_total(after, "499") - _requests_total(before, "499") == 1, "the request is counted as client-closed"
    assert _requests_total(after, "200") == _requests_total(before, "200"), "the request is not counted as a success"
    prompt_tokens = estimate_prompt_tokens([ChatMessage(**m) for m in messages])
    bucket = (await client.get("/v1/rate-limiter/status", headers=auth)).json()["bucket"]
    assert bucket["level"] == pytest.approx(prompt_tokens + 1), "the one streamed completion token is charged"
    ended = [s for s in spans.get_finished_spans() if s.attributes.get("gateway.request_id") == request_id]
    assert len(ended) == 1, "the span is ended"
    assert ended[0].attributes.get("gateway.client_disconnected") is True, "the span is marked as cancelled by the client"
    assert ended[0].status.status_code == StatusCode.UNSET, "a client disconnect is not a gateway error"
    breaker = await primary_cb.get_metrics()
    assert (breaker["state"], breaker["half_open_successes"]) == ("HALF-OPEN", 0), "a disconnect is not a provider success"
    assert breaker["recent_failures_count"] == 0, "a disconnect is not a provider failure"
    spans.shutdown()

    async with client.stream("POST", "/v1/chat/completions", json=payload, headers=auth) as replay:
        assert replay.headers.get("X-Cache") == "MISS", "the partial answer is not cached"
