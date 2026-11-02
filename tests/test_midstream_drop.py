"""Integration tests for handling upstream mid-stream disconnection edge cases."""

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
