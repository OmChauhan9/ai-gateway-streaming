"""Integration tests for handling upstream mid-stream disconnection edge cases."""

import time
import pytest
from gateway.stream.interceptor import telemetry_stream_interceptor


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
