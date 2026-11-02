"""Unit tests for Streaming SSE Interceptor and TTFT/TPOT latency profiling."""

import asyncio
import json
import time
import pytest
from gateway.schemas.gateway_models import StreamTelemetryData
from gateway.stream.interceptor import telemetry_stream_interceptor
from gateway.stream.sse_parser import extract_content_from_sse_chunk


def test_extract_content_from_sse_chunk():
    # Role announcement (no content)
    chunk1 = b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
    assert extract_content_from_sse_chunk(chunk1) is None

    # Normal token
    chunk2 = b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n'
    assert extract_content_from_sse_chunk(chunk2) == "Hello"

    # Token with whitespace
    chunk3 = b'data: {"choices": [{"delta": {"content": " world!"}}]}\n\n'
    assert extract_content_from_sse_chunk(chunk3) == " world!"

    # Escaped quotes
    chunk4 = b'data: {"choices": [{"delta": {"content": "He said \\"hi\\""}}]}\n\n'
    assert extract_content_from_sse_chunk(chunk4) == 'He said "hi"'

    # Done chunk
    chunk_done = b'data: [DONE]\n\n'
    assert extract_content_from_sse_chunk(chunk_done) is None


@pytest.mark.asyncio
async def test_telemetry_stream_interceptor_profiling():
    # Create mock chunks with intentional delays to test TTFT and TPOT
    async def mock_stream():
        # First: 30ms delay before first chunk (TTFT simulation)
        await asyncio.sleep(0.03)
        yield b'data: {"choices": [{"delta": {"content": "Token1"}}]}\n\n'

        # Next chunks with 10ms delay each (TPOT simulation)
        for i in range(2, 6):
            await asyncio.sleep(0.01)
            chunk_dict = {"choices": [{"delta": {"content": f" Token{i}"}}]}
            yield f"data: {json.dumps(chunk_dict)}\n\n".encode("utf-8")

        yield b"data: [DONE]\n\n"

    collected_telemetry = None

    async def on_complete(telemetry: StreamTelemetryData):
        nonlocal collected_telemetry
        collected_telemetry = telemetry

    t_start = time.perf_counter()
    chunks_received = []

    async for chunk in telemetry_stream_interceptor(
        upstream_stream=mock_stream(),
        request_id="test-req-001",
        provider_name="groq",
        model_name="llama3-8b",
        t_start=t_start,
        prompt_tokens=10,
        on_stream_complete=on_complete,
    ):
        chunks_received.append(chunk)

    # 1. Verify chunks were yielded without drops
    assert len(chunks_received) == 6  # 5 tokens + 1 [DONE]

    # 2. Verify telemetry accuracy
    assert collected_telemetry is not None
    assert collected_telemetry.completion_tokens == 5
    assert collected_telemetry.prompt_tokens == 10
    assert collected_telemetry.provider == "groq"
    assert collected_telemetry.model == "llama3-8b"

    # TTFT should reflect ~30ms prefill delay (allowing +/- 15ms margin for test scheduling)
    assert 20.0 <= collected_telemetry.ttft_ms <= 80.0

    # TPOT should reflect ~10ms per token (4 intervals across 5 tokens)
    assert 5.0 <= collected_telemetry.tpot_ms <= 30.0


@pytest.mark.asyncio
async def test_streamed_completion_arrives_in_multiple_chunks(client):
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Chunked delivery through the gateway"}],
        "stream": True,
    }
    async with client.stream("POST", "/v1/chat/completions", json=payload) as resp:
        chunks = [chunk async for chunk in resp.aiter_raw()]

    assert resp.status_code == 200
    assert len(chunks) > 1, "the SSE body reaches the client as the gateway sends it, not as one buffered chunk"
    assert b"".join(chunks).endswith(b"data: [DONE]\n\n")
