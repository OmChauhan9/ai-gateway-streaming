"""Unit tests for Streaming SSE Interceptor and TTFT/TPOT latency profiling."""

import asyncio
import json
import time
import pytest
from gateway.schemas.gateway_models import StreamTelemetryData
from gateway.stream.interceptor import telemetry_stream_interceptor
from gateway.stream.sse_parser import iter_sse_events

FINISH_EVENT = b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'

@pytest.mark.parametrize(
    "event, content, finished",
    [
        (b'data: {"choices": [{"delta": {"role": "assistant"}, "finish_reason": null}]}\n\n', None, False),
        (b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n', "Hello", False),
        (b'data: {"choices": [{"delta": {"content": " world!"}}]}\n\n', " world!", False),
        (b'data: {"choices": [{"delta": {"content": "He said \\"hi\\""}}]}\n\n', 'He said "hi"', False),
        (FINISH_EVENT, None, True),
        (b'data: {"choices": [{"delta": {"content": "."}, "finish_reason": "length"}]}\n\n', ".", True),
        (b"data: [DONE]\n\n", None, False),
    ],
    ids=["role", "token", "leading-space", "escaped-quotes", "finish", "content-and-finish", "done"],
)
async def test_iter_sse_events_extracts_content_and_finish(event, content, finished):
    async def upstream():
        yield event

    assert [item async for item in iter_sse_events(upstream())] == [(event, content, finished)]


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

        yield FINISH_EVENT
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
    assert len(chunks_received) == 7  # 5 tokens + finish + [DONE]

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


def _delta_event(content: str, eol: str = "\n") -> bytes:
    payload = json.dumps({"choices": [{"delta": {"content": content}}]})
    return f"data: {payload}{eol}{eol}".encode("utf-8")


async def _intercept(chunks) -> tuple:
    """Run the interceptor over (delay_seconds, bytes) pairs; return forwarded bytes and telemetry."""
    async def upstream():
        for delay, chunk in chunks:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    telemetry = []

    async def on_complete(t: StreamTelemetryData):
        telemetry.append(t)

    forwarded = b""
    async for out in telemetry_stream_interceptor(
        upstream_stream=upstream(),
        request_id="test-framing",
        provider_name="groq",
        model_name="llama3-8b",
        t_start=time.perf_counter(),
        on_stream_complete=on_complete,
    ):
        forwarded += out
    assert len(telemetry) == 1, "stream completion is reported exactly once"
    return forwarded, telemetry[0]


async def test_events_coalesced_into_one_chunk_are_each_counted():
    chunk = (
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'
        + _delta_event("Hello")
        + _delta_event(" world")
        + FINISH_EVENT
        + b"data: [DONE]\n\n"
    )
    forwarded, telemetry = await _intercept([(0, chunk)])
    assert forwarded == chunk, "the client receives every upstream byte"
    assert telemetry.accumulated_text == "Hello world", "every event in the chunk is accumulated"
    assert telemetry.completion_tokens == 2, "every content event in the chunk is counted"


async def test_event_split_across_chunks_is_counted_when_it_completes():
    event = _delta_event("Hi")
    forwarded, telemetry = await _intercept(
        [(0, event[:20]), (0.1, event[20:]), (0.1, FINISH_EVENT)]
    )
    assert forwarded == event + FINISH_EVENT, "the client receives every upstream byte"
    assert telemetry.accumulated_text == "Hi", "the split event is reassembled"
    assert telemetry.completion_tokens == 1, "the split event is counted once"
    assert telemetry.ttft_ms >= 100.0, "TTFT cannot precede the arrival of the event's second half"
    assert telemetry.ttft_ms <= telemetry.total_latency_ms - 50.0, "TTFT is stamped when the event completes, not at EOF"


async def test_data_field_without_space_after_colon_is_parsed():
    chunk = b'data:{"choices": [{"delta": {"content": "Hi"}}]}\n\ndata:{"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
    _, telemetry = await _intercept([(0, chunk)])
    assert telemetry.accumulated_text == "Hi", "the space after `data:` is optional"
    assert telemetry.completion_tokens == 1


@pytest.mark.parametrize("eol", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
async def test_multiline_data_with_any_line_ending_fed_byte_by_byte(eol):
    stream = (
        f'data: {{"choices": [{{"delta":{eol}'
        f'data: {{"content": "Hi"}}}}]}}{eol}'
        f": keep-alive comment{eol}{eol}"
    ).encode("utf-8") + _delta_event("!", eol) + FINISH_EVENT
    forwarded, telemetry = await _intercept([(0, stream[i : i + 1]) for i in range(len(stream))])
    assert forwarded == stream, "the client receives every upstream byte"
    assert telemetry.accumulated_text == "Hi!", "data lines join with LF and every line ending ends a line"
    assert telemetry.completion_tokens == 2


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
