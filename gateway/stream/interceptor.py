"""Non-blocking Server-Sent Events (SSE) Streaming Interceptor.

Wraps the upstream response chunk generator inside an async generator to profile:
- TTFT (Time To First Token): T_first - T_start
- TPOT (Time Per Output Token): (T_last - T_first) / (N_tokens - 1)
- Inter-Token Latencies (ITL) and token throughput (tokens/sec)

Maintains <5ms gateway latency overhead by avoiding whole-response buffering.
"""

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable, List, Optional
from gateway.schemas.gateway_models import StreamTelemetryData
from gateway.stream.error_frames import generate_sse_error_frame
from gateway.stream.sse_parser import extract_content_from_sse_chunk


async def telemetry_stream_interceptor(
    upstream_stream: AsyncIterator[bytes],
    request_id: str,
    provider_name: str,
    model_name: str,
    t_start: float,
    prompt_tokens: int = 0,
    fallback_used: bool = False,
    on_stream_complete: Optional[Callable[[StreamTelemetryData], Awaitable[None]]] = None,
    on_stream_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
) -> AsyncIterator[bytes]:
    """Async generator interceptor profiling streaming LLM metrics in real time.
    
    Args:
        upstream_stream: Raw byte chunks from httpx response.aiter_bytes()
        request_id: Unique correlation ID
        provider_name: Selected upstream provider name
        model_name: Model identifier
        t_start: Monotonic timestamp when upstream request was initiated
        prompt_tokens: Estimated prompt tokens
        fallback_used: Whether failover occurred
        on_stream_complete: Coroutine invoked with finalized telemetry data
        on_stream_error: Coroutine invoked if upstream drops mid-stream
    """
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    token_count: int = 0
    inter_token_intervals: List[float] = []
    previous_token_time: Optional[float] = None
    accumulated_content: List[str] = []

    try:
        async for chunk in upstream_stream:
            now = time.perf_counter()

            # Fast inspection of chunk content
            content = extract_content_from_sse_chunk(chunk)
            if content:
                token_count += 1
                accumulated_content.append(content)

                if t_first is None:
                    # First non-empty token arrival
                    t_first = now
                    previous_token_time = now
                else:
                    # Subsequent tokens: calculate inter-token latency (ITL)
                    if previous_token_time is not None:
                        itl = (now - previous_token_time) * 1000.0  # ms
                        inter_token_intervals.append(itl)
                    previous_token_time = now

                t_last = now

            # Yield raw byte chunk immediately to the client stream without delay
            yield chunk

        # End of stream reached successfully
        t_end = time.perf_counter()
        if t_first is None:
            # Fallback if no content tokens parsed (e.g. empty stream)
            t_first = t_end
        if t_last is None:
            t_last = t_end

        # Metrics computation
        ttft_ms = (t_first - t_start) * 1000.0
        total_latency_ms = (t_end - t_start) * 1000.0

        if token_count > 1:
            tpot_ms = ((t_last - t_first) / (token_count - 1)) * 1000.0
        else:
            tpot_ms = 0.0

        avg_itl_ms = (
            sum(inter_token_intervals) / len(inter_token_intervals)
            if inter_token_intervals
            else 0.0
        )

        telemetry = StreamTelemetryData(
            request_id=request_id,
            provider=provider_name,
            model=model_name,
            ttft_ms=round(ttft_ms, 3),
            tpot_ms=round(tpot_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
            completion_tokens=token_count,
            prompt_tokens=prompt_tokens,
            inter_token_latencies_avg_ms=round(avg_itl_ms, 3),
            accumulated_text="".join(accumulated_content),
            fallback_used=fallback_used,
            cached=False,
        )

        # Notify telemetry collectors (Prometheus / OpenTelemetry)
        if on_stream_complete:
            try:
                await on_stream_complete(telemetry)
            except Exception:
                pass  # Telemetry emission must not interrupt client streaming

    except Exception as exc:
        # Mid-stream connection drop or exception
        if on_stream_error:
            try:
                await on_stream_error(exc)
            except Exception:
                pass

        # Emit standardized SSE error frame to client before closing socket
        error_frame = generate_sse_error_frame(
            message=f"Upstream provider connection error mid-stream: {str(exc)}",
            error_type="upstream_connection_broken",
            code=502,
            provider=provider_name,
        )
        yield error_frame
