"""High-fidelity synthetic LLM provider mock server for testing & fault injection.

Can be run as a standalone server or mounted via httpx.ASGITransport for unit tests.
Supports configurable TTFT, TPOT, token counts, and fault injection (503, 429, mid-stream aborts).
"""

import asyncio
import json
import time
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel


class FaultConfig(BaseModel):
    fail_with_status: Optional[int] = None
    fail_midstream: bool = False
    added_ttft_delay_ms: float = 30.0
    added_tpot_delay_ms: float = 10.0
    tokens_to_generate: int = 15


def create_mock_upstream_app(provider_name: str = "mock-upstream") -> FastAPI:
    """Create FastAPI application acting as a mock LLM provider."""
    app = FastAPI(title=f"Mock LLM Provider ({provider_name})")

    # Mutable operational state for dynamic fault injection during tests
    state = FaultConfig()

    @app.post("/admin/fault-injection")
    async def configure_fault(config: FaultConfig):
        """Dynamically configure faults (503s, 429s, midstream aborts, latencies)."""
        nonlocal state
        state = config
        return {"status": "updated", "provider": provider_name, "config": state.model_dump()}

    @app.post("/admin/reset")
    async def reset_fault():
        """Reset fault injection to healthy defaults."""
        nonlocal state
        state = FaultConfig()
        return {"status": "reset", "provider": provider_name}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "provider": provider_name}

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(
        request: Request,
        x_mock_fault: Optional[str] = Header(None),
    ):
        body = await request.json()
        model = body.get("model", "llama3-8b")
        stream = body.get("stream", False)
        messages = body.get("messages", [])

        # 1. Fault injection evaluation (via header or server state)
        fault_code = state.fail_with_status
        if x_mock_fault and x_mock_fault.isdigit():
            fault_code = int(x_mock_fault)

        if fault_code:
            if fault_code == 429:
                return JSONResponse(
                    status_code=429,
                    content={"error": {"message": "Rate limit exceeded on upstream", "type": "tokens_exceeded"}},
                )
            elif fault_code >= 500:
                return JSONResponse(
                    status_code=fault_code,
                    content={"error": {"message": f"Upstream service unavailable ({provider_name})", "type": "server_error"}},
                )

        simulate_midstream_drop = state.fail_midstream or (x_mock_fault == "midstream-drop")

        # 2. Streaming Response (SSE)
        if stream:
            async def event_generator() -> AsyncIterator[bytes]:
                req_id = f"chatcmpl-mock-{int(time.time()*1000)}"
                created_ts = int(time.time())

                # Simulate TTFT delay (network propagation + prefill compute)
                if state.added_ttft_delay_ms > 0:
                    await asyncio.sleep(state.added_ttft_delay_ms / 1000.0)

                # Initial role chunk
                role_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8")

                # Sample tokens to stream
                tokens = [
                    "Hello", "!", " I", " am", f" {provider_name}", ".",
                    " Streaming", " with", " sub", "-5", "ms", " telemetry",
                    " and", " circuit", " breaking", " active", ".",
                ]

                # Limit or pad based on config
                num_tokens = max(1, min(state.tokens_to_generate, len(tokens)))
                selected_tokens = tokens[:num_tokens]

                for i, token_str in enumerate(selected_tokens):
                    # Simulate TPOT delay (autoregressive generation)
                    if state.added_tpot_delay_ms > 0:
                        await asyncio.sleep(state.added_tpot_delay_ms / 1000.0)

                    # Mid-stream fault injection test
                    if simulate_midstream_drop and i >= 2:
                        # Forcibly raise an error to abruptly break the HTTP stream
                        raise RuntimeError(f"Simulated network drop mid-stream on {provider_name}")

                    chunk_data = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model,
                        "choices": [{"index": 0, "delta": {"content": token_str}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n".encode("utf-8")

                # Final termination chunk
                final_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # 3. Non-Streaming Response
        content_text = f"Hello! This is a complete non-streaming response from {provider_name}."
        prompt_tokens = max(5, len(str(messages)) // 4)
        completion_tokens = max(5, len(content_text) // 4)

        resp_payload = {
            "id": f"chatcmpl-mock-{int(time.time()*1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return JSONResponse(content=resp_payload)

    return app
