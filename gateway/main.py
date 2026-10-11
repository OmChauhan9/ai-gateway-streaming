"""Async AI Gateway with Streaming Telemetry & Circuit Breaking.

Production-grade inference proxy providing:
1. Token Leaky-Bucket Rate Limiter (prompt token estimation + reconciliation)
2. Semantic / Exact Cache Check (instant hit delivery)
3. Tri-State Circuit Breakers with Transparent Multi-Provider Failover
4. Non-Blocking SSE Streaming Interceptor (real-time TTFT & TPOT profiling)
5. OpenTelemetry GenAI Semantic Conventions & Prometheus Observability
"""

import asyncio
from contextlib import asynccontextmanager
import json
import time
import uuid
from typing import AsyncIterator, Optional
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response as PlainResponse, StreamingResponse

from config.settings import Settings, get_settings
from gateway.cache.semantic_cache import SemanticCache
from gateway.circuit_breaker.provider_registry import (
    AllProvidersUnavailableException,
    ProviderInfo,
    ProviderRegistry,
)
from gateway.circuit_breaker.state_machine import CircuitState
from gateway.limiter.leaky_bucket import RateLimitExceeded, TokenLeakyBucketLimiter
from gateway.limiter.tokenizer_estimator import estimate_prompt_tokens, estimate_text_tokens
from gateway.schemas.gateway_models import HealthResponse, StreamTelemetryData
from gateway.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    UsageInfo,
)
from gateway.stream.interceptor import telemetry_stream_interceptor
from gateway.telemetry.otel import OpenTelemetryGenAIManager, get_otel_manager
from gateway.telemetry.prometheus import (
    CONTENT_TYPE_LATEST,
    PrometheusMetricsRegistry,
    get_metrics_registry,
)
from gateway.upstream.client import (
    AsyncUpstreamClient,
    UpstreamConnectionError,
    UpstreamRateLimitedError,
    UpstreamServerError,
)


class GatewayContext:
    """Dependency container holding lifecycle singletons."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.registry = ProviderRegistry(
            failure_threshold=settings.cb_failure_threshold,
            failure_window_seconds=settings.cb_failure_window_seconds,
            recovery_timeout_seconds=settings.cb_recovery_timeout_seconds,
            half_open_success_threshold=settings.cb_half_open_success_threshold,
        )
        self.limiter = TokenLeakyBucketLimiter(
            capacity_tokens=settings.rate_limiter_capacity_tokens,
            leak_rate_tokens_per_second=settings.rate_limiter_leak_rate_tokens_per_second,
        )
        self.cache = SemanticCache(
            ttl_seconds=settings.cache_ttl_seconds,
            max_size=settings.cache_max_size,
            enabled=settings.cache_enabled,
        )
        self.otel = get_otel_manager(settings.otel_service_name)
        self.metrics = get_metrics_registry()
        self.client = AsyncUpstreamClient()


def create_app(settings: Optional[Settings] = None, client: Optional[AsyncUpstreamClient] = None) -> FastAPI:
    """Factory function for FastAPI Gateway application."""
    current_settings = settings or get_settings()
    ctx = GatewayContext(current_settings)
    if client:
        ctx.client = client

    # Register providers
    ctx.registry.register_provider(
        name=current_settings.primary_provider_name,
        base_url=current_settings.primary_base_url,
        api_key=current_settings.primary_api_key,
        is_primary=True,
        timeout_seconds=current_settings.primary_timeout_seconds,
    )
    ctx.registry.register_provider(
        name=current_settings.fallback_provider_name,
        base_url=current_settings.fallback_base_url,
        api_key=current_settings.fallback_api_key,
        is_primary=False,
        timeout_seconds=current_settings.fallback_timeout_seconds,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        yield
        # Shutdown
        await ctx.client.close()

    app = FastAPI(
        title=current_settings.app_name,
        version=current_settings.version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach context to app state
    app.state.ctx = ctx

    # -------------------------------------------------------------------------
    # Ingress Helper
    # -------------------------------------------------------------------------
    def _extract_client_key(request: Request, authorization: Optional[str]) -> str:
        """Extract client identifier from Authorization header or remote IP."""
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:].strip()
        client_host = request.client.host if request.client else "unknown_client"
        return client_host

    # -------------------------------------------------------------------------
    # Core Endpoint: POST /v1/chat/completions
    # -------------------------------------------------------------------------
    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        req_start_time = time.perf_counter()
        req_id = f"gw-{uuid.uuid4().hex[:12]}"
        client_key = _extract_client_key(request, authorization)

        # 1. Ingress Validation & Leaky-Bucket Rate Limiting
        prompt_tokens = estimate_prompt_tokens(payload.messages)
        if ctx.settings.rate_limiter_enabled:
            try:
                await ctx.limiter.acquire(client_key, prompt_tokens)
            except RateLimitExceeded as rle:
                ctx.metrics.record_rate_limit(client_key)
                headers = {"Retry-After": str(int(rle.retry_after) + 1)}
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": {
                            "message": f"Token rate limit exceeded. Retry after {rle.retry_after:.2f} seconds.",
                            "type": "tokens_per_second_exceeded",
                            "retry_after_seconds": round(rle.retry_after, 2),
                        }
                    },
                    headers=headers,
                )

        # 2. Semantic / Exact Cache Check
        cached_result = await ctx.cache.get(payload)
        if cached_result:
            response_text, usage_info = cached_result
            ctx.metrics.record_cache_hit("exact")

            if payload.stream:
                # Deliver cached response via instant SSE stream
                async def cached_stream() -> AsyncIterator[bytes]:
                    chunk = {
                        "id": f"chatcmpl-cache-{req_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": payload.model,
                        "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": "stop"}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    cached_stream(),
                    media_type="text/event-stream",
                    headers={"X-Cache": "HIT", "X-Gateway-Request-Id": req_id},
                )
            else:
                return JSONResponse(
                    content={
                        "id": f"chatcmpl-cache-{req_id}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": payload.model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": "stop"}],
                        "usage": usage_info,
                    },
                    headers={"X-Cache": "HIT", "X-Gateway-Request-Id": req_id},
                )

        # 3. Circuit Breaker & Provider Selection
        try:
            target_provider, is_fallback = await ctx.registry.select_provider_for_request()
        except AllProvidersUnavailableException as apue:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"error": {"message": str(apue), "type": "all_providers_unavailable"}},
            )

        # Start OpenTelemetry GenAI Span
        span = ctx.otel.start_inference_span(
            request_id=req_id,
            provider_name=target_provider.name,
            model_name=payload.model,
            is_streaming=payload.stream,
            prompt_tokens=prompt_tokens,
        )

        # 4. Upstream Dispatch with Transparent Failover
        resp = None
        t_upstream_start = time.perf_counter()
        active_provider = target_provider
        fallback_diverted = is_fallback

        if payload.stream:
            try:
                resp, t_upstream_start = await ctx.client.send_streaming_request(active_provider, payload)
            except (UpstreamConnectionError, UpstreamServerError, UpstreamRateLimitedError) as exc:
                # Primary failed before headers committed! Trip circuit & failover
                await active_provider.circuit_breaker.record_failure(exc)
                ctx.metrics.set_circuit_state(
                    active_provider.name, active_provider.circuit_breaker.state.value
                )
                if active_provider.circuit_breaker.state == CircuitState.OPEN:
                    ctx.metrics.record_circuit_trip(active_provider.name)

                # Attempt transparent fallback if available
                if not fallback_diverted and ctx.registry.fallback_name:
                    fallback_prov = ctx.registry.get_provider(ctx.registry.fallback_name)
                    if fallback_prov:
                        allowed, _, _ = await fallback_prov.circuit_breaker.can_execute()
                        if allowed:
                            try:
                                active_provider = fallback_prov
                                fallback_diverted = True
                                resp, t_upstream_start = await ctx.client.send_streaming_request(active_provider, payload)
                            except Exception as fb_exc:
                                await fallback_prov.circuit_breaker.record_failure(fb_exc)
                                ctx.otel.record_completion_telemetry(
                                    span, 0, 0, 0, fallback_used=True, error=str(fb_exc)
                                )
                                raise HTTPException(
                                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                    detail={"error": {"message": f"Both primary and fallback failed: {str(fb_exc)}"}},
                                )
                else:
                    ctx.otel.record_completion_telemetry(
                        span, 0, 0, 0, fallback_used=fallback_diverted, error=str(exc)
                    )
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail={"error": {"message": f"Upstream failure: {str(exc)}"}},
                    )

            # 5. Non-Blocking SSE Streaming Interceptor
            accumulated_tokens: list[str] = []

            async def on_stream_complete(telemetry: StreamTelemetryData):
                # Upstream succeeded!
                await active_provider.circuit_breaker.record_success()
                ctx.metrics.set_circuit_state(
                    active_provider.name, active_provider.circuit_breaker.state.value
                )

                # Prometheus metrics
                ctx.metrics.record_request_metrics(
                    provider=active_provider.name,
                    model=payload.model,
                    status="200",
                    is_stream=True,
                    ttft_ms=telemetry.ttft_ms,
                    tpot_ms=telemetry.tpot_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=telemetry.completion_tokens,
                    total_latency_ms=telemetry.total_latency_ms,
                )

                # OpenTelemetry GenAI Span enrichment
                ctx.otel.record_completion_telemetry(
                    span=span,
                    ttft_ms=telemetry.ttft_ms,
                    tpot_ms=telemetry.tpot_ms,
                    completion_tokens=telemetry.completion_tokens,
                    fallback_used=fallback_diverted,
                    cached=False,
                )

                # Rate limiter token reconciliation
                actual_total = prompt_tokens + telemetry.completion_tokens
                await ctx.limiter.release_or_adjust(client_key, prompt_tokens, actual_total)

                # Save response to cache if populated
                if telemetry.accumulated_text:
                    await ctx.cache.set(
                        payload,
                        telemetry.accumulated_text,
                        usage={"prompt_tokens": prompt_tokens, "completion_tokens": telemetry.completion_tokens, "total_tokens": actual_total},
                    )

            async def on_stream_error(exc: Exception):
                # Mid-stream error
                await active_provider.circuit_breaker.record_failure(exc)
                ctx.metrics.set_circuit_state(
                    active_provider.name, active_provider.circuit_breaker.state.value
                )
                ctx.otel.record_completion_telemetry(
                    span=span,
                    ttft_ms=0,
                    tpot_ms=0,
                    completion_tokens=0,
                    fallback_used=fallback_diverted,
                    error=str(exc),
                )

            # Upstream byte stream generator
            async def upstream_byte_iter() -> AsyncIterator[bytes]:
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()

            interceptor = telemetry_stream_interceptor(
                upstream_stream=upstream_byte_iter(),
                request_id=req_id,
                provider_name=active_provider.name,
                model_name=payload.model,
                t_start=t_upstream_start,
                prompt_tokens=prompt_tokens,
                fallback_used=fallback_diverted,
                on_stream_complete=on_stream_complete,
                on_stream_error=on_stream_error,
            )

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Gateway-Request-Id": req_id,
                "X-Gateway-Provider": active_provider.name,
                "X-Gateway-Fallback": str(fallback_diverted).lower(),
            }

            return StreamingResponse(interceptor, media_type="text/event-stream", headers=headers)

        else:
            # Non-streaming Request
            try:
                data, t_upstream_start = await ctx.client.send_non_streaming_request(active_provider, payload)
            except (UpstreamConnectionError, UpstreamServerError, UpstreamRateLimitedError) as exc:
                await active_provider.circuit_breaker.record_failure(exc)
                if not fallback_diverted and ctx.registry.fallback_name:
                    fallback_prov = ctx.registry.get_provider(ctx.registry.fallback_name)
                    if fallback_prov:
                        active_provider = fallback_prov
                        fallback_diverted = True
                        data, t_upstream_start = await ctx.client.send_non_streaming_request(active_provider, payload)
                    else:
                        raise HTTPException(status_code=502, detail=f"Upstream error: {str(exc)}")
                else:
                    raise HTTPException(status_code=502, detail=f"Upstream error: {str(exc)}")

            now = time.perf_counter()
            total_duration_ms = (now - t_upstream_start) * 1000.0
            usage = data.get("usage", {})
            comp_tokens = usage.get("completion_tokens", 0)

            await active_provider.circuit_breaker.record_success()

            ctx.metrics.record_request_metrics(
                provider=active_provider.name,
                model=payload.model,
                status="200",
                is_stream=False,
                ttft_ms=0.0,
                tpot_ms=0.0,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
                total_latency_ms=total_duration_ms,
            )

            ctx.otel.record_completion_telemetry(
                span=span,
                ttft_ms=0.0,
                tpot_ms=0.0,
                completion_tokens=comp_tokens,
                fallback_used=fallback_diverted,
                cached=False,
            )

            choices = data.get("choices", [])
            content_text = choices[0]["message"]["content"] if choices else ""
            if content_text:
                await ctx.cache.set(payload, content_text, usage)

            headers = {
                "X-Gateway-Request-Id": req_id,
                "X-Gateway-Provider": active_provider.name,
                "X-Gateway-Fallback": str(fallback_diverted).lower(),
            }
            return JSONResponse(content=data, headers=headers)

    # -------------------------------------------------------------------------
    # Observability & Administration Endpoints
    # -------------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics():
        """Prometheus scrape endpoint."""
        body = ctx.metrics.scrape()
        return PlainResponse(content=body, media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    async def health():
        """Gateway health status check."""
        status_map = await ctx.registry.get_all_status()
        is_healthy = any(
            info["circuit"]["state"] != "OPEN"
            for info in status_map.values()
        )
        return {
            "status": "healthy" if is_healthy else "degraded",
            "app": ctx.settings.app_name,
            "version": ctx.settings.version,
            "providers": status_map,
        }

    @app.get("/v1/circuit-breakers")
    async def get_circuit_breakers():
        """Retrieve telemetry and operational state for all circuit breakers."""
        return await ctx.registry.get_all_status()

    @app.post("/v1/circuit-breakers/{provider_name}/trip")
    async def trip_circuit_breaker(provider_name: str):
        """Force a circuit breaker to OPEN state (for testing & maintenance)."""
        prov = ctx.registry.get_provider(provider_name)
        if not prov:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        await prov.circuit_breaker.trip_manually()
        ctx.metrics.set_circuit_state(provider_name, "OPEN")
        return {"status": "tripped", "provider": provider_name, "state": "OPEN"}

    @app.post("/v1/circuit-breakers/{provider_name}/reset")
    async def reset_circuit_breaker(provider_name: str):
        """Reset a circuit breaker to CLOSED state."""
        prov = ctx.registry.get_provider(provider_name)
        if not prov:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        await prov.circuit_breaker.reset_manually()
        ctx.metrics.set_circuit_state(provider_name, "CLOSED")
        return {"status": "reset", "provider": provider_name, "state": "CLOSED"}

    @app.get("/v1/rate-limiter/status")
    async def get_rate_limiter_status(
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        """Inspect token leaky-bucket level for current client."""
        key = _extract_client_key(request, authorization)
        status_dict = await ctx.limiter.get_status(key)
        return {"client_key": key, "bucket": status_dict}

    @app.post("/v1/cache/clear")
    async def clear_cache():
        """Purge semantic/exact cache."""
        ctx.cache.clear()
        return {"status": "cache_cleared"}

    return app


# Default application instance for uvicorn
app = create_app()
