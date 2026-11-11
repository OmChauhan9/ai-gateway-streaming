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
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response as PlainResponse, StreamingResponse

from config.settings import Settings, get_settings
from gateway.cache.semantic_cache import SemanticCache
from gateway.circuit_breaker.provider_registry import (
    AllProvidersUnavailableException,
    ProviderInfo,
    ProviderRegistry,
)
from gateway.limiter.leaky_bucket import RateLimitExceeded, RequestExceedsCapacity, TokenLeakyBucketLimiter
from gateway.limiter.tokenizer_estimator import estimate_request_tokens, estimate_text_tokens
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
    UpstreamAuthError,
    UpstreamClientError,
    UpstreamConnectionError,
    UpstreamRateLimitedError,
    UpstreamServerError,
)

PROVIDER_FAILURES = (UpstreamConnectionError, UpstreamServerError, UpstreamRateLimitedError, UpstreamAuthError)


class GatewayContext:
    """Dependency container holding lifecycle singletons."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.metrics = get_metrics_registry()
        self.registry = ProviderRegistry(
            failure_threshold=settings.cb_failure_threshold,
            failure_window_seconds=settings.cb_failure_window_seconds,
            recovery_timeout_seconds=settings.cb_recovery_timeout_seconds,
            half_open_success_threshold=settings.cb_half_open_success_threshold,
            on_transition=self.metrics.on_circuit_transition,
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

    @app.exception_handler(HTTPException)
    async def openai_error_body(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "error" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
        return await http_exception_handler(request, exc)

    @app.exception_handler(UpstreamClientError)
    async def relay_upstream_client_error(request: Request, exc: UpstreamClientError):
        try:
            return JSONResponse(status_code=exc.status_code, content=json.loads(exc.message))
        except ValueError:
            return PlainResponse(content=exc.message, status_code=exc.status_code)

    # -------------------------------------------------------------------------
    # Ingress Helper
    # -------------------------------------------------------------------------
    def _extract_client_key(request: Request, authorization: Optional[str]) -> str:
        """Extract client identifier from Authorization header or remote IP."""
        if authorization and authorization.startswith("Bearer "):
            return authorization[7:].strip()
        client_host = request.client.host if request.client else "unknown_client"
        return client_host

    async def _reject_non_finite_numbers(request: Request) -> None:
        # Python's JSON parser accepts NaN, Infinity and 1e999, but no upstream does and the
        # upstream encoder refuses them. As a dependency this runs before body validation,
        # whose 422 would echo the value and fail to render.
        try:
            body = await request.json()
        except ValueError:
            return
        try:
            json.dumps(body, allow_nan=False)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": {
                        "message": "Request body contains NaN or Infinity, which is not valid JSON.",
                        "type": "invalid_request_error",
                    }
                },
            )

    def _cannot_serve(last_exc: Optional[Exception], status_code: int, detail) -> HTTPException:
        # The gateway calls providers with its own key, so a 401/403 means the gateway is
        # misconfigured, and the provider's body is not the caller's to see.
        if isinstance(last_exc, UpstreamAuthError):
            status_code = status.HTTP_502_BAD_GATEWAY
            detail = {
                "error": {
                    "message": (
                        f"Provider '{last_exc.provider}' rejected the gateway's provider credentials "
                        f"(HTTP {last_exc.status_code})."
                    ),
                    "type": "upstream_auth_failed",
                }
            }
        return HTTPException(status_code=status_code, detail=detail)

    async def _fail_over(send, payload: ChatCompletionRequest, primary_exc: Exception, span):
        """After the primary failed, serve from the fallback or raise a 503 (502 on a provider auth rejection).

        Returns the fallback, its permit to settle, and the result of `send`.
        """
        fallback = ctx.registry.get_provider(ctx.registry.fallback_name)
        permit = fallback.circuit_breaker.try_acquire()
        last_exc = primary_exc
        if permit is None:
            message = f"Primary failed and fallback circuit is OPEN: {primary_exc}"
        else:
            span.set_attribute("gen_ai.system", fallback.name)
            try:
                return fallback, permit, await send(fallback, payload)
            except PROVIDER_FAILURES as fb_exc:
                last_exc = fb_exc
                permit.fail()
                message = f"Both primary and fallback failed: {fb_exc}"
            except BaseException as exc:
                permit.abandon()
                ctx.otel.record_completion_telemetry(span, 0, 0, 0, fallback_used=True, error=str(exc))
                raise
        ctx.otel.record_completion_telemetry(span, 0, 0, 0, fallback_used=permit is not None, error=message)
        raise _cannot_serve(
            last_exc,
            status.HTTP_503_SERVICE_UNAVAILABLE,
            {"error": {"message": message, "type": "all_providers_unavailable"}},
        ) from last_exc

    # -------------------------------------------------------------------------
    # Core Endpoint: POST /v1/chat/completions
    # -------------------------------------------------------------------------
    @app.post("/v1/chat/completions", dependencies=[Depends(_reject_non_finite_numbers)])
    @app.post("/chat/completions", dependencies=[Depends(_reject_non_finite_numbers)])
    async def chat_completions(
        payload: ChatCompletionRequest,
        request: Request,
        authorization: Optional[str] = Header(None),
    ):
        req_start_time = time.perf_counter()
        try:
            return await _serve_chat(payload, request, authorization, req_start_time)
        except (HTTPException, UpstreamClientError) as exc:
            # An error response caused by a provider's failure is counted under that provider.
            cause = exc if isinstance(exc, UpstreamClientError) else exc.__cause__
            ctx.metrics.record_request_metrics(
                provider=getattr(cause, "provider", "none"),
                model=payload.model,
                status=str(exc.status_code),
                is_stream=payload.stream,
                ttft_ms=0.0,
                tpot_ms=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                total_latency_ms=(time.perf_counter() - req_start_time) * 1000.0,
            )
            raise

    async def _serve_chat(
        payload: ChatCompletionRequest,
        request: Request,
        authorization: Optional[str],
        req_start_time: float,
    ):
        req_id = f"gw-{uuid.uuid4().hex[:12]}"
        client_key = _extract_client_key(request, authorization)

        # 1. Ingress Validation & Leaky-Bucket Rate Limiting
        prompt_tokens = estimate_request_tokens(payload)
        if ctx.settings.rate_limiter_enabled:
            try:
                await ctx.limiter.acquire(client_key, prompt_tokens)
            except RequestExceedsCapacity as rec:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "error": {
                            "message": (
                                f"Request needs an estimated {rec.requested} tokens, more than the "
                                f"token capacity of {rec.capacity}. Retrying will not help; shorten the prompt."
                            ),
                            "type": "request_exceeds_token_capacity",
                        }
                    },
                )
            except RateLimitExceeded as rle:
                ctx.metrics.record_rate_limit()
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
            response_text, finish_reason, usage_info = cached_result
            ctx.metrics.record_cache_hit("exact")

            if payload.stream:
                # Deliver cached response via instant SSE stream
                async def cached_stream() -> AsyncIterator[bytes]:
                    chunk = {
                        "id": f"chatcmpl-cache-{req_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": payload.model,
                        "choices": [{"index": 0, "delta": {"content": response_text}, "finish_reason": finish_reason}],
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
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": response_text}, "finish_reason": finish_reason}],
                        "usage": usage_info,
                    },
                    headers={"X-Cache": "HIT", "X-Gateway-Request-Id": req_id},
                )

        cache_status_headers = {"X-Cache": "MISS"} if ctx.cache.enabled else {}

        # 3. Circuit Breaker & Provider Selection
        try:
            target_provider, permit, is_fallback = ctx.registry.admit()
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
        active_provider = target_provider
        fallback_diverted = is_fallback

        if payload.stream:
            try:
                resp = await ctx.client.send_streaming_request(active_provider, payload)
            except PROVIDER_FAILURES as exc:
                # Primary failed before headers committed! Trip circuit & failover
                permit.fail()

                # Attempt transparent fallback if available
                if not fallback_diverted and ctx.registry.fallback_name:
                    active_provider, permit, resp = await _fail_over(
                        ctx.client.send_streaming_request, payload, exc, span
                    )
                    fallback_diverted = True
                else:
                    ctx.otel.record_completion_telemetry(
                        span, 0, 0, 0, fallback_used=fallback_diverted, error=str(exc)
                    )
                    raise _cannot_serve(
                        exc,
                        status.HTTP_502_BAD_GATEWAY,
                        {"error": {"message": f"Upstream failure: {str(exc)}", "type": "upstream_error"}},
                    ) from exc
            except BaseException as exc:
                permit.abandon()
                ctx.otel.record_completion_telemetry(
                    span, 0, 0, 0, fallback_used=fallback_diverted, error=str(exc)
                )
                raise

            # 5. Non-Blocking SSE Streaming Interceptor
            accumulated_tokens: list[str] = []

            async def on_stream_complete(telemetry: StreamTelemetryData):
                # Upstream succeeded!
                permit.succeed()

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
                        telemetry.finish_reason,
                        usage={"prompt_tokens": prompt_tokens, "completion_tokens": telemetry.completion_tokens, "total_tokens": actual_total},
                    )

            async def on_stream_error(exc: Exception):
                # Mid-stream error
                permit.fail()
                ctx.metrics.record_request_metrics(
                    provider=active_provider.name,
                    model=payload.model,
                    status="502",
                    is_stream=True,
                    ttft_ms=0.0,
                    tpot_ms=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_latency_ms=(time.perf_counter() - req_start_time) * 1000.0,
                )
                ctx.otel.record_completion_telemetry(
                    span=span,
                    ttft_ms=0,
                    tpot_ms=0,
                    completion_tokens=0,
                    fallback_used=fallback_diverted,
                    error=str(exc),
                )

            async def on_client_disconnect(completion_tokens: int):
                # Says nothing about provider health, so the circuit breaker records no outcome.
                ctx.metrics.record_request_metrics(
                    provider=active_provider.name,
                    model=payload.model,
                    status="499",
                    is_stream=True,
                    ttft_ms=0.0,
                    tpot_ms=0.0,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_latency_ms=(time.perf_counter() - req_start_time) * 1000.0,
                )
                span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
                span.set_attribute("gateway.fallback_used", fallback_diverted)
                span.set_attribute("gateway.client_disconnected", True)
                span.end()
                await ctx.limiter.release_or_adjust(client_key, prompt_tokens, prompt_tokens + completion_tokens)

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
                t_start=req_start_time,
                prompt_tokens=prompt_tokens,
                fallback_used=fallback_diverted,
                on_stream_complete=on_stream_complete,
                on_stream_error=on_stream_error,
                on_client_disconnect=on_client_disconnect,
            )

            async def settled_body() -> AsyncIterator[bytes]:
                # A body cut short, as by a client disconnect, gives the permit back without a verdict.
                try:
                    async for chunk in interceptor:
                        yield chunk
                finally:
                    permit.abandon()

            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Gateway-Request-Id": req_id,
                "X-Gateway-Provider": active_provider.name,
                "X-Gateway-Fallback": str(fallback_diverted).lower(),
                **cache_status_headers,
            }

            return StreamingResponse(settled_body(), media_type="text/event-stream", headers=headers)

        else:
            # Non-streaming Request
            try:
                data = await ctx.client.send_non_streaming_request(active_provider, payload)
            except PROVIDER_FAILURES as exc:
                permit.fail()
                if not fallback_diverted and ctx.registry.fallback_name:
                    active_provider, permit, data = await _fail_over(
                        ctx.client.send_non_streaming_request, payload, exc, span
                    )
                    fallback_diverted = True
                else:
                    ctx.otel.record_completion_telemetry(
                        span, 0, 0, 0, fallback_used=fallback_diverted, error=str(exc)
                    )
                    raise _cannot_serve(
                        exc,
                        status.HTTP_502_BAD_GATEWAY,
                        {"error": {"message": f"Upstream error: {str(exc)}", "type": "upstream_error"}},
                    ) from exc
            except BaseException as exc:
                permit.abandon()
                ctx.otel.record_completion_telemetry(
                    span, 0, 0, 0, fallback_used=fallback_diverted, error=str(exc)
                )
                raise
            permit.succeed()

            total_duration_ms = (time.perf_counter() - req_start_time) * 1000.0
            # Providers may send `"usage": null`, or null counts inside it.
            usage = data.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            comp_tokens = usage.get("completion_tokens")
            if not isinstance(comp_tokens, int):
                comp_tokens = 0

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

            await ctx.limiter.release_or_adjust(client_key, prompt_tokens, prompt_tokens + comp_tokens)

            choices = data["choices"]
            content_text = choices[0]["message"].get("content") if choices else ""
            if content_text:
                await ctx.cache.set(payload, content_text, choices[0].get("finish_reason"), usage)

            headers = {
                "X-Gateway-Request-Id": req_id,
                "X-Gateway-Provider": active_provider.name,
                "X-Gateway-Fallback": str(fallback_diverted).lower(),
                **cache_status_headers,
            }
            return JSONResponse(content=data, headers=headers)

    # -------------------------------------------------------------------------
    # Observability & Administration Endpoints
    # -------------------------------------------------------------------------
    @app.get("/metrics")
    async def metrics():
        """Prometheus scrape endpoint."""
        # Reading the breakers applies any cooldown that ran out since the last request, so the gauge is current.
        await ctx.registry.get_all_status()
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
        return {"status": "tripped", "provider": provider_name, "state": "OPEN"}

    @app.post("/v1/circuit-breakers/{provider_name}/reset")
    async def reset_circuit_breaker(provider_name: str):
        """Reset a circuit breaker to CLOSED state."""
        prov = ctx.registry.get_provider(provider_name)
        if not prov:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        await prov.circuit_breaker.reset_manually()
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
