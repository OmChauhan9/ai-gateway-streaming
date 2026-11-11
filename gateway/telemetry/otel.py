"""OpenTelemetry span for each chat completion request.

Each request gets one SERVER span named `chat <model>`. The gateway sets these attributes on it:
- gen_ai.system is the provider name. After a failover, gateway/main.py sets it to the fallback's name.
- gen_ai.request.model is the model name from the request.
- gen_ai.request.streaming is the request's stream flag.
- gen_ai.usage.prompt_tokens is the gateway's prompt estimate.
- gen_ai.usage.completion_tokens is the completion token count.
- gen_ai.server.ttft and gen_ai.server.tpot are TTFT and TPOT in milliseconds.
- gateway.request_id, gateway.fallback_used, and gateway.cached.
- gateway.client_disconnected, set by gateway/main.py when the client leaves mid-stream.
- error.message, set with an ERROR status when the request fails.

Some of these names differ from the OpenTelemetry GenAI conventions at
https://opentelemetry.io/docs/specs/semconv/gen-ai/. gen_ai.system, gen_ai.usage.prompt_tokens, and
gen_ai.usage.completion_tokens are deprecated. gen_ai.request.streaming, gen_ai.server.ttft, and
gen_ai.server.tpot are custom. The gateway configures no span exporter. The Limitations section of
README.md lists these gaps.
"""

from typing import Any, Dict, Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.trace import Status, StatusCode, Span


class OpenTelemetryGenAIManager:
    """Creates and closes the gateway's request spans."""

    def __init__(self, service_name: str = "ai-gateway-proxy"):
        self.service_name = service_name
        self.resource = Resource.create({SERVICE_NAME: service_name})
        
        # Check if global tracer provider is already configured
        current_provider = trace.get_tracer_provider()
        if isinstance(current_provider, TracerProvider):
            self.tracer_provider = current_provider
        else:
            self.tracer_provider = TracerProvider(resource=self.resource)
            trace.set_tracer_provider(self.tracer_provider)

        self.tracer = trace.get_tracer(service_name, "1.0.0")

    def start_inference_span(
        self,
        request_id: str,
        provider_name: str,
        model_name: str,
        is_streaming: bool = True,
        prompt_tokens: int = 0,
    ) -> Span:
        """Create and start a GenAI client/server span."""
        span_name = f"chat {model_name}"
        span = self.tracer.start_span(span_name, kind=trace.SpanKind.SERVER)

        span.set_attribute("gen_ai.system", provider_name)
        span.set_attribute("gen_ai.request.model", model_name)
        span.set_attribute("gen_ai.request.streaming", is_streaming)
        span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
        span.set_attribute("gateway.request_id", request_id)

        return span

    def record_completion_telemetry(
        self,
        span: Span,
        ttft_ms: float,
        tpot_ms: float,
        completion_tokens: int,
        fallback_used: bool = False,
        cached: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """Enrich and close the GenAI span with performance metrics."""
        try:
            span.set_attribute("gen_ai.server.ttft", ttft_ms)
            span.set_attribute("gen_ai.server.tpot", tpot_ms)
            span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
            span.set_attribute("gateway.fallback_used", fallback_used)
            span.set_attribute("gateway.cached", cached)

            if error is not None:
                span.set_status(Status(StatusCode.ERROR, description=error))
                span.set_attribute("error.message", error)
            else:
                span.set_status(Status(StatusCode.OK))
        finally:
            span.end()


_otel_manager_instance: Optional[OpenTelemetryGenAIManager] = None


def get_otel_manager(service_name: str = "ai-gateway-proxy") -> OpenTelemetryGenAIManager:
    global _otel_manager_instance
    if _otel_manager_instance is None:
        _otel_manager_instance = OpenTelemetryGenAIManager(service_name)
    return _otel_manager_instance
