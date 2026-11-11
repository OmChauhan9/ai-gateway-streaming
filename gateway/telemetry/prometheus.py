"""Prometheus Metrics Exporter for AI Gateway.

Defines the Prometheus metrics that GET /metrics serves:
- Request counter by provider, model, status, and stream flag
- LLM Time To First Token (TTFT) histograms
- LLM Time Per Output Token (TPOT) histograms
- Total prompt and completion token counters
- Tri-state circuit breaker gauges and trip counters
- Gateway latency histogram
- Token rate limiter exhaustion counters
- Cache hit counter
"""

from typing import Optional
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from gateway.circuit_breaker.state_machine import CircuitState

_CIRCUIT_STATE_VALUES = {CircuitState.CLOSED: 0.0, CircuitState.HALF_OPEN: 1.0, CircuitState.OPEN: 2.0}


class PrometheusMetricsRegistry:
    """Manages all Prometheus metrics collected by the gateway."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()

        # Request counter
        self.requests_total = Counter(
            "llm_requests_total",
            "Chat completion responses by the HTTP status the client received. A stream that fails after "
            "its headers counts as 502, the code in its error event, and a client that disconnects mid-stream "
            "counts as 499. Cache hits and requests rejected "
            "before the handler (400, 422) are not counted.",
            ["provider", "model", "status", "stream"],
            registry=self.registry,
        )

        # TTFT Histogram (seconds)
        # Buckets calibrated for modern LLM inference: 20ms to 5 seconds
        self.ttft_seconds = Histogram(
            "llm_ttft_seconds",
            "Time to First Token (TTFT) in seconds",
            ["provider", "model"],
            buckets=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0],
            registry=self.registry,
        )

        # TPOT Histogram (seconds per token)
        # Buckets calibrated for token generation rates (5ms to 250ms per token = 4 to 200 tokens/sec)
        self.tpot_seconds = Histogram(
            "llm_tpot_seconds",
            "Time Per Output Token (TPOT) in seconds",
            ["provider", "model"],
            buckets=[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.25],
            registry=self.registry,
        )

        # Token Counters
        self.tokens_total = Counter(
            "llm_tokens_total",
            "Total tokens processed across requests",
            ["provider", "model", "type"],  # type: prompt | completion
            registry=self.registry,
        )

        # End-to-end Gateway Latency Histogram
        self.gateway_latency_seconds = Histogram(
            "llm_gateway_latency_seconds",
            "End-to-end gateway request latency in seconds",
            ["provider", "status"],
            buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0],
            registry=self.registry,
        )

        # Circuit Breaker Gauge: 0 = CLOSED, 1 = HALF-OPEN, 2 = OPEN
        self.circuit_state = Gauge(
            "circuit_breaker_state",
            "Current state of provider circuit breaker (0=CLOSED, 1=HALF-OPEN, 2=OPEN)",
            ["provider"],
            registry=self.registry,
        )

        # Circuit Breaker Trip Counter
        self.circuit_trips_total = Counter(
            "circuit_breaker_trips_total",
            "Total times circuit breaker tripped to OPEN state",
            ["provider"],
            registry=self.registry,
        )

        # Rate Limiting Counter
        self.rate_limit_exceeded_total = Counter(
            "llm_rate_limit_exceeded_total",
            "Total requests rejected by Token Leaky-Bucket Limiter",
            registry=self.registry,
        )

        # Cache Hits Counter
        self.cache_hits_total = Counter(
            "llm_cache_hits_total",
            "Total requests served from cache",
            ["cache_type"],
            registry=self.registry,
        )

    def record_request_metrics(
        self,
        provider: str,
        model: str,
        status: str,
        is_stream: bool,
        ttft_ms: float,
        tpot_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        total_latency_ms: float,
    ) -> None:
        """Update Prometheus metric points for a completed inference call."""
        stream_label = "true" if is_stream else "false"
        self.requests_total.labels(
            provider=provider, model=model, status=status, stream=stream_label
        ).inc()

        if is_stream and ttft_ms > 0:
            self.ttft_seconds.labels(provider=provider, model=model).observe(ttft_ms / 1000.0)

        if is_stream and tpot_ms > 0:
            self.tpot_seconds.labels(provider=provider, model=model).observe(tpot_ms / 1000.0)

        if prompt_tokens > 0:
            self.tokens_total.labels(provider=provider, model=model, type="prompt").inc(prompt_tokens)

        if completion_tokens > 0:
            self.tokens_total.labels(provider=provider, model=model, type="completion").inc(completion_tokens)

        self.gateway_latency_seconds.labels(provider=provider, status=status).observe(total_latency_ms / 1000.0)

    def on_circuit_transition(self, provider: str, old: Optional[CircuitState], new: CircuitState) -> None:
        """Circuit breaker listener, and the only writer of the circuit metrics.

        `old` is None for the state a breaker starts in, which publishes both series at startup.
        """
        self.circuit_state.labels(provider=provider).set(_CIRCUIT_STATE_VALUES[new])
        trips = self.circuit_trips_total.labels(provider=provider)
        if new == CircuitState.OPEN:
            trips.inc()

    def record_rate_limit(self) -> None:
        self.rate_limit_exceeded_total.inc()

    def record_cache_hit(self, cache_type: str = "exact") -> None:
        self.cache_hits_total.labels(cache_type=cache_type).inc()

    def scrape(self) -> bytes:
        """Generate Prometheus exposition text format."""
        return generate_latest(self.registry)


_metrics_registry_instance: Optional[PrometheusMetricsRegistry] = None


def get_metrics_registry() -> PrometheusMetricsRegistry:
    global _metrics_registry_instance
    if _metrics_registry_instance is None:
        _metrics_registry_instance = PrometheusMetricsRegistry()
    return _metrics_registry_instance
