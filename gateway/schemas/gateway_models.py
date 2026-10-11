"""Internal gateway operational models for telemetry, circuit state, and health checks."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class StreamTelemetryData(BaseModel):
    """Real-time latency profiling and token throughput record for an inference stream."""
    request_id: str
    provider: str
    model: str
    ttft_ms: float = Field(..., description="Time to First Token in milliseconds (T_first - T_start)")
    tpot_ms: float = Field(..., description="Time per Output Token in milliseconds ((T_last - T_first) / (N - 1))")
    total_latency_ms: float = Field(..., description="End-to-end stream duration in milliseconds")
    completion_tokens: int = Field(..., description="Total output tokens counted in the stream")
    prompt_tokens: int = Field(default=0, description="Estimated prompt tokens")
    inter_token_latencies_avg_ms: float = Field(default=0.0)
    accumulated_text: str = Field(default="", description="Reconstructed text output for caching")
    fallback_used: bool = False
    cached: bool = False


class CircuitBreakerStatus(BaseModel):
    """Status snapshot of a single circuit breaker."""
    provider_name: str
    state: Literal["CLOSED", "OPEN", "HALF-OPEN"]
    failure_count: int
    success_count: int
    last_failure_timestamp: Optional[float] = None
    last_state_change_timestamp: float
    recovery_cooldown_remaining_seconds: float = 0.0


class ProviderHealthStatus(BaseModel):
    """Health check details for an upstream provider."""
    provider_name: str
    base_url: str
    is_healthy: bool
    circuit_state: str
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Gateway health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    active_primary_provider: str
    active_fallback_provider: str
    circuit_breakers: Dict[str, CircuitBreakerStatus]
