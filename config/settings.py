"""Gateway configuration module using Pydantic Settings.

Supports environment variable overrides for multi-environment deployments
(Local Dev, Docker, Kubernetes, Production).
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseSettings):
    """Configuration for an upstream LLM provider."""
    name: str = "primary"
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "mock-key"
    model_override: Optional[str] = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    is_active: bool = True


class Settings(BaseSettings):
    """Central gateway configuration."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Server settings
    app_name: str = "Async AI Gateway"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 1

    # Upstream Providers
    # Primary: e.g. Groq / Local Ollama / vLLM
    primary_provider_name: str = "groq"
    primary_base_url: str = "http://127.0.0.1:8001/v1"
    primary_api_key: str = "gsk-mock-primary"
    primary_timeout_seconds: float = 30.0

    # Fallback: e.g. Together AI / OpenAI
    fallback_provider_name: str = "together"
    fallback_base_url: str = "http://127.0.0.1:8002/v1"
    fallback_api_key: str = "together-mock-fallback"
    fallback_timeout_seconds: float = 30.0

    # Synthetic mock upstream for local testing & fault injection
    use_mock_upstream: bool = True
    mock_upstream_port: int = 8001
    mock_fallback_port: int = 8002

    # Circuit Breaker Parameters (Nygard pattern)
    cb_failure_threshold: int = Field(
        default=5,
        description="Number of consecutive failures (5xx, 429, timeout) before tripping OPEN",
    )
    cb_failure_window_seconds: float = Field(
        default=10.0,
        description="Rolling window duration for counting failures",
    )
    cb_recovery_timeout_seconds: float = Field(
        default=15.0,
        description="Duration in seconds before OPEN transitions to HALF-OPEN",
    )
    cb_half_open_success_threshold: int = Field(
        default=2,
        description="Consecutive successful requests required in HALF-OPEN to reset to CLOSED",
    )

    # Token Leaky-Bucket Rate Limiter
    rate_limiter_enabled: bool = True
    rate_limiter_capacity_tokens: int = Field(
        default=50000,
        description="Bucket token capacity (burst maximum)",
    )
    rate_limiter_leak_rate_tokens_per_second: float = Field(
        default=2000.0,
        description="Leak rate in tokens per second (steady-state rate)",
    )

    # Semantic / Exact Cache
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_size: int = 1000

    # Telemetry & Observability
    otel_service_name: str = "ai-gateway-proxy"
    otel_exporter_endpoint: Optional[str] = None
    prometheus_metrics_enabled: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
