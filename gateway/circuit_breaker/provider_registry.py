"""Provider Registry & Dynamic Upstream Failover Router.

Maintains health records and circuit breakers for all configured upstream LLM endpoints.
Coordinates transparent failover from Primary to Secondary providers.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from gateway.circuit_breaker.state_machine import CircuitBreaker, CircuitState


class AllProvidersUnavailableException(Exception):
    """Raised when all configured LLM providers are failing or have OPEN circuits."""
    def __init__(self, message: str = "All upstream model providers are currently unavailable."):
        super().__init__(message)


@dataclass
class ProviderInfo:
    """Operational metadata for an upstream provider."""
    name: str
    base_url: str
    api_key: str
    circuit_breaker: CircuitBreaker
    is_primary: bool = False
    timeout_seconds: float = 30.0


class ProviderRegistry:
    """Central registry of LLM providers with automatic health-based routing."""

    def __init__(
        self,
        failure_threshold: int = 5,
        failure_window_seconds: float = 10.0,
        recovery_timeout_seconds: float = 15.0,
        half_open_success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window_seconds
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self._providers: Dict[str, ProviderInfo] = {}
        self.primary_name: Optional[str] = None
        self.fallback_name: Optional[str] = None

    def register_provider(
        self,
        name: str,
        base_url: str,
        api_key: str,
        is_primary: bool = False,
        timeout_seconds: float = 30.0,
    ) -> ProviderInfo:
        """Register an LLM provider and initialize its dedicated circuit breaker."""
        cb = CircuitBreaker(
            name=name,
            failure_threshold=self.failure_threshold,
            failure_window_seconds=self.failure_window,
            recovery_timeout_seconds=self.recovery_timeout,
            half_open_success_threshold=self.half_open_success_threshold,
        )
        info = ProviderInfo(
            name=name,
            base_url=base_url.rstrip("/"),
            api_key=api_key,
            circuit_breaker=cb,
            is_primary=is_primary,
            timeout_seconds=timeout_seconds,
        )
        self._providers[name] = info
        if is_primary:
            self.primary_name = name
        elif not self.fallback_name:
            self.fallback_name = name

        return info

    def get_provider(self, name: str) -> Optional[ProviderInfo]:
        return self._providers.get(name)

    async def select_provider_for_request(self) -> Tuple[ProviderInfo, bool]:
        """Select healthy provider to service the request.
        
        Returns:
            Tuple[ProviderInfo, bool]: (selected_provider, is_fallback_diverted)
            
        Raises:
            AllProvidersUnavailableException: If no provider can take traffic.
        """
        # 1. Inspect Primary Provider
        if self.primary_name and self.primary_name in self._providers:
            primary = self._providers[self.primary_name]
            allowed, state, cooldown = await primary.circuit_breaker.can_execute()
            if allowed:
                return primary, False

        # 2. Primary is OPEN or missing -> Evaluate Fallback Provider
        if self.fallback_name and self.fallback_name in self._providers:
            fallback = self._providers[self.fallback_name]
            allowed, state, cooldown = await fallback.circuit_breaker.can_execute()
            if allowed:
                return fallback, True

        # 3. Both are unavailable
        raise AllProvidersUnavailableException(
            "Primary circuit is OPEN and Fallback provider is also unavailable."
        )

    async def get_all_status(self) -> Dict[str, any]:
        """Snapshot status of all providers and circuit breakers."""
        status = {}
        for name, provider in self._providers.items():
            metrics = await provider.circuit_breaker.get_metrics()
            status[name] = {
                "base_url": provider.base_url,
                "is_primary": provider.is_primary,
                "circuit": metrics,
            }
        return status
