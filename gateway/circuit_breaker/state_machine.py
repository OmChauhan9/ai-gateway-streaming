"""Circuit Breaker State Machine (Michael Nygard Pattern).

Protects the gateway and upstream providers from cascading failures:
- CLOSED: Requests pass freely. Failures increment a sliding/rolling counter.
- OPEN: Fast-fail immediately without hitting the broken upstream.
- HALF-OPEN: After recovery cooldown expires, permits limited test probes to verify upstream health.
"""

import asyncio
from enum import Enum
import time
from typing import Dict, List, Optional, Tuple


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF-OPEN"


class CircuitBreakerOpenException(Exception):
    """Raised when an upstream request is blocked by an OPEN circuit."""
    def __init__(self, provider_name: str, cooldown_remaining: float):
        self.provider_name = provider_name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker for provider '{provider_name}' is OPEN. "
            f"Upstream failing. Fast-failing request. Cooldown remaining: {cooldown_remaining:.2f}s."
        )


class CircuitBreaker:
    """Thread-safe, asynchronous tri-state Circuit Breaker."""

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        failure_window_seconds: float = 10.0,
        recovery_timeout_seconds: float = 15.0,
        half_open_success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window_seconds
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_timestamps: List[float] = []
        self._half_open_successes: int = 0
        self._last_state_change: float = time.monotonic()
        self._last_failure_timestamp: Optional[float] = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def _clean_old_failures(self, now: float) -> None:
        """Remove failure timestamps outside the rolling failure window."""
        cutoff = now - self.failure_window
        self._failure_timestamps = [t for t in self._failure_timestamps if t >= cutoff]

    async def can_execute(self) -> Tuple[bool, CircuitState, float]:
        """Check whether a request is allowed to hit upstream.
        
        Returns:
            Tuple[bool, CircuitState, float]: (allowed, current_state, cooldown_remaining)
        """
        now = time.monotonic()
        async with self._lock:
            # Check if OPEN state has expired and should transition to HALF-OPEN
            if self._state == CircuitState.OPEN:
                elapsed = now - self._last_state_change
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_successes = 0
                    self._last_state_change = now
                    return True, self._state, 0.0
                else:
                    cooldown_remaining = max(0.0, self.recovery_timeout - elapsed)
                    return False, self._state, cooldown_remaining

            elif self._state == CircuitState.HALF_OPEN:
                # In HALF-OPEN, permit trial probe
                return True, self._state, 0.0

            else:  # CLOSED
                return True, self._state, 0.0

    async def record_success(self) -> None:
        """Record a successful response from upstream."""
        now = time.monotonic()
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_success_threshold:
                    # Upstream proven recovered! Reset to CLOSED
                    self._state = CircuitState.CLOSED
                    self._failure_timestamps.clear()
                    self._half_open_successes = 0
                    self._last_state_change = now

            elif self._state == CircuitState.CLOSED:
                # In closed state, periodically keep failures clean
                self._clean_old_failures(now)

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record an upstream failure (5xx, 429, timeout, network error)."""
        now = time.monotonic()
        async with self._lock:
            self._last_failure_timestamp = now
            if self._state == CircuitState.CLOSED:
                self._failure_timestamps.append(now)
                self._clean_old_failures(now)
                if len(self._failure_timestamps) >= self.failure_threshold:
                    # Trip to OPEN
                    self._state = CircuitState.OPEN
                    self._last_state_change = now

            elif self._state == CircuitState.HALF_OPEN:
                # A single failure in HALF-OPEN immediately trips back to OPEN
                self._state = CircuitState.OPEN
                self._half_open_successes = 0
                self._last_state_change = now

            elif self._state == CircuitState.OPEN:
                # Already OPEN; update timestamp
                self._last_failure_timestamp = now

    async def get_metrics(self) -> Dict[str, any]:
        """Snapshot current circuit breaker status."""
        now = time.monotonic()
        async with self._lock:
            self._clean_old_failures(now)
            elapsed = now - self._last_state_change
            cooldown_rem = max(0.0, self.recovery_timeout - elapsed) if self._state == CircuitState.OPEN else 0.0
            return {
                "name": self.name,
                "state": self._state.value,
                "recent_failures_count": len(self._failure_timestamps),
                "failure_threshold": self.failure_threshold,
                "half_open_successes": self._half_open_successes,
                "half_open_threshold": self.half_open_success_threshold,
                "cooldown_remaining_seconds": round(cooldown_rem, 2),
                "last_failure_time": self._last_failure_timestamp,
            }

    async def trip_manually(self) -> None:
        """Force the circuit breaker to OPEN state (for testing/maintenance)."""
        async with self._lock:
            self._state = CircuitState.OPEN
            self._last_state_change = time.monotonic()

    async def reset_manually(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_timestamps.clear()
            self._half_open_successes = 0
            self._last_state_change = time.monotonic()
