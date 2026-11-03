"""Circuit Breaker State Machine (Michael Nygard Pattern).

Protects the gateway and upstream providers from cascading failures:
- CLOSED: Requests pass freely. Failures increment a sliding/rolling counter.
- OPEN: Fast-fail immediately without hitting the broken upstream.
- HALF-OPEN: After recovery cooldown expires, admits one trial probe at a time to verify upstream health.

Every admission is a Permit stamped with the breaker's epoch, which changes on every state change.
Only a permit from the current epoch, and in HALF-OPEN only the probe, can move the state.
"""

from enum import Enum
import time
from typing import Callable, Dict, List, Optional


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF-OPEN"


class Outcome(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    # No verdict on the upstream, such as a client disconnect. Frees a probe slot, counts as neither.
    ABANDONED = "abandoned"


class CircuitBreakerOpenException(Exception):
    """Raised when an upstream request is blocked by an OPEN circuit."""
    def __init__(self, provider_name: str, cooldown_remaining: float):
        self.provider_name = provider_name
        self.cooldown_remaining = cooldown_remaining
        super().__init__(
            f"Circuit breaker for provider '{provider_name}' is OPEN. "
            f"Upstream failing. Fast-failing request. Cooldown remaining: {cooldown_remaining:.2f}s."
        )


TransitionListener = Callable[[str, Optional[CircuitState], CircuitState], None]


class Permit:
    """One admission to the upstream. The first of succeed, fail or abandon settles it; later calls do nothing."""

    __slots__ = ("_breaker", "epoch", "outcome")

    def __init__(self, breaker: "CircuitBreaker", epoch: int):
        self._breaker = breaker
        self.epoch = epoch
        self.outcome: Optional[Outcome] = None

    def succeed(self) -> None:
        self._breaker._settle(self, Outcome.SUCCESS)

    def fail(self) -> None:
        self._breaker._settle(self, Outcome.FAILURE)

    def abandon(self) -> None:
        self._breaker._settle(self, Outcome.ABANDONED)


class CircuitBreaker:
    """Tri-state Circuit Breaker, confined to one event loop.

    Admission and settlement are synchronous, so no request can interleave inside them, and a
    cancelled request can settle its permit from a finally block without awaiting.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        failure_window_seconds: float = 10.0,
        recovery_timeout_seconds: float = 15.0,
        half_open_success_threshold: int = 2,
        probe_lease_seconds: float = 30.0,
        on_transition: Optional[TransitionListener] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.failure_window = failure_window_seconds
        self.recovery_timeout = recovery_timeout_seconds
        self.half_open_success_threshold = half_open_success_threshold
        self.probe_lease = probe_lease_seconds
        self._on_transition = on_transition

        self._state: CircuitState = CircuitState.CLOSED
        self._epoch: int = 0
        self._failure_timestamps: List[float] = []
        self._half_open_successes: int = 0
        self._probe: Optional[Permit] = None
        self._probe_expires_at: float = 0.0
        self._last_state_change: float = time.monotonic()
        self._last_failure_timestamp: Optional[float] = None
        if on_transition:
            on_transition(name, None, self._state)

    @property
    def state(self) -> CircuitState:
        return self._state

    def _clean_old_failures(self, now: float) -> None:
        """Remove failure timestamps outside the rolling failure window."""
        cutoff = now - self.failure_window
        self._failure_timestamps = [t for t in self._failure_timestamps if t >= cutoff]

    def try_acquire(self) -> Optional[Permit]:
        """Admit one request to the upstream, or return None when the breaker refuses it."""
        now = time.monotonic()
        self._advance(now)
        if self._state == CircuitState.OPEN:
            return None
        if self._state == CircuitState.HALF_OPEN:
            # A probe that never settles (its response body never started) loses the slot when its lease ends.
            if self._probe is not None and now < self._probe_expires_at:
                return None
            self._probe = Permit(self, self._epoch)
            self._probe_expires_at = now + self.probe_lease
            return self._probe
        return Permit(self, self._epoch)

    def _advance(self, now: float) -> None:
        """Apply the time-driven OPEN to HALF-OPEN change, which happens lazily on the next admission or read."""
        if self._state == CircuitState.OPEN and now - self._last_state_change >= self.recovery_timeout:
            self._transition(CircuitState.HALF_OPEN, now)

    def _settle(self, permit: Permit, outcome: Outcome) -> None:
        if permit.outcome is not None:
            return
        permit.outcome = outcome
        now = time.monotonic()
        if outcome == Outcome.FAILURE:
            self._last_failure_timestamp = now

        if permit.epoch != self._epoch:
            return
        if self._state == CircuitState.HALF_OPEN:
            if permit is not self._probe:
                return
            self._probe = None
            if outcome == Outcome.SUCCESS:
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_success_threshold:
                    self._transition(CircuitState.CLOSED, now)
            elif outcome == Outcome.FAILURE:
                self._transition(CircuitState.OPEN, now)
        elif outcome == Outcome.FAILURE:
            self._failure_timestamps.append(now)
            self._clean_old_failures(now)
            if len(self._failure_timestamps) >= self.failure_threshold:
                self._transition(CircuitState.OPEN, now)

    def _transition(self, new_state: CircuitState, now: float) -> None:
        """Enter a state and tell the listener. The new epoch makes every permit issued before it stale."""
        old_state = self._state
        self._state = new_state
        self._epoch += 1
        self._last_state_change = now
        self._half_open_successes = 0
        self._probe = None
        if new_state == CircuitState.CLOSED:
            self._failure_timestamps.clear()
        if self._on_transition and new_state != old_state:
            self._on_transition(self.name, old_state, new_state)

    async def get_metrics(self) -> Dict[str, any]:
        """Snapshot current circuit breaker status."""
        now = time.monotonic()
        self._advance(now)
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
        self._transition(CircuitState.OPEN, time.monotonic())

    async def reset_manually(self) -> None:
        """Reset the circuit breaker to CLOSED state."""
        self._transition(CircuitState.CLOSED, time.monotonic())
