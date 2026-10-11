"""In-memory Token Leaky-Bucket Rate Limiter.

Production LLM inference requires token-budgeting rather than simple request counts:
a 10-token prompt and an 8,000-token prompt have vastly different GPU resource costs.

The Leaky Bucket algorithm smooths out token bursts:
- Capacity (C): The burst limit. Maximum tokens that can be committed immediately.
- Leak Rate (r): The continuous draining rate (tokens per second) sustained over time.
- Water Level L(t): Current accumulated token pressure in the bucket.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


class RateLimitExceeded(Exception):
    """Raised when token bucket capacity is exceeded."""
    def __init__(self, key: str, retry_after: float, current_level: float, requested: int, capacity: int):
        self.key = key
        self.retry_after = retry_after
        self.current_level = current_level
        self.requested = requested
        self.capacity = capacity
        super().__init__(
            f"Rate limit exceeded for key '{key}'. "
            f"Current level: {current_level:.1f}/{capacity} tokens. "
            f"Requested: {requested}. Retry after {retry_after:.2f}s."
        )


@dataclass
class BucketState:
    """State tracking for a single rate limiter key."""
    level: float
    last_leak_time: float


class TokenLeakyBucketLimiter:
    """Asynchronous, thread-safe In-Memory Leaky Bucket Token Rate Limiter."""

    def __init__(
        self,
        capacity_tokens: int = 50000,
        leak_rate_tokens_per_second: float = 2000.0,
        cleanup_interval_seconds: float = 300.0,
    ):
        self.capacity = float(capacity_tokens)
        self.leak_rate = float(leak_rate_tokens_per_second)
        self.cleanup_interval = cleanup_interval_seconds
        self._buckets: Dict[str, BucketState] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = time.monotonic()

    def _leak(self, state: BucketState, now: float) -> None:
        """Drain accumulated tokens based on elapsed time."""
        delta = now - state.last_leak_time
        if delta > 0:
            leaked = delta * self.leak_rate
            state.level = max(0.0, state.level - leaked)
            state.last_leak_time = now

    async def acquire(self, key: str, tokens: int) -> Tuple[bool, float]:
        """Attempt to acquire a token allocation for a given key.
        
        Args:
            key: Client identifier (API key, user ID, or IP address).
            tokens: Number of tokens requested (estimated prompt tokens).
            
        Returns:
            Tuple[bool, float]: (admitted, retry_after_seconds)
            
        Raises:
            RateLimitExceeded if admitted is False.
        """
        now = time.monotonic()
        async with self._lock:
            # Periodic cleanup of empty buckets to prevent memory leak
            if now - self._last_cleanup > self.cleanup_interval:
                self._cleanup_stale_buckets(now)
                self._last_cleanup = now

            if key not in self._buckets:
                self._buckets[key] = BucketState(level=0.0, last_leak_time=now)

            state = self._buckets[key]
            self._leak(state, now)

            # Check if bucket can accommodate the requested tokens
            if state.level + tokens <= self.capacity:
                state.level += tokens
                return True, 0.0
            else:
                # Calculate required drain time
                overflow = (state.level + tokens) - self.capacity
                retry_after = max(0.01, overflow / self.leak_rate)
                raise RateLimitExceeded(
                    key=key,
                    retry_after=retry_after,
                    current_level=state.level,
                    requested=tokens,
                    capacity=int(self.capacity),
                )

    async def release_or_adjust(self, key: str, estimated_tokens: int, actual_tokens: int) -> None:
        """Reconcile bucket balance after actual response token count is known."""
        diff = actual_tokens - estimated_tokens
        now = time.monotonic()
        async with self._lock:
            if key in self._buckets:
                state = self._buckets[key]
                self._leak(state, now)
                # If actual was lower than estimated, reduce level; if higher, increase
                state.level = max(0.0, min(self.capacity, state.level + diff))

    async def get_status(self, key: str) -> Dict[str, float]:
        """Get the current level and utilization percentage for a key."""
        now = time.monotonic()
        async with self._lock:
            if key not in self._buckets:
                return {
                    "level": 0.0,
                    "capacity": self.capacity,
                    "remaining": self.capacity,
                    "utilization_pct": 0.0,
                }
            state = self._buckets[key]
            self._leak(state, now)
            return {
                "level": round(state.level, 2),
                "capacity": self.capacity,
                "remaining": round(max(0.0, self.capacity - state.level), 2),
                "utilization_pct": round((state.level / self.capacity) * 100, 2),
            }

    def _cleanup_stale_buckets(self, now: float) -> None:
        """Prune keys where level has been 0 for longer than 1 hour."""
        keys_to_delete = []
        for key, state in self._buckets.items():
            delta = now - state.last_leak_time
            if state.level <= 0.0 and delta > 3600.0:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self._buckets[key]

    async def reset(self) -> None:
        """Clear all bucket states (useful for testing)."""
        async with self._lock:
            self._buckets.clear()
