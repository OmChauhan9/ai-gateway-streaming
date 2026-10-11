"""Unit tests for the Tri-State Circuit Breaker state machine."""

import asyncio
import pytest
from gateway.circuit_breaker.state_machine import CircuitBreaker, CircuitState


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    cb = CircuitBreaker(
        name="test-provider",
        failure_threshold=3,
        failure_window_seconds=2.0,
        recovery_timeout_seconds=0.1,  # 100ms cooldown
        half_open_success_threshold=2,
    )

    # 1. Starts CLOSED
    assert cb.state == CircuitState.CLOSED
    allowed, state, _ = await cb.can_execute()
    assert allowed is True
    assert state == CircuitState.CLOSED

    # 2. Record 2 failures (< threshold 3) -> remains CLOSED
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3. 3rd failure reaches threshold -> trips to OPEN
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # 4. Immediate execution check should fast-reject
    allowed, state, cooldown = await cb.can_execute()
    assert allowed is False
    assert state == CircuitState.OPEN
    assert cooldown > 0.0

    # 5. Wait for recovery timeout (100ms) -> transitions to HALF-OPEN
    await asyncio.sleep(0.12)
    allowed, state, _ = await cb.can_execute()
    assert allowed is True
    assert state == CircuitState.HALF_OPEN

    # 6. First probe in HALF-OPEN succeeds
    await cb.record_success()
    # Still needs 1 more success to satisfy threshold=2
    assert cb.state == CircuitState.HALF_OPEN

    # 7. Second probe succeeds -> resets to CLOSED!
    await cb.record_success()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_immediately_trips_to_open():
    cb = CircuitBreaker(
        name="test-probe-fail",
        failure_threshold=2,
        failure_window_seconds=1.0,
        recovery_timeout_seconds=0.05,
        half_open_success_threshold=2,
    )

    # Trip to OPEN
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # Cooldown into HALF-OPEN
    await asyncio.sleep(0.06)
    allowed, state, _ = await cb.can_execute()
    assert allowed is True
    assert state == CircuitState.HALF_OPEN

    # A single failure in HALF-OPEN immediately trips back to OPEN
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_manual_trip_and_reset():
    cb = CircuitBreaker(name="manual-test")
    assert cb.state == CircuitState.CLOSED

    await cb.trip_manually()
    assert cb.state == CircuitState.OPEN

    await cb.reset_manually()
    assert cb.state == CircuitState.CLOSED
