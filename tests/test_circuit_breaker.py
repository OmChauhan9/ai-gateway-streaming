"""Unit tests for the Tri-State Circuit Breaker state machine."""

import asyncio
import pytest
from gateway.circuit_breaker.state_machine import CircuitBreaker, CircuitState


async def _half_open(cb: CircuitBreaker) -> None:
    await cb.trip_manually()
    await asyncio.sleep(cb.recovery_timeout + 0.02)


@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    cb = CircuitBreaker(
        name="test-provider",
        failure_threshold=3,
        failure_window_seconds=2.0,
        recovery_timeout_seconds=0.1,  # 100ms cooldown
        half_open_success_threshold=2,
    )

    assert cb.state == CircuitState.CLOSED
    permits = [cb.try_acquire() for _ in range(3)]
    assert all(p is not None for p in permits), "CLOSED admits every request"

    permits[0].fail()
    permits[1].fail()
    assert cb.state == CircuitState.CLOSED, "2 failures stay below the threshold of 3"

    permits[2].fail()
    assert cb.state == CircuitState.OPEN, "the 3rd failure trips the breaker"

    assert cb.try_acquire() is None, "OPEN fast-rejects"
    assert (await cb.get_metrics())["cooldown_remaining_seconds"] > 0.0

    await asyncio.sleep(0.12)
    first_probe = cb.try_acquire()
    assert first_probe is not None, "the cooldown admits a probe"
    assert cb.state == CircuitState.HALF_OPEN

    first_probe.succeed()
    assert cb.state == CircuitState.HALF_OPEN, "one success is below the threshold of 2"

    second_probe = cb.try_acquire()
    assert second_probe is not None, "a settled probe frees the slot for the next one"
    second_probe.succeed()
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

    for permit in [cb.try_acquire(), cb.try_acquire()]:
        permit.fail()
    assert cb.state == CircuitState.OPEN

    await asyncio.sleep(0.06)
    probe = cb.try_acquire()
    assert probe is not None
    assert cb.state == CircuitState.HALF_OPEN

    probe.fail()
    assert cb.state == CircuitState.OPEN, "a failed probe trips straight back to OPEN"


@pytest.mark.asyncio
async def test_manual_trip_and_reset():
    cb = CircuitBreaker(name="manual-test")
    assert cb.state == CircuitState.CLOSED

    await cb.trip_manually()
    assert cb.state == CircuitState.OPEN

    await cb.reset_manually()
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_half_open_admits_one_probe_at_a_time():
    cb = CircuitBreaker(name="single-probe", recovery_timeout_seconds=0.05, half_open_success_threshold=2)
    await _half_open(cb)

    async def request() -> bool:
        permit = cb.try_acquire()
        if permit is None:
            return False
        await asyncio.sleep(0.01)
        permit.succeed()
        return True

    admitted = await asyncio.gather(*(request() for _ in range(50)))

    assert sum(admitted) == 1, "50 concurrent requests in HALF-OPEN admit exactly one probe"
    assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_results_admitted_before_the_trip_cannot_move_half_open():
    cb = CircuitBreaker(name="stale", recovery_timeout_seconds=0.05, half_open_success_threshold=2)
    long_stream_ok = cb.try_acquire()
    long_stream_broken = cb.try_acquire()
    await _half_open(cb)

    probe = cb.try_acquire()
    probe.succeed()
    long_stream_ok.succeed()
    assert cb.state == CircuitState.HALF_OPEN, "a success admitted while CLOSED is not a probe success"

    long_stream_broken.fail()
    assert cb.state == CircuitState.HALF_OPEN, "a failure admitted while CLOSED does not reopen the breaker"

    second_probe = cb.try_acquire()
    assert second_probe is not None, "stale results do not take the probe slot"
    second_probe.succeed()
    assert cb.state == CircuitState.CLOSED, "two real probe successes close the breaker"


@pytest.mark.asyncio
async def test_abandoned_probe_frees_the_slot_without_a_verdict():
    cb = CircuitBreaker(name="abandon", recovery_timeout_seconds=0.05, half_open_success_threshold=2)
    await _half_open(cb)

    disconnected = cb.try_acquire()
    assert cb.try_acquire() is None, "the probe holds the slot"
    disconnected.abandon()
    assert cb.state == CircuitState.HALF_OPEN, "abandoning is neither success nor failure"

    next_probe = cb.try_acquire()
    assert next_probe is not None, "the next request becomes the probe"
    disconnected.succeed()
    next_probe.succeed()
    assert cb.state == CircuitState.HALF_OPEN, "a permit settles once, so the late success does not count"


@pytest.mark.asyncio
async def test_listener_hears_each_state_change_once():
    heard = []
    cb = CircuitBreaker(
        name="listened",
        failure_threshold=3,
        recovery_timeout_seconds=0.05,
        on_transition=lambda name, old, new: heard.append((name, old, new)),
    )
    in_flight = [cb.try_acquire() for _ in range(10)]
    for permit in in_flight:
        permit.fail()
    await asyncio.sleep(0.06)
    assert (await cb.get_metrics())["state"] == "HALF-OPEN", "a status read reports the state after the cooldown"
    cb.try_acquire().fail()
    await cb.reset_manually()
    await cb.trip_manually()
    await cb.trip_manually()

    S = CircuitState
    assert heard == [
        ("listened", None, S.CLOSED),
        ("listened", S.CLOSED, S.OPEN),
        ("listened", S.OPEN, S.HALF_OPEN),
        ("listened", S.HALF_OPEN, S.OPEN),
        ("listened", S.OPEN, S.CLOSED),
        ("listened", S.CLOSED, S.OPEN),
    ], "10 failures in flight open the breaker once, and a trip while OPEN is not a state change"


@pytest.mark.asyncio
async def test_added_listener_hears_later_changes_alongside_the_first():
    first, added = [], []
    cb = CircuitBreaker(name="two", on_transition=lambda *change: first.append(change))
    cb.add_transition_listener(lambda *change: added.append(change))
    await cb.trip_manually()
    await cb.reset_manually()

    S = CircuitState
    assert added == [("two", S.CLOSED, S.OPEN), ("two", S.OPEN, S.CLOSED)], "only changes after it was added"
    assert first == [("two", None, S.CLOSED), *added], "the constructor's listener still hears everything"


@pytest.mark.asyncio
async def test_unsettled_probe_is_reclaimed_after_its_lease():
    cb = CircuitBreaker(name="lease", recovery_timeout_seconds=0.05, probe_lease_seconds=0.1)
    await _half_open(cb)

    lost = cb.try_acquire()
    assert cb.try_acquire() is None, "the slot is held while the lease runs"

    await asyncio.sleep(0.12)
    replacement = cb.try_acquire()
    assert replacement is not None, "an expired lease frees the slot"

    lost.fail()
    assert cb.state == CircuitState.HALF_OPEN, "the reclaimed probe's late result is ignored"
    replacement.fail()
    assert cb.state == CircuitState.OPEN, "the current probe still decides"
