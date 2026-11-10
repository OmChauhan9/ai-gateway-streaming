"""Failover under an injected primary outage: what clients saw and how the primary's circuit breaker moved.

Three phases against an in-process gateway with two mock upstreams (primary and fallback):
1. normal: `--normal` sequential requests while the primary is healthy.
2. outage: the primary mock returns HTTP 503 for every request. `--outage` requests go out in batches of
   `--concurrency`, each batch fired together with asyncio.gather.
3. recovery: the primary mock is reset to healthy, the script waits out the breaker's recovery timeout,
   then sends `--recovery` sequential requests.

Every breaker transition is recorded with its time through CircuitBreaker.add_transition_listener, and
every summary line is computed from the recorded responses: status codes, primary vs fallback per phase
(X-Gateway-Fallback header), streams that did not end normally (no finish_reason, an error event, or no
final `data: [DONE]`), latency per phase, the observed transition sequence and the final states. The
script exits 1 if any request did not get a 200 or a complete stream.

What it does not measure: real providers or real network faults (mocks run in the same event loop over
in-memory transports that stream body chunks, see tests/conftest.py StreamingASGITransport), mid-stream
failures (the 503 arrives before any body), or failover latency under production load. Latency is the
full stream, request sent to last byte read, and includes the mocks' default TTFT and TPOT sleeps
(30 ms + 15 x 10 ms). Requests in a concurrent batch share one event loop, so their latencies include
waiting for each other.

Run from the repository root:
    PYTHONPATH=. python benchmarks/benchmark_failover.py
"""

import argparse
import asyncio
import collections
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

from benchmarks._report import describe_machine, print_machine, stream_completed, summarize, write_results
from config.settings import Settings
from gateway.circuit_breaker.state_machine import CircuitState
from gateway.main import create_app
from gateway.upstream.client import AsyncUpstreamClient
from gateway.upstream.mock_server import create_mock_upstream_app
from tests.conftest import MultiAppTransport, StreamingASGITransport

PRIMARY = "groq-primary"
FALLBACK = "together-fallback"
PHASES = ("normal", "outage", "recovery")


async def send(client: httpx.AsyncClient, phase: str, index: int, t_ref: float) -> Dict[str, Any]:
    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": f"{phase} prompt {index}"}],
        "stream": True,
    }
    t0 = time.perf_counter()
    sample: Dict[str, Any] = {"phase": phase, "index": index, "start_ms": (t0 - t_ref) * 1000.0}
    try:
        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            body = b"".join([chunk async for chunk in response.aiter_bytes()])
    except httpx.HTTPError as exc:
        return {**sample, "status": None, "error": repr(exc), "provider": None, "fallback": None,
                "complete": False, "latency_ms": (time.perf_counter() - t0) * 1000.0}
    return {
        **sample,
        "status": response.status_code,
        "provider": response.headers.get("X-Gateway-Provider"),
        "fallback": response.headers.get("X-Gateway-Fallback"),
        "complete": stream_completed(body),
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
    }


async def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    primary_mock = create_mock_upstream_app(PRIMARY)
    fallback_mock = create_mock_upstream_app(FALLBACK)
    upstream = AsyncUpstreamClient(transport=MultiAppTransport(primary_mock, fallback_mock))
    settings = Settings(
        app_name="Failover benchmark",
        primary_provider_name=PRIMARY,
        primary_base_url="http://127.0.0.1:8001/v1",
        fallback_provider_name=FALLBACK,
        fallback_base_url="http://127.0.0.1:8002/v1",
        cb_failure_threshold=args.failure_threshold,
        cb_failure_window_seconds=args.failure_window_s,
        cb_recovery_timeout_seconds=args.recovery_timeout_s,
        cb_half_open_success_threshold=args.half_open_successes,
        rate_limiter_capacity_tokens=500_000,
        cache_enabled=False,
    )
    gateway = create_app(settings=settings, client=upstream)
    registry = gateway.state.ctx.registry

    t_ref = time.perf_counter()
    transitions: List[Dict[str, Any]] = []

    def record(provider: str, old: Optional[CircuitState], new: CircuitState) -> None:
        transitions.append({"t_ms": (time.perf_counter() - t_ref) * 1000.0, "provider": provider,
                            "from": old.value if old else None, "to": new.value})

    for name in (PRIMARY, FALLBACK):
        breaker = registry.get_provider(name).circuit_breaker
        record(name, None, breaker.state)
        breaker.add_transition_listener(record)

    samples: List[Dict[str, Any]] = []
    phase_start_ms: Dict[str, float] = {}
    try:
        async with httpx.AsyncClient(transport=StreamingASGITransport(gateway), base_url="http://gw") as gw, \
                   httpx.AsyncClient(transport=StreamingASGITransport(primary_mock), base_url="http://primary") as adm:
            phase_start_ms["normal"] = (time.perf_counter() - t_ref) * 1000.0
            for i in range(args.normal):
                samples.append(await send(gw, "normal", i, t_ref))

            (await adm.post("/admin/fault-injection", json={"fail_with_status": 503})).raise_for_status()
            phase_start_ms["outage"] = (time.perf_counter() - t_ref) * 1000.0
            for start in range(0, args.outage, args.concurrency):
                batch = range(start, min(start + args.concurrency, args.outage))
                samples.extend(await asyncio.gather(*(send(gw, "outage", i, t_ref) for i in batch)))

            (await adm.post("/admin/reset")).raise_for_status()
            await asyncio.sleep(args.recovery_timeout_s + 0.1)
            phase_start_ms["recovery"] = (time.perf_counter() - t_ref) * 1000.0
            for i in range(args.recovery):
                samples.append(await send(gw, "recovery", i, t_ref))

            final_states = {name: (await registry.get_provider(name).circuit_breaker.get_metrics())["state"]
                            for name in (PRIMARY, FALLBACK)}
    finally:
        await upstream.close()

    return {"samples": samples, "transitions": transitions, "phase_start_ms": phase_start_ms,
            "final_states": final_states}


def _status_counts(samples: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(sorted(collections.Counter(str(s["status"]) for s in samples).items()))


def build_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    samples = result["samples"]
    phases = {}
    for phase in PHASES:
        rows = [s for s in samples if s["phase"] == phase]
        phases[phase] = {
            "requests": len(rows),
            "start_ms": result["phase_start_ms"].get(phase),
            "status_counts": _status_counts(rows),
            "served_by_primary": sum(s["fallback"] == "false" for s in rows),
            "served_by_fallback": sum(s["fallback"] == "true" for s in rows),
            "incomplete_streams": sum(s["status"] == 200 and not s["complete"] for s in rows),
            "latency_ms": summarize([s["latency_ms"] for s in rows]),
        }
    primary_states = [t["to"] for t in result["transitions"] if t["provider"] == PRIMARY]
    return {
        "requests": len(samples),
        "status_counts": _status_counts(samples),
        "non_200": sum(s["status"] != 200 for s in samples),
        "incomplete_streams": sum(s["status"] == 200 and not s["complete"] for s in samples),
        "phases": phases,
        "primary_state_sequence": primary_states,
        "transitions": result["transitions"],
        "final_states": result["final_states"],
    }


def print_summary(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    print("mode: in-process ASGI transports with streamed bodies; no sockets are involved")
    print(f"breaker: threshold {config['failure_threshold']} failures in {config['failure_window_s']} s, "
          f"recovery timeout {config['recovery_timeout_s']} s, {config['half_open_successes']} half-open successes; "
          f"outage batches of {config['concurrency']} concurrent requests")
    print()
    print(f"{'phase':<9} | {'reqs':>4} | {'status codes':<16} | {'primary':>7} | {'fallback':>8} | "
          f"{'incomplete':>10} | {'p50 ms':>8} | {'p90 ms':>8}")
    for phase, p in summary["phases"].items():
        lat = p["latency_ms"]
        p50 = f"{lat['p50']:>8.1f}" if lat["n"] else f"{'-':>8}"
        p90 = f"{lat['p90']:>8.1f}" if lat["n"] else f"{'-':>8}"
        codes = ", ".join(f"{k}: {v}" for k, v in p["status_counts"].items()) or "-"
        print(f"{phase:<9} | {p['requests']:>4} | {codes:<16} | {p['served_by_primary']:>7} | "
              f"{p['served_by_fallback']:>8} | {p['incomplete_streams']:>10} | {p50} | {p90}")
    print()
    print("breaker transitions (ms since start):")
    for t in summary["transitions"]:
        print(f"  {t['t_ms']:>9.1f}  {t['provider']:<18} {t['from'] or '(initial)':>9} -> {t['to']}")
    print(f"primary state sequence: {' -> '.join(summary['primary_state_sequence'])}")
    print(f"final states: {', '.join(f'{k} {v}' for k, v in summary['final_states'].items())}")
    print(f"requests: {summary['requests']}, status codes {summary['status_counts']}, "
          f"non-200: {summary['non_200']}, incomplete streams: {summary['incomplete_streams']}")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--normal", type=int, default=10, help="sequential requests before the outage")
    parser.add_argument("--outage", type=int, default=25, help="requests during the outage")
    parser.add_argument("--concurrency", type=int, default=25, help="outage requests fired together per batch")
    parser.add_argument("--recovery", type=int, default=15, help="sequential requests after the primary recovers")
    parser.add_argument("--failure-threshold", type=int, default=5)
    parser.add_argument("--failure-window-s", type=float, default=10.0)
    parser.add_argument("--recovery-timeout-s", type=float, default=0.5)
    parser.add_argument("--half-open-successes", type=int, default=2)
    parser.add_argument("--out", help="results JSON path (default: benchmarks/results/failover-<time>.json)")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    machine = describe_machine(__file__, argv)
    print_machine(machine)
    result = asyncio.run(run_benchmark(args))
    summary = build_summary(result)
    config = vars(args)
    print_summary(summary, config)
    samples = sorted(result["samples"], key=lambda s: s["start_ms"])
    path = write_results("failover", machine, config, samples, summary, args.out)
    print(f"\nresults: {path}")
    return 1 if summary["non_200"] or summary["incomplete_streams"] else 0


if __name__ == "__main__":
    sys.exit(main())
