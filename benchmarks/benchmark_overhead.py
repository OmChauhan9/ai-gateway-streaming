"""Gateway latency overhead: the same streaming request sent to the mock upstream directly and through the gateway.

Each pair sends one request straight to the mock upstream and one through the gateway to that same mock,
back to back, alternating which goes first so warm-up and drift favor neither side. The overhead of a pair
is gateway minus direct, for the full stream (request sent to last byte read) and for the first body chunk.
Negative differences are kept, because clipping them at zero would bias every statistic upward.
Percentiles are linearly interpolated (statistics.quantiles, inclusive method).

The mock's TTFT and TPOT sleeps default to 0 here. With the mock's own defaults (30 ms + 15 x 10 ms) each
request sleeps about 180 ms, and the overhead becomes the difference of two ~200 ms timings whose sleep
jitter is larger than the overhead being measured.

Modes:
- inprocess (default): mock, gateway and client share one event loop and talk through in-memory ASGI
  transports that stream body chunks (tests/conftest.py StreamingASGITransport). No sockets are involved,
  so the number excludes TCP, HTTP parsing in a server, and the uvicorn event loop, and it includes one
  extra in-memory hop for the gateway side.
- loopback: mock and gateway run as separate uvicorn processes on free 127.0.0.1 ports, and the client
  streams over TCP with httpx. The gateway side includes one extra loopback TCP hop and the gateway's
  own upstream HTTP client, which is what a co-located deployment adds.

What it does not measure: a real LLM (the mock streams canned tokens), a real network between hosts,
cache hits (the cache is off so every request runs the full proxy path), or behavior under concurrency
(pairs run one at a time). A pair counts as an error, and is left out of the statistics, unless both
sides return 200 with a body ending in `data: [DONE]` and the gateway served it from the primary, not
from the fallback or the cache.

Run from the repository root:
    PYTHONPATH=. python benchmarks/benchmark_overhead.py --mode inprocess
    PYTHONPATH=. python benchmarks/benchmark_overhead.py --mode loopback
"""

import argparse
import asyncio
import contextlib
import os
import socket
import statistics
import subprocess
import sys
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

import httpx

from benchmarks._report import REPO_ROOT, describe_machine, print_machine, summarize, write_results
from config.settings import Settings
from gateway.main import create_app
from gateway.upstream.client import AsyncUpstreamClient
from gateway.upstream.mock_server import create_mock_upstream_app
from tests.conftest import MultiAppTransport, StreamingASGITransport

PAYLOAD = {
    "model": "llama3-8b",
    "messages": [{"role": "user", "content": "Benchmark token generation latency test."}],
    "stream": True,
}
GATEWAY_SETTINGS: Dict[str, Any] = {
    "cache_enabled": False,
    "rate_limiter_enabled": True,
    "rate_limiter_capacity_tokens": 500_000,
    "rate_limiter_leak_rate_tokens_per_second": 100_000,
}
MODE_NOTES = {
    "inprocess": "in-process ASGI transports with streamed bodies; no sockets are involved",
    "loopback": "mock and gateway are separate uvicorn processes; the client streams over 127.0.0.1 TCP",
}

Clients = Tuple[httpx.AsyncClient, httpx.AsyncClient]


@contextlib.asynccontextmanager
async def inprocess_clients() -> AsyncIterator[Clients]:
    primary = create_mock_upstream_app("benchmark-primary")
    fallback = create_mock_upstream_app("benchmark-fallback")
    upstream = AsyncUpstreamClient(transport=MultiAppTransport(primary, fallback))
    settings = Settings(
        primary_provider_name="benchmark-primary",
        primary_base_url="http://127.0.0.1:8001/v1",
        fallback_provider_name="benchmark-fallback",
        fallback_base_url="http://127.0.0.1:8002/v1",
        **GATEWAY_SETTINGS,
    )
    gateway = create_app(settings=settings, client=upstream)
    try:
        async with httpx.AsyncClient(transport=StreamingASGITransport(primary), base_url="http://mock") as direct, \
                   httpx.AsyncClient(transport=StreamingASGITransport(gateway), base_url="http://gateway") as via_gw:
            yield direct, via_gw
    finally:
        await upstream.close()


def _free_ports(count: int) -> List[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for s in sockets:
            s.bind(("127.0.0.1", 0))
        return [s.getsockname()[1] for s in sockets]
    finally:
        for s in sockets:
            s.close()


async def _wait_until_healthy(url: str, proc: subprocess.Popen, timeout_s: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient() as probe:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"server for {url} exited with code {proc.returncode}")
            with contextlib.suppress(httpx.TransportError):
                if (await probe.get(url)).status_code == 200:
                    return
            await asyncio.sleep(0.05)
    raise RuntimeError(f"server for {url} not healthy after {timeout_s} s")


def _uvicorn(target: str, port: int, env: Dict[str, str], factory: bool = False) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "uvicorn", target, "--host", "127.0.0.1", "--port", str(port),
           "--log-level", "warning", "--no-access-log"]
    return subprocess.Popen(cmd + (["--factory"] if factory else []), cwd=REPO_ROOT, env=env)


@contextlib.asynccontextmanager
async def loopback_clients() -> AsyncIterator[Clients]:
    mock_port, gateway_port = _free_ports(2)
    mock_url = f"http://127.0.0.1:{mock_port}"
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    gateway_env = {
        **env,
        **{key.upper(): str(value).lower() if isinstance(value, bool) else str(value)
           for key, value in GATEWAY_SETTINGS.items()},
        "PRIMARY_PROVIDER_NAME": "benchmark-primary",
        "PRIMARY_BASE_URL": f"{mock_url}/v1",
        # Never used while the primary is healthy; a pair served by the fallback counts as an error.
        "FALLBACK_PROVIDER_NAME": "benchmark-fallback",
        "FALLBACK_BASE_URL": f"{mock_url}/v1",
    }
    procs: List[subprocess.Popen] = []
    try:
        procs.append(_uvicorn("gateway.upstream.mock_server:create_mock_upstream_app", mock_port, env, factory=True))
        procs.append(_uvicorn("gateway.main:app", gateway_port, gateway_env))
        await _wait_until_healthy(f"{mock_url}/health", procs[0])
        await _wait_until_healthy(f"http://127.0.0.1:{gateway_port}/health", procs[1])
        async with httpx.AsyncClient(base_url=mock_url) as direct, \
                   httpx.AsyncClient(base_url=f"http://127.0.0.1:{gateway_port}") as via_gw:
            yield direct, via_gw
    finally:
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


async def timed_stream(client: httpx.AsyncClient, expect_primary: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    t_first: Optional[float] = None
    body = bytearray()
    async with client.stream("POST", "/v1/chat/completions", json=PAYLOAD) as response:
        async for chunk in response.aiter_bytes():
            if t_first is None:
                t_first = time.perf_counter()
            body += chunk
    t_end = time.perf_counter()
    ok = response.status_code == 200 and body.rstrip().endswith(b"data: [DONE]")
    if expect_primary:
        ok = ok and response.headers.get("X-Gateway-Fallback") == "false" and response.headers.get("X-Cache") != "HIT"
    return {
        "ok": ok,
        "status": response.status_code,
        "total_ms": (t_end - t0) * 1000.0,
        "ttfb_ms": ((t_first or t_end) - t0) * 1000.0,
    }


async def measure_pair(direct: httpx.AsyncClient, via_gw: httpx.AsyncClient, gateway_first: bool) -> Dict[str, Any]:
    if gateway_first:
        gw = await timed_stream(via_gw, expect_primary=True)
        dr = await timed_stream(direct, expect_primary=False)
    else:
        dr = await timed_stream(direct, expect_primary=False)
        gw = await timed_stream(via_gw, expect_primary=True)
    return {
        "gateway_first": gateway_first,
        "ok": dr["ok"] and gw["ok"],
        "direct_status": dr["status"],
        "gateway_status": gw["status"],
        "direct_ms": dr["total_ms"],
        "gateway_ms": gw["total_ms"],
        "overhead_ms": gw["total_ms"] - dr["total_ms"],
        "direct_ttfb_ms": dr["ttfb_ms"],
        "gateway_ttfb_ms": gw["ttfb_ms"],
        "ttfb_overhead_ms": gw["ttfb_ms"] - dr["ttfb_ms"],
    }


def _column(pairs: List[Dict[str, Any]], key: str) -> List[float]:
    return [p[key] for p in pairs if p["ok"]]


def build_summary(mode: str, samples: List[Dict[str, Any]], runs: int) -> Dict[str, Any]:
    per_run = []
    for run in range(runs):
        pairs = [s for s in samples if s["run"] == run]
        per_run.append({
            "run": run,
            "errors": sum(not p["ok"] for p in pairs),
            "overhead_ms": summarize(_column(pairs, "overhead_ms")),
            "ttfb_overhead_ms": summarize(_column(pairs, "ttfb_overhead_ms")),
        })
    run_p50s = [r["overhead_ms"]["p50"] for r in per_run if r["overhead_ms"]["n"]]
    return {
        "mode": mode,
        "pairs": len(samples),
        "errors": sum(not s["ok"] for s in samples),
        "pooled": {key: summarize(_column(samples, key))
                   for key in ("direct_ms", "gateway_ms", "overhead_ms", "direct_ttfb_ms", "gateway_ttfb_ms",
                               "ttfb_overhead_ms")},
        "per_run": per_run,
        "overhead_p50_across_runs_ms": {
            "median": statistics.median(run_p50s), "min": min(run_p50s), "max": max(run_p50s),
        } if run_p50s else None,
    }


def _row(label: str, s: Dict[str, Any]) -> str:
    if not s["n"]:
        return f"{label:<26} | no successful pairs"
    return (f"{label:<26} | {s['n']:>5} | {s['p50']:>8.3f} | {s['p90']:>8.3f} | {s['p99']:>8.3f} | "
            f"{s['mean']:>8.3f} | {s['stdev']:>8.3f}")


def print_summary(summary: Dict[str, Any], config: Dict[str, Any]) -> None:
    header = f"{'ms':<26} | {'n':>5} | {'p50':>8} | {'p90':>8} | {'p99':>8} | {'mean':>8} | {'stdev':>8}"
    print(f"mode: {summary['mode']} ({MODE_NOTES[summary['mode']]})")
    print(f"mock: ttft {config['mock']['added_ttft_delay_ms']} ms, tpot {config['mock']['added_tpot_delay_ms']} ms, "
          f"{config['mock']['tokens_to_generate']} tokens; pairs per run {config['iterations']}, runs {config['runs']}, "
          f"warmup pairs {config['warmup']}")
    print(f"errors: {summary['errors']} of {summary['pairs']} pairs")
    print()
    print("Gateway minus direct, full stream, per run")
    print(header)
    for run in summary["per_run"]:
        print(_row(f"run {run['run']}", run["overhead_ms"]))
    print()
    print("Pooled over all runs")
    print(header)
    pooled = summary["pooled"]
    print(_row("direct, full stream", pooled["direct_ms"]))
    print(_row("gateway, full stream", pooled["gateway_ms"]))
    print(_row("overhead, full stream", pooled["overhead_ms"]))
    print(_row("direct, first chunk", pooled["direct_ttfb_ms"]))
    print(_row("gateway, first chunk", pooled["gateway_ttfb_ms"]))
    print(_row("overhead, first chunk", pooled["ttfb_overhead_ms"]))
    spread = summary["overhead_p50_across_runs_ms"]
    if spread:
        print()
        print(f"full-stream overhead p50 across runs: median {spread['median']:.3f} ms, "
              f"range {spread['min']:.3f} to {spread['max']:.3f} ms")


async def run_benchmark(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    mock_config = {
        "added_ttft_delay_ms": args.ttft_ms,
        "added_tpot_delay_ms": args.tpot_ms,
        "tokens_to_generate": args.tokens,
    }
    clients = inprocess_clients() if args.mode == "inprocess" else loopback_clients()
    samples: List[Dict[str, Any]] = []
    async with clients as (direct, via_gw):
        configured = await direct.post("/admin/fault-injection", json=mock_config)
        configured.raise_for_status()
        for i in range(args.warmup):
            await measure_pair(direct, via_gw, gateway_first=i % 2 == 1)
        for run in range(args.runs):
            for i in range(args.iterations):
                pair = await measure_pair(direct, via_gw, gateway_first=i % 2 == 1)
                samples.append({"run": run, "pair": i, **pair})
    return samples, mock_config


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=sorted(MODE_NOTES), default="inprocess")
    parser.add_argument("--iterations", type=int, default=200, help="measured pairs per run")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured pairs before the first run")
    parser.add_argument("--runs", type=int, default=5, help="repeat count, to show run-to-run spread")
    parser.add_argument("--ttft-ms", type=float, default=0.0, help="mock delay before the first chunk")
    parser.add_argument("--tpot-ms", type=float, default=0.0, help="mock delay before each token")
    parser.add_argument("--tokens", type=int, default=15, help="tokens the mock streams (at most 17)")
    parser.add_argument("--out", help="results JSON path (default: benchmarks/results/overhead-<time>.json)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    machine = describe_machine(__file__, argv)
    print_machine(machine)
    samples, mock_config = asyncio.run(run_benchmark(args))
    config = {**vars(args), "mock": mock_config, "gateway_settings": GATEWAY_SETTINGS, "payload": PAYLOAD}
    summary = build_summary(args.mode, samples, args.runs)
    print_summary(summary, config)
    path = write_results("overhead", machine, config, samples, summary, args.out)
    print(f"\nresults: {path}")
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
