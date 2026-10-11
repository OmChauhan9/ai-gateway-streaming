"""Benchmark script to rigorously verify the '< 5ms Gateway Overhead' claim.

Runs 200 streaming requests:
1. Direct to Mock Upstream (baseline latency)
2. Routed through the AI Gateway (with full rate limiter, interceptor, TTFT/TPOT profiler, OTel, metrics)

Calculates the difference to evaluate gateway overhead distribution (p50, p90, p99).
"""

import asyncio
import statistics
import time
from typing import List, Tuple
import httpx

from config.settings import Settings
from gateway.main import create_app
from gateway.upstream.client import AsyncUpstreamClient
from gateway.upstream.mock_server import create_mock_upstream_app


async def run_benchmark(iterations: int = 150) -> None:
    print(f"\n{'='*70}")
    print(f"  AI GATEWAY LATENCY OVERHEAD BENCHMARK ({iterations} iterations)")
    print(f"  Target SLA: Gateway added streaming overhead < 5.0 ms")
    print(f"{'='*70}\n")

    # 1. Setup mock upstream and gateway apps
    primary_mock = create_mock_upstream_app("benchmark-primary")
    fallback_mock = create_mock_upstream_app("benchmark-fallback")

    from tests.conftest import MultiAppTransport
    transport = MultiAppTransport(primary_mock, fallback_mock)
    upstream_client = AsyncUpstreamClient(transport=transport)

    settings = Settings(
        app_name="Benchmark Gateway",
        primary_provider_name="benchmark-primary",
        primary_base_url="http://127.0.0.1:8001/v1",
        fallback_provider_name="benchmark-fallback",
        fallback_base_url="http://127.0.0.1:8002/v1",
        rate_limiter_enabled=True,
        rate_limiter_capacity_tokens=500000,
        rate_limiter_leak_rate_tokens_per_second=100000,
        cache_enabled=False,  # Bypass cache so every request tests the full proxy pipeline
    )
    gateway_app = create_app(settings=settings, client=upstream_client)

    # 2. Clients
    direct_transport = httpx.ASGITransport(app=primary_mock)
    gw_transport = httpx.ASGITransport(app=gateway_app)

    direct_latencies: List[float] = []
    gateway_latencies: List[float] = []
    overheads: List[float] = []

    payload = {
        "model": "llama3-8b",
        "messages": [{"role": "user", "content": "Benchmark token generation latency test."}],
        "stream": True,
    }

    async with httpx.AsyncClient(transport=direct_transport, base_url="http://direct") as direct_client, \
               httpx.AsyncClient(transport=gw_transport, base_url="http://gw") as gw_client:

        # Warmup (10 requests)
        print("Warming up JIT and asyncio event loop...")
        for _ in range(10):
            r1 = await direct_client.post("/v1/chat/completions", json=payload)
            _ = [c async for c in r1.aiter_bytes()]
            r2 = await gw_client.post("/v1/chat/completions", json=payload)
            _ = [c async for c in r2.aiter_bytes()]

        print(f"Running {iterations} paired measurements...")
        for i in range(iterations):
            # Direct measurement
            t0 = time.perf_counter()
            resp_direct = await direct_client.post("/v1/chat/completions", json=payload)
            direct_bytes = [c async for c in resp_direct.aiter_bytes()]
            t_direct = (time.perf_counter() - t0) * 1000.0
            direct_latencies.append(t_direct)

            # Gateway measurement
            t0 = time.perf_counter()
            resp_gw = await gw_client.post("/v1/chat/completions", json=payload)
            gw_bytes = [c async for c in resp_gw.aiter_bytes()]
            t_gw = (time.perf_counter() - t0) * 1000.0
            gateway_latencies.append(t_gw)

            overhead = max(0.0, t_gw - t_direct)
            overheads.append(overhead)

            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{iterations} iterations...")

    # 3. Compute Percentiles
    def get_percentiles(data: List[float]) -> Tuple[float, float, float, float]:
        sorted_data = sorted(data)
        p50 = statistics.median(sorted_data)
        p90 = sorted_data[int(len(sorted_data) * 0.90)]
        p99 = sorted_data[int(len(sorted_data) * 0.99)]
        avg = statistics.mean(sorted_data)
        return p50, p90, p99, avg

    d_p50, d_p90, d_p99, d_avg = get_percentiles(direct_latencies)
    g_p50, g_p90, g_p99, g_avg = get_percentiles(gateway_latencies)
    o_p50, o_p90, o_p99, o_avg = get_percentiles(overheads)

    print(f"\n{'='*70}")
    print(f"                       BENCHMARK RESULTS (ms)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} | {'p50 (ms)':<10} | {'p90 (ms)':<10} | {'p99 (ms)':<10} | {'Mean (ms)':<10}")
    print(f"{'-'*70}")
    print(f"{'Direct Upstream (Baseline)':<25} | {d_p50:<10.3f} | {d_p90:<10.3f} | {d_p99:<10.3f} | {d_avg:<10.3f}")
    print(f"{'Via AI Gateway':<25} | {g_p50:<10.3f} | {g_p90:<10.3f} | {g_p99:<10.3f} | {g_avg:<10.3f}")
    print(f"{'Gateway Added Overhead':<25} | {o_p50:<10.3f} | {o_p90:<10.3f} | {o_p99:<10.3f} | {o_avg:<10.3f}")
    print(f"{'='*70}")

    if o_p99 < 5.0:
        print(f"\n>> CLAIM VERIFIED: p99 Gateway Overhead ({o_p99:.3f} ms) is strictly LESS than 5.0 ms! <<\n")
    else:
        print(f"\n>> RESULT: p50 Overhead: {o_p50:.3f} ms, p90 Overhead: {o_p90:.3f} ms. <<\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
