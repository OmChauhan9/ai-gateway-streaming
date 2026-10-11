# Async AI Gateway with Streaming Telemetry & Circuit Breaking

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-GenAI%20Conventions-purple.svg)](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
[![Tests](https://img.shields.io/badge/Tests-22%20Passed-brightgreen.svg)]()
[![Gateway Overhead](https://img.shields.io/badge/p90%20Overhead-%3C%204.4ms-success.svg)]()

A production-grade **Async AI Gateway & Reverse Proxy** engineered for high-concurrency LLM inference. Deploys the true layer between production applications and foundation model APIs: **non-blocking streaming Server-Sent Events (SSE) byte processing**, **Time To First Token (TTFT)** & **Time Per Output Token (TPOT)** profiling, **Token Leaky-Bucket Rate Limiting**, and automated **tri-state Circuit Breaking with transparent multi-provider failover**.

---

## Architecture Overview

```
                      Client Request (OpenAI-compatible)
                           POST /v1/chat/completions
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 AI GATEWAY                                  │
│                                                                             │
│  1. Ingress Validation & Token Rate Limiter                                 │
│     - Schema validation (Pydantic v2)                                       │
│     - Leaky Bucket Token Budget (heuristic prompt token estimation)         │
│                                                                             │
│  2. Exact & Semantic Cache Check                                            │
│     - SHA-256 canonical hashing; instant 0ms TTFT replay on hit             │
│                                                                             │
│  3. Circuit Breaker & Upstream Health Router (Michael Nygard Pattern)       │
│     - Tri-state machine: CLOSED ──► OPEN ──► HALF-OPEN                      │
│     - Primary: Groq / Local Ollama  |  Fallback: Together AI / OpenAI       │
│                                                                             │
│  4. Async HTTP Connection Pool (httpx.AsyncClient)                          │
│     - Low-latency keep-alives; transparent failover on 5xx/429/timeouts     │
│                                                                             │
│  5. Non-Blocking SSE Streaming Interceptor                                  │
│     - Chunk 0: Capture TTFT timestamp (T_first - T_start)                   │
│     - Chunks 1..N: Accumulate completion tokens, measure ITL jitter         │
│     - Stream End: Calculate TPOT = (T_last - T_first) / (N - 1)             │
│     - Mid-stream abort: Inject standardized inline SSE error frame          │
│                                                                             │
│  6. Observability Exporters                                                 │
│     - OpenTelemetry GenAI Semantic Conventions (gen_ai.server.ttft/tpot)    │
│     - Prometheus Scrape Endpoint (/metrics)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                         Client Receives SSE Stream
                         Added Latency Overhead: < 4.4ms (p90)
```

---

## Core Features

- **Sub-5ms Streaming Overhead:** Non-blocking async chunk interceptor streams bytes directly without buffering entire responses in memory.
- **Microsecond Token Metrics:** Real-time calculation of **Time To First Token (TTFT)** and **Time Per Output Token (TPOT)**.
- **Token Leaky-Bucket Rate Limiting:** Enforces token-level budgets ($L(t) = \max(0, L(t_{\text{last}}) - r \times \Delta t)$) instead of simple request counts, smoothing GPU compute spikes.
- **Tri-State Circuit Breakers:** Protects upstream clusters from cascading failures. Fast-fails broken upstreams and diverts to secondary providers with **0 dropped client connections**.
- **OpenTelemetry GenAI Semantic Conventions:** Implements official OTel GenAI span attributes (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.server.ttft`, `gen_ai.server.tpot`, `gen_ai.usage.completion_tokens`).
- **Prometheus Metrics Exporter:** Exposes `/metrics` with dedicated histograms for TTFT and TPOT, token counters, and circuit breaker gauges.
- **Mid-Stream Drop Resilience:** Intercepts mid-stream connection crashes after HTTP headers have been committed and emits standard SSE error frames.

---

## Empirical Benchmark Results

Tested with `benchmarks/benchmark_overhead.py` (150 paired iterations) and `benchmarks/benchmark_failover.py` (50 concurrent requests under injected 503 outage):

### 1. Latency Overhead SLA

| Metric | Direct Upstream (Baseline) | Via AI Gateway | Gateway Added Overhead | Target SLA |
| :--- | :--- | :--- | :--- | :--- |
| **p50 (Median)** | 197.58 ms | 199.73 ms | **2.04 ms** | $< 5.0$ ms |
| **p90** | 199.36 ms | 201.54 ms | **4.31 ms** | $< 5.0$ ms |
| **Mean** | 197.68 ms | 199.76 ms | **2.17 ms** | $< 5.0$ ms |

### 2. Fault Injection & Zero-Drop Failover

| Test Metric | Value |
| :--- | :--- |
| Total Requests Issued | `50` |
| Primary Outage Injected | `HTTP 503 Service Unavailable` |
| Requests Handled by Primary (`groq`) | `25` |
| Requests Rescued by Fallback (`together`) | `25` |
| Client Success Rate | **100% (50/50 HTTP 200 OK)** |
| Dropped Client Connections | **0** |
| Circuit State Recovery | `CLOSED` $\to$ `OPEN` $\to$ `HALF-OPEN` $\to$ `CLOSED` |

---

## Quickstart

### 1. Installation
```bash
# Clone and enter directory
cd "AI Gateway with Streaming Telemetry & Circuit Breaking."

# Create virtual environment and install dependencies
uv venv --python python3
source .venv/bin/activate
uv pip install -e .
```

### 2. Run Synthetic Multi-Provider Upstreams
For local zero-cost testing, start the synthetic multi-provider server (Primary on `:8001`, Fallback on `:8002`):
```bash
PYTHONPATH=. .venv/bin/python scripts/run_mock_upstream.py
```

### 3. Start the AI Gateway
In a separate terminal:
```bash
PYTHONPATH=. .venv/bin/python scripts/run_gateway.py
```
*The gateway starts on `http://127.0.0.1:8000`.*

### 4. Interactive Live Client Stream
Run the interactive terminal client to watch streaming tokens and inspect real-time TTFT and TPOT:
```bash
PYTHONPATH=. .venv/bin/python scripts/demo_client.py --prompt "How does an async AI gateway manage inference reliability?"
```

---

## Testing & Verification

### Run Automated Test Suite (22 Tests)
```bash
.venv/bin/python -m pytest -v tests
```

### Run Latency Overhead Benchmark (< 5ms SLA)
```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_overhead.py
```

### Run Circuit Breaker & Fault Injection Benchmark
```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_failover.py
```

### Run Locust Load Test (50 Concurrent Streaming Users)
```bash
.venv/bin/locust -f benchmarks/locustfile.py --host http://127.0.0.1:8000 --users 50 --spawn-rate 10 --run-time 30s --headless
```

---

## API Reference

### `POST /v1/chat/completions`
OpenAI-compatible chat completion endpoint.

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer my-api-key" \
  -d '{
    "model": "llama3-70b-8192",
    "messages": [{"role": "user", "content": "Explain circuit breaking."}],
    "stream": true
  }'
```

**Response Headers:**
- `X-Gateway-Request-Id`: `gw-011e663dc00a`
- `X-Gateway-Provider`: `groq`
- `X-Gateway-Fallback`: `false`
- `X-Cache`: `MISS` (or `HIT` on cached requests)

### `GET /metrics`
Prometheus metrics exposition endpoint.
```bash
curl http://127.0.0.1:8000/metrics
```
Exposes:
- `llm_ttft_seconds_bucket{provider="groq",model="llama3-70b-8192",le="0.05"}`
- `llm_tpot_seconds_bucket{provider="groq",model="llama3-70b-8192",le="0.02"}`
- `llm_requests_total{provider="groq",status="200",stream="true"}`
- `circuit_breaker_state{provider="groq"}` (0=CLOSED, 1=HALF-OPEN, 2=OPEN)

### `GET /health`
Gateway health check and circuit status:
```bash
curl http://127.0.0.1:8000/health
```

### `POST /v1/circuit-breakers/{provider}/trip`
Manually trip a provider's circuit breaker to test failover routing drills:
```bash
curl -X POST http://127.0.0.1:8000/v1/circuit-breakers/groq/trip
```

### `POST /v1/circuit-breakers/{provider}/reset`
Reset a provider's circuit breaker to CLOSED:
```bash
curl -X POST http://127.0.0.1:8000/v1/circuit-breakers/groq/reset
```

---

## In-Depth Documentation

For detailed architectural, theoretical, and operational documentation:

1. **[EXPLANATION_AND_THEORY.md](file:///Users/omchauhan/Desktop/AI:ML%20Projects/AI%20Gateway%20with%20Streaming%20Telemetry%20&%20Circuit%20Breaking./EXPLANATION_AND_THEORY.md)**:
   - Deep-dive into inference economics, TTFT / TPOT mathematical formulas, Leaky Bucket calculus, Circuit Breaker state machines, and RFC 8895 SSE byte stream specifications.
   - Comprehensive bibliography of all official web links, RFCs, and documentation.
2. **[WORKFLOW_AND_LIFECYCLE.md](file:///Users/omchauhan/Desktop/AI:ML%20Projects/AI%20Gateway%20with%20Streaming%20Telemetry%20&%20Circuit%20Breaking./WORKFLOW_AND_LIFECYCLE.md)**:
   - End-to-end 10-stage request lifecycle breakdown.
   - 5 Mermaid sequence diagrams detailing Happy Path, Transparent Failover, Rate Limiting, Cache Hits, and Mid-Stream Drop handling.
