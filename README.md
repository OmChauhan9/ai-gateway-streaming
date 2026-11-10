# Async AI Gateway with Streaming Telemetry & Circuit Breaking

[![Python](https://img.shields.io/badge/Python-3.11%20to%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![CI](https://github.com/OmChauhan9/ai-gateway-streaming/actions/workflows/ci.yml/badge.svg)](https://github.com/OmChauhan9/ai-gateway-streaming/actions/workflows/ci.yml)

An OpenAI-compatible reverse proxy for chat completions, built on FastAPI and httpx. A client sends `POST /v1/chat/completions` to the gateway instead of to a model provider, and the gateway does five things with it:

- It streams the provider's Server-Sent Events (SSE) to the client one complete event at a time, and measures time to first token (TTFT) and time per output token (TPOT) on the way.
- It limits each client to an estimated token budget with a leaky bucket.
- It answers exact repeat requests from an in-memory cache.
- It sends the request to one fallback provider when the primary's circuit breaker is open, or when the primary fails before it sends a response.
- It exports Prometheus metrics and creates OpenTelemetry spans.

---

## Architecture

```
                      Client request (OpenAI-compatible)
                           POST /v1/chat/completions
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 AI GATEWAY                                  │
│                                                                             │
│  1. Validation and token rate limit                                         │
│     - Pydantic v2 schema; NaN or Infinity in the body gets a 400            │
│     - Leaky bucket per client key; prompt tokens estimated as chars / 4     │
│       (429 when the bucket is full, 413 when one prompt exceeds capacity)   │
│                                                                             │
│  2. Exact-match cache                                                       │
│     - SHA-256 of the request; a hit is replayed without calling upstream    │
│                                                                             │
│  3. Circuit breakers and provider choice                                    │
│     - Per provider: CLOSED ──► OPEN ──► HALF-OPEN (one probe at a time)     │
│     - One primary and one fallback, each any OpenAI-compatible base URL     │
│                                                                             │
│  4. Upstream call (pooled httpx.AsyncClient)                                │
│     - Fails over before the first byte on connection errors, timeouts,      │
│       5xx, 429, 401 and 403; other 4xx go back to the client as they are    │
│                                                                             │
│  5. SSE streaming interceptor                                               │
│     - TTFT: gateway ingress to the first event with content                 │
│     - TPOT: (T_last - T_first) / (N - 1), N = content-bearing events        │
│     - Upstream failure mid-stream: inline SSE error event, then [DONE]      │
│                                                                             │
│  6. Observability                                                           │
│     - OpenTelemetry spans (no exporter configured by default)               │
│     - Prometheus scrape endpoint (/metrics)                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                           Client receives SSE stream
```

---

## Features

- **Streaming pass-through.** The gateway reads the upstream body with httpx and forwards each complete SSE event as soon as it arrives. It holds at most one partial event and never waits for the whole response. It does keep the answer text, to store in the cache.
- **TTFT and TPOT.** The interceptor measures both for each stream with `time.perf_counter`, and exports them as Prometheus histograms and span attributes. A "token" here is an SSE event that carries content. A provider may pack several tokens into one event.
- **Token leaky-bucket rate limiting.** Each client key has a bucket of capacity C that drains at r tokens per second: $L(t) = \max(0, L(t_{\text{last}}) - r \times \Delta t)$. The limiter admits a request if its estimated prompt tokens fit in the bucket. When the response ends, the gateway corrects the bucket by the completion size. For non-streaming responses that size comes from the upstream `usage`, and for streams it is the count of content events. The client key is the bearer token if the client sends one, and the client IP otherwise.
- **Circuit breakers with failover.** A breaker opens after 5 failures within 10 s. Both numbers are settings. After a cooldown the breaker lets one probe request through, and it closes after 2 successful probes. The gateway sends a request to the fallback in two cases: while the primary's breaker is open, and when a primary request fails before any byte reaches the client. If the fallback cannot serve it either, the client gets a 503 or a 502, as listed under Errors below.
- **Mid-stream failures.** Once the response has started, the client already has the status code, so the gateway cannot fail over. It ends the stream with an `event: error` SSE event and `data: [DONE]`. A stream that ends without any `finish_reason` gets the same treatment.
- **Prometheus metrics** at `/metrics`. They include TTFT and TPOT histograms, an end-to-end latency histogram, token counters, request counters, breaker state gauges, breaker trip counters, a rate-limit counter, and a cache-hit counter.
- **OpenTelemetry spans** for each request, with `gen_ai.request.model` and other `gen_ai.*` and `gateway.*` attributes. Some of the `gen_ai.*` names are deprecated or are not in the OpenTelemetry GenAI conventions. The Limitations section lists them.

---

## Benchmark results

All numbers come from the scripts in `benchmarks/`, run against the bundled mock upstream, not a real model. Each run prints the machine, the git commit, the package versions, and the load average. It saves them to `benchmarks/results/` together with every raw sample.

The numbers below come from an Apple M3 Pro with 11 logical CPUs and 18 GiB of memory, on macOS 26.5.2, with CPython 3.13.14, fastapi 0.162.4, httpx 0.28.1, and uvicorn 0.54.0, at commit c29a4d2. The 1-minute load average was between 2.6 and 3.0 during the runs.

### Added latency is under 1 ms with an instant upstream, and grows when the upstream is slow

Each pair sends the same streaming request once straight to the mock and once through the gateway, and alternates which one goes first. The overhead of a pair is the gateway time minus the direct time. The script keeps negative differences. Pairs run one at a time, so these numbers describe latency with no concurrent load.

| Setup | Pairs | Overhead p50 | p90 | p99 |
| :--- | ---: | ---: | ---: | ---: |
| In-process, mock with no delays | 5 × 200 | 0.44 ms | 0.63 ms | 0.77 ms |
| Loopback TCP, mock with no delays | 5 × 200 | 0.90 ms | 1.19 ms | 2.03 ms |
| Loopback TCP, mock with its default delays | 2 × 150 | 4.52 ms | 8.47 ms | 14.43 ms |

In-process mode runs the mock, the gateway, and the client in one event loop, with no sockets. Loopback mode runs the mock and the gateway as separate uvicorn processes on 127.0.0.1. The mock's default delays are 30 ms before the first token and 10 ms before each of 15 tokens.

With the delays on, the gateway's first chunk arrives 3.4 ms after the direct one at the median. With no delays the gap is 0.6 ms. In a separate check the gap grew with the time the gateway waits idle for the upstream: 1.1 ms after 5 ms and 3.5 ms after 30 ms. The gap also appears in-process, where it is 1.9 ms after 30 ms, so the network is not the only cause. The cause is not established. CPU frequency or process scheduling after idle on this laptop is one candidate. With the delays on, the full-stream p90 and p99 are also wide, because the mock's sleeps vary by a few milliseconds (standard deviation 4.3 ms).

### Every request got a 200 during an injected primary outage

This benchmark runs in-process against two mocks. During the outage phase, the primary mock returns 503 for every request.

| Phase | Requests | How they were sent | HTTP 200 | Served by primary | Served by fallback |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Normal | 10 | one at a time | 10 | 10 | 0 |
| Primary outage (503) | 25 | all at once | 25 | 0 | 25 |
| After recovery | 15 | one at a time | 15 | 15 | 0 |

All 50 requests got HTTP 200 and a stream that ended normally, with no error events. The primary's breaker went from CLOSED to OPEN at 2.06 s, to HALF-OPEN at 2.88 s, and back to CLOSED at 3.29 s.

This run covers failures that happen before the response starts. It does not cover mid-stream failures, real providers, or real network faults.

---

## Quickstart

### 1. Install
```bash
git clone https://github.com/OmChauhan9/ai-gateway-streaming.git
cd ai-gateway-streaming

# The dev extra includes pytest and locust
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Start the mock upstreams
The mock streams canned tokens and supports fault injection. It runs the primary on `:8001` and the fallback on `:8002`.
```bash
PYTHONPATH=. .venv/bin/python scripts/run_mock_upstream.py
```

### 3. Start the gateway
In a separate terminal, run:
```bash
PYTHONPATH=. .venv/bin/python scripts/run_gateway.py
```
The gateway listens on `http://127.0.0.1:8000`. To use real providers, set `PRIMARY_BASE_URL`, `PRIMARY_API_KEY`, `FALLBACK_BASE_URL`, and `FALLBACK_API_KEY`. See `.env.example` and `config/settings.py` for every setting.

### 4. Stream from the demo client
```bash
PYTHONPATH=. .venv/bin/python scripts/demo_client.py --prompt "How does an async AI gateway manage inference reliability?"
```

---

## Testing and benchmarks

### Run the test suite
```bash
.venv/bin/python -m pytest -v tests
```

### Measure the added latency
```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_overhead.py --mode inprocess
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_overhead.py --mode loopback
# With the mock's default delays, as in the third table row:
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_overhead.py --mode loopback --ttft-ms 30 --tpot-ms 10 --iterations 150 --runs 2
```

### Run the failover benchmark
```bash
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_failover.py
```

### Run the locust load test
At default settings, the cache answers most of locust's identical prompts, and the token limiter returns 429 for many of the rest. To load the proxy path itself, start the gateway with the cache off and higher limits:
```bash
CACHE_ENABLED=false RATE_LIMITER_CAPACITY_TOKENS=10000000 RATE_LIMITER_LEAK_RATE_TOKENS_PER_SECOND=1000000 \
  PYTHONPATH=. .venv/bin/python scripts/run_gateway.py
.venv/bin/locust -f benchmarks/locustfile.py --host http://127.0.0.1:8000 --users 50 --spawn-rate 10 --run-time 30s --headless
```
Locust's own timing of the POST stops at the response headers. The `stream:ttft` and `stream:full` rows time the first token and the whole stream.

---

## API reference

### `POST /v1/chat/completions`
The OpenAI-compatible chat completion endpoint, also served at `/chat/completions`. It accepts streaming and non-streaming requests with text content.

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

The gateway uses the bearer token only as the rate-limit key. Upstream calls use the provider keys from the gateway's own settings.

Response headers:
- `X-Gateway-Request-Id`, for example `gw-011e663dc00a`.
- `X-Gateway-Provider`, the provider that served the request, for example `groq`.
- `X-Gateway-Fallback`, `true` if the fallback served it.
- `X-Cache`, either `MISS` or `HIT`. The gateway sends it only while the cache is enabled. A cache hit carries no provider header and no fallback header.

Errors use the OpenAI shape, `{"error": {"message": ..., "type": ...}}`, except for 422.

| Status | Meaning |
| :--- | :--- |
| 400 | The body contains NaN or Infinity. |
| 422 | Schema validation failed. The body is FastAPI's `{"detail": [...]}`. |
| 413 | One prompt is larger than the whole bucket. |
| 429 | The token budget is used up. The response has a `Retry-After` header. |
| Other 4xx | The provider returned it, and the gateway passes it on. |
| 502 | The fallback served the request directly, because the primary's breaker was open, and it failed. Also returned when the last failure was a provider rejecting the gateway's own credentials with 401 or 403. |
| 503 | No breaker admits the request, or the primary failed and the fallback was unavailable or failed too. |

### `GET /metrics`
Prometheus exposition. Some of the series:
- `llm_ttft_seconds_bucket{provider="groq",model="llama3-70b-8192",le="0.05"}`
- `llm_tpot_seconds_bucket{provider="groq",model="llama3-70b-8192",le="0.02"}`
- `llm_requests_total{provider="groq",model="llama3-70b-8192",status="200",stream="true"}`
- `circuit_breaker_state{provider="groq"}`, where 0 is CLOSED, 1 is HALF-OPEN, and 2 is OPEN.

### `GET /health`
Returns the gateway status and every breaker's state. The status is `healthy` while at least one provider's breaker is not OPEN.

### Admin endpoints
These endpoints have no authentication.
- `GET /v1/circuit-breakers` returns every breaker's state.
- `POST /v1/circuit-breakers/{provider}/trip` forces a breaker OPEN, for failover drills.
- `POST /v1/circuit-breakers/{provider}/reset` forces a breaker CLOSED.
- `GET /v1/rate-limiter/status` returns the caller's bucket level.
- `POST /v1/cache/clear` empties the cache.

---

## Limitations

- **No client authentication.** The gateway does not check who a client is. A client that changes its bearer string gets a fresh rate-limit bucket. Anyone who can reach the gateway can call the admin endpoints, which can trip breakers and clear the cache. CORS allows any origin, with credentials.
- **State lives in one process.** Breakers, rate-limit buckets, and the cache are in memory. Each worker or replica has its own copy, and a restart clears them.
- **The cache matches exact requests only.** Any change to the messages or the generation parameters is a miss. The gateway skips the cache for `n > 1` and for request fields it does not model. A streamed hit arrives as a single SSE event. Cache hits create no span, so the `gateway.cached` span attribute is always false.
- **Token counts are estimates.** The prompt estimate is the character count divided by 4, plus a fixed amount for each message, not a tokenizer's count. On streams, the completion count is the number of content-bearing SSE events.
- **Failover is one hop, and happens only before the first byte.** There is one fallback. The gateway tries it once and does not retry with backoff. A failure after the stream starts ends in an error event.
- **The OpenTelemetry names are not current.** `gen_ai.system`, `gen_ai.usage.prompt_tokens`, and `gen_ai.usage.completion_tokens` are deprecated. Their replacements are `gen_ai.provider.name`, `gen_ai.usage.input_tokens`, and `gen_ai.usage.output_tokens`. `gen_ai.request.streaming`, `gen_ai.server.ttft`, and `gen_ai.server.tpot` are custom names. The gateway configures no span exporter, so the host application must set one up.
- **Only the mock is tested.** No test or benchmark in this repository runs against a real provider.

---

## Further reading

1. [EXPLANATION_AND_THEORY.md](EXPLANATION_AND_THEORY.md) covers the background: TTFT and TPOT, the leaky bucket, circuit breakers, and SSE, which the WHATWG HTML Living Standard specifies. It ends with references.
2. [WORKFLOW_AND_LIFECYCLE.md](WORKFLOW_AND_LIFECYCLE.md) walks through the request lifecycle stage by stage. It has Mermaid sequence diagrams for the happy path, failover, rate limiting, cache hits, and mid-stream failures.
