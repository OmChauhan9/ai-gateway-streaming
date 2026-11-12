# Concepts behind the async AI gateway

This document explains the ideas the gateway is built on and how the code in this repository applies each one. It covers latency metrics, server-sent events, token rate limiting, the circuit breaker, observability, and the response cache. Statements about the gateway describe the code on `main` and name the files that implement them. [WORKFLOW_AND_LIFECYCLE.md](WORKFLOW_AND_LIFECYCLE.md) follows a request through the gateway step by step.

---

## Table of contents
1. [Why put a gateway in front of model APIs](#1-why-put-a-gateway-in-front-of-model-apis)
2. [Inference latency metrics](#2-inference-latency-metrics)
3. [Server-sent events and the streaming path](#3-server-sent-events-and-the-streaming-path)
4. [Token leaky-bucket rate limiting](#4-token-leaky-bucket-rate-limiting)
5. [Circuit breaker and failover](#5-circuit-breaker-and-failover)
6. [OpenTelemetry spans and Prometheus metrics](#6-opentelemetry-spans-and-prometheus-metrics)
7. [Exact-match response cache](#7-exact-match-response-cache)
8. [What this gateway does not do](#8-what-this-gateway-does-not-do)
9. [References](#9-references)

---

## 1. Why put a gateway in front of model APIs

An application can call a model provider such as OpenAI, Groq, Together, or a self-hosted vLLM or Ollama server directly, or through a gateway. A gateway is a reverse proxy between the two. It gives one place to handle problems every caller would otherwise handle on its own:

- Providers fail. They return `429 Too Many Requests` when they rate-limit a caller and `5xx` when something breaks on their side. A gateway can count those failures and send traffic to another provider.
- Prompts differ in size. A limit on requests per minute treats a 5-token prompt and a 12,000-token prompt the same. A gateway can limit by tokens instead.
- Streamed answers have their own latency shape. The useful numbers are the time to the first token and the time between tokens, and a gateway in the stream path can measure both.
- Identical requests repeat. A gateway can answer a repeat from a cache without calling a provider.

This gateway handles each of these in one process. The handler for `POST /v1/chat/completions` in `gateway/main.py` runs these steps in order:

```
1. Estimate prompt tokens and charge the client's bucket.      Refused: 413 or 429.
2. Look up the exact-match cache.                              Hit: answer now, no provider call.
3. Ask the primary's circuit breaker for a permit.             Refused: ask the fallback's breaker.
                                                               Both refuse: 503.
4. Send the request to the chosen provider.                    Provider failure before headers:
                                                               one try on the fallback.
5. For a stream, pass events through the SSE interceptor,      Failure after headers: SSE error frame.
   which measures TTFT and TPOT.
6. Settle the breaker permit, record metrics and the span,
   reconcile the bucket, and store the answer in the cache.
```

The default primary is `groq` and the default fallback is `together` (`config/settings.py`). Their default URLs are `127.0.0.1:8001` and `127.0.0.1:8002`, where `scripts/run_mock_upstream.py` serves the mock upstream.

---

## 2. Inference latency metrics

Generation in a decoder-only language model has two phases:

```
[Request enters the gateway handler]
        │
        ▼ (T_start)
┌────────────────────────────────────────────────────────┐
│  Phase 1: Prefill (prompt processing)                  │
│  - Processes all prompt tokens in parallel             │
│  - Fills the key-value (KV) cache                      │
└────────────────────────────────────────────────────────┘
        │
        ▼ (T_first) ──► TTFT = T_first - T_start
┌────────────────────────────────────────────────────────┐
│  Phase 2: Autoregressive decoding (generation)         │
│  - Usually one token per forward pass                  │
│  - Streams tokens to the caller as SSE events          │
└────────────────────────────────────────────────────────┘
        │
        ▼ (T_last)  ──► TPOT = (T_last - T_first) / (N - 1)
[Last content event]
```

### Time to first token (TTFT)
In general use, TTFT is the time from sending a request to receiving the first token. The gateway measures a narrower interval. The clock starts on entry to the `chat_completions` handler in `gateway/main.py`, after FastAPI has parsed and validated the request body. It stops when the SSE parser yields the first event whose `choices[0].delta.content` is a non-empty string (`gateway/stream/interceptor.py`). That is when the event reaches the gateway, not when it reaches the client.

$$\text{TTFT} = T_{\text{first}} - T_{\text{start}}$$

The interval includes the gateway's own work before dispatch (the rate limiter, the cache lookup, and the breaker), the connection to the provider, and the provider's queueing and prefill time. Prefill attention cost grows with the square of the prompt length, $O(P^2)$ for a prompt of $P$ tokens.

### Time per output token (TPOT)
TPOT is the average time per output token after the first one.

For a stream with $N > 1$ content events:

$$\text{TPOT} = \frac{T_{\text{last}} - T_{\text{first}}}{N - 1}$$

In this gateway, $N$ is the number of SSE events that carry content, not the number of tokens. A provider can put several tokens in one event. A test stream whose single event held five words gave $N = 1$. The same count is what the gateway reports as completion tokens for a stream. When $N \le 1$, the gateway reports a TPOT of 0.

TPOT converts to a generation rate:

$$\text{Throughput (tokens/sec)} = \frac{1000}{\text{TPOT}_{\text{ms}}}$$

The gateway does not compute or export this rate.

### Inter-token latency (ITL)
ITL is the gap between consecutive content events $k-1$ and $k$:

$$\text{ITL}_k = T_k - T_{k-1} \quad \text{for } k \in [2, N]$$

Variance in ITL shows up as stutter in the streamed text. The interceptor records each gap and stores their mean as `inter_token_latencies_avg_ms` in `StreamTelemetryData`. No metric or span exports it. The gaps sum to $T_{\text{last}} - T_{\text{first}}$, so their mean equals TPOT and adds no information. The gateway does not measure ITL variance.

### End-to-end latency (E2E)

$$E2E = T_{\text{end}} - T_{\text{start}} = \text{TTFT} + \sum_{k=2}^{N} \text{ITL}_k + (T_{\text{end}} - T_{\text{last}}) = \text{TTFT} + (N - 1) \times \text{TPOT} + (T_{\text{end}} - T_{\text{last}})$$

The interceptor takes $T_{\text{end}}$ after the upstream stream ends. The last term covers the events after the last content event, such as the event that carries `finish_reason` and the closing `data: [DONE]`. For a stream, this E2E value is what `llm_gateway_latency_seconds` records. For a non-streaming request, the gateway records the time from handler entry until the provider's JSON body has arrived.

---

## 3. Server-sent events and the streaming path

### Why stream with SSE
A model can take seconds to generate an answer. With a plain JSON response, the caller sees nothing until generation ends. With server-sent events (SSE), the server sends each piece of text as one event on a single long-lived HTTP response. SSE is specified in the [WHATWG HTML Living Standard, §9.2 Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html), and its MIME type is `text/event-stream`. Over HTTP/1.1, a server can stream the body with chunked transfer encoding, though the standard's authoring notes (§9.2.7) warn that chunking "can have unexpected negative effects on the reliability of this protocol". HTTP/2 cannot use chunked encoding ([RFC 9113 §8.1](https://www.rfc-editor.org/rfc/rfc9113.html)) and carries the body in DATA frames instead.

### Wire format
An event is one or more `field: value` lines. A blank line ends the event and dispatches it. Lines may end in CRLF, LF, or CR (§9.2.5). OpenAI-compatible providers put one JSON chunk in each `data:` field and end the stream with `data: [DONE]`.

The gateway sends these response headers on a streamed answer (`gateway/main.py`), followed by the provider's events byte for byte:

```http
HTTP/1.1 200 OK
content-type: text/event-stream; charset=utf-8
cache-control: no-cache
connection: keep-alive
x-accel-buffering: no
x-gateway-request-id: gw-4f9a9551b419
x-gateway-provider: groq
x-gateway-fallback: false
x-cache: MISS

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"}}]}

data: {"id":"chatcmpl-01","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

`x-cache` appears only when the cache is enabled.

### How the gateway streams
The gateway forwards each upstream event as soon as that event is complete. It never waits for the whole response.

1. `upstream_byte_iter` in `gateway/main.py` reads the provider's body with httpx `Response.aiter_bytes()`. These are httpx's decoded body chunks, and their boundaries do not line up with events.
2. `iter_sse_events` in `gateway/stream/sse_parser.py` buffers bytes until a blank line ends an event, then yields that event's exact upstream bytes. It handles CRLF, LF, and CR line ends, including a CRLF split across two reads.
3. For each event, the parser runs `json.loads` on the `data` lines. It reads `choices[0].delta.content` and checks whether `choices[0]` carries a `finish_reason`.
4. `telemetry_stream_interceptor` in `gateway/stream/interceptor.py` timestamps each content event, keeps its text, and yields the event unchanged to Starlette's `StreamingResponse`.

Per stream, the gateway holds at most one incomplete event of raw bytes plus the full response text. It keeps the text so it can store the answer in the cache when the stream ends. If the upstream body ends without a final blank line, the parser still forwards the last event, with a blank line appended, when its data is whole JSON or `[DONE]`. Otherwise it drops the event as a cut-off fragment.

`benchmarks/benchmark_overhead.py` measures the latency the gateway adds against the mock upstream, either in-process or over loopback TCP, one request at a time. It does not measure behaviour under concurrency or across a real network. This document quotes no overhead figure.

### Failures after the headers are sent
Once the gateway has sent the response headers, the HTTP status is fixed at 200. A failure after that point cannot turn into a 502 status, so the interceptor ends the stream with an SSE error event instead. Two cases lead there:

- Any exception while the gateway reads or parses the upstream body. Dropped connections and read timeouts are the usual causes, but the interceptor catches every `Exception`.
- A stream that ends before any event's `choices[0]` carried a `finish_reason`. A closing `data: [DONE]` alone does not prove the answer is whole, so the interceptor raises `UpstreamConnectionError` for this case.

In both cases, the gateway counts a failure on the provider's circuit breaker, marks the span as an error, and sends this frame. The frame below was captured from a run in which the upstream dropped the connection:

```
event: error
data: {"error": {"message": "Upstream provider connection error mid-stream: peer closed", "type": "upstream_connection_broken", "code": 502, "provider": "groq"}}

data: [DONE]

```

The message ends with the exception's text. After the frame, the generator returns and the gateway closes the upstream response. The failed stream is counted in `llm_requests_total` with status `502`, the code in the error event, not `200`.

A client disconnect is not a provider failure. When the client goes away mid-stream, the gateway records status `499` in `llm_requests_total`, ends the span with `gateway.client_disconnected` set to true, and reconciles the client's bucket with the content events streamed so far. The gateway abandons the breaker permit, so the breaker records no outcome.

---

## 4. Token leaky-bucket rate limiting

### Count tokens, not requests
A requests-per-minute limit counts a 5-token prompt and a 12,000-token prompt as one request each, though the second costs the provider much more work. This gateway limits each client by estimated tokens instead. The client key is the Bearer token from the `Authorization` header, or the client IP when that header is absent (`_extract_client_key` in `gateway/main.py`). The gateway does not validate the token. It only uses it as the bucket key.

### The bucket model
The limiter lives in `gateway/limiter/leaky_bucket.py`. Each client has a bucket with capacity $C$ that drains at a constant rate $r$ tokens per second.

Let:
- $C$ be the bucket capacity in tokens, the largest burst a client can send.
- $r$ be the leak rate in tokens per second, the sustained rate.
- $L(t)$ be the bucket level at time $t$.
- $t_{\text{last}}$ be the time of the previous drain.

When a request that needs $T_{\text{req}}$ tokens arrives at time $t$, the limiter runs these steps:

1. **Drain.**
   $$\Delta t = t - t_{\text{last}}$$
   $$L(t) = \max\left(0, \, L(t_{\text{last}}) - r \times \Delta t\right)$$

2. **Size check.** If $T_{\text{req}} > C$, the request can never fit, and waiting cannot help. The gateway returns HTTP 413 with type `request_exceeds_token_capacity` instead of a 429. This check runs before the limiter touches the bucket.

3. **Admission.**
   $$\text{If } L(t) + T_{\text{req}} \le C: \quad \text{admit and set } L(t) \leftarrow L(t) + T_{\text{req}}$$
   $$\text{If } L(t) + T_{\text{req}} > C: \quad \text{reject with HTTP 429, type } \texttt{tokens\_per\_second\_exceeded}$$

4. **Retry time.**
   $$\text{Overflow} = (L(t) + T_{\text{req}}) - C$$
   $$\text{retry\_after} = \max\left(0.01, \, \frac{\text{Overflow}}{r}\right) \quad (\text{seconds})$$
   The JSON body carries `retry_after_seconds`, which is `retry_after` rounded to two decimals. The `Retry-After` header carries whole seconds:
   $$\text{Retry-After} = \lfloor \text{retry\_after} \rfloor + 1$$
   In a run with $C = 100$ and $r = 10$, the body said `6.7` and the header said `7`.

5. **Reconciliation.** Admission charges only the prompt estimate. After a successful response, a client disconnect, or a mid-stream failure, `release_or_adjust` adds the completion tokens and keeps the level at or above 0. The level can go above $C$. Later requests then wait until it drains far enough to admit them. After a mid-stream failure, the completion count is the content events streamed before the failure. For a stream, the completion count is the number of content events (section 2). For a non-streaming response, it is the provider's `usage.completion_tokens`.

When the gateway answers 502 or 503, no provider served the request, so `release_or_adjust` refunds the whole prompt estimate and the request costs the client nothing. LiteLLM's proxy also refunds its upfront token reservation when a call fails. A provider 4xx that the gateway relays keeps the prompt charge, because the fault is the client's. The gateway has no separate limit on requests, so that charge is what slows a client that keeps sending bad requests. OpenAI and Azure OpenAI also count failed requests toward their rate limits.

The limiter runs before the cache lookup. A cache hit is charged its prompt estimate like any other request, and nothing reconciles that charge.

This is the meter form of the leaky bucket. The bucket is a counter beside the traffic, and the limiter admits or refuses each request without queueing it.

```
          [Incoming token bursts]
                   │
                   ▼
         ┌───────────────────┐
         │ \               / │
         │  \  Bucket     /  │  Capacity C = 50,000 tokens
         │   \ Capacity  /   │
         │    \         /    │
         │     \~~~~~~~/<────┼── Current level L(t)
         │      \     /      │
         └───────┬───┬───────┘
                 │   │
                 ▼   ▼
            [Leak rate r]  <── 2,000 tokens / sec
```

The diagram shows the defaults from `config/settings.py`: `rate_limiter_capacity_tokens = 50000` and `rate_limiter_leak_rate_tokens_per_second = 2000.0`.

### Estimating prompt tokens
The gateway estimates prompt tokens with a rough heuristic, so it loads no tokenizer and runs no BPE on each request. The repo does not measure what a real tokenizer would cost, or how close the estimate comes to one. `gateway/limiter/tokenizer_estimator.py` adds up these parts:

- $\lceil \text{chars} / 4 \rceil$ for each non-empty string among the role, the content, and the name, computed as `max(1, (len + 3) // 4)`. A message with a name gets 1 more token.
- 3 tokens of framing per message, plus 3 priming tokens per request. These follow the shape of OpenAI's chat framing.
- The JSON of every field the schema does not model, such as `tools` or `tool_calls`, at the same rate. Those fields reach the provider unchanged, so they count as input.

One user message `"Hi"` estimates to 8 tokens: 3 for framing, 1 for `user`, 1 for `Hi`, and 3 for priming.

---

## 5. Circuit breaker and failover

A circuit breaker stops sending requests to a provider that keeps failing, then tests the provider before it sends normal traffic again. Michael Nygard popularized the pattern in *Release It!*, and Martin Fowler's essay describes its closed, open, and half-open states. Each provider has its own breaker (`gateway/circuit_breaker/state_machine.py`). `ProviderRegistry` in `gateway/circuit_breaker/provider_registry.py` picks which provider gets a request.

```
       ┌───────────────────────────────┐
       │                               │
       │            CLOSED             │ ◄─────────────────────────┐
       │   (requests go to the         │                           │
       │      provider)                │                           │
       │                               │                           │
       └──────────────┬────────────────┘                           │
                      │                                            │
                      │ >= failure_threshold (5) failures          │ half_open_success_threshold (2)
                      │ within failure_window (10 s)               │ probe successes
                      ▼                                            │
       ┌───────────────────────────────┐                           │
       │                               │                           │
       │             OPEN              │                           │
       │   (no permits; requests go    │                           │
       │      to the fallback)         │                           │
       │                               │                           │
       └──────────────┬────────────────┘                           │
                      │                                            │
                      │ recovery_timeout (15 s) passed,            │
                      │ applied on the next admit or status read   │
                      ▼                                            │
       ┌───────────────────────────────┐                           │
       │                               │                           │
       │           HALF-OPEN           │ ──────────────────────────┘
       │   (one probe at a time,       │
       │      held under a lease)      │ ────► Probe failure returns
       │                               │       to OPEN
       └───────────────────────────────┘
```

The numbers in the diagram are the defaults in `config/settings.py`.

### State transitions
1. **CLOSED.**
   - Every request whose breaker grants a permit goes to the provider.
   - Each failure adds a timestamp, and failures older than `cb_failure_window_seconds` (10 s) drop out. Successes do not clear the count. When `cb_failure_threshold` (5) failures fall inside the window, the breaker opens. A run with 4 failures, then 1 success, then 1 failure ended in OPEN.
2. **OPEN.**
   - The breaker opened because of too many recent failures, or because an operator called `POST /v1/circuit-breakers/{provider}/trip`.
   - The breaker grants no permits. The registry sends the request to the fallback, or the gateway returns 503 if the fallback's breaker refuses too.
   - Once `cb_recovery_timeout_seconds` (15 s) have passed since the breaker opened, it moves to HALF-OPEN. No timer drives this change. It happens on the next admission or the next status read, and `/metrics`, `/health`, and `/v1/circuit-breakers` all read the status.
3. **HALF-OPEN.**
   - The breaker lets one probe through at a time. While the probe is out, other requests get no permit from this breaker and go to the fallback.
   - The probe holds a lease as long as the provider's timeout (30 s by default). If the probe never settles, its slot frees when the lease ends. A probe that ends in a client disconnect frees the slot with no verdict.
   - If the probe fails, the breaker returns to OPEN and the cooldown starts again.
   - After `cb_half_open_success_threshold` (2) probe successes, the breaker closes and clears its failure history.

Every state change starts a new epoch. A permit issued before the change cannot move the state after it, so a slow request admitted before a trip cannot close the breaker. `POST /v1/circuit-breakers/{provider}/reset` forces a breaker to CLOSED.

### What counts as a failure
The upstream client in `gateway/upstream/client.py` sorts each outcome into an error class, and `gateway/main.py` settles the breaker permit from it.

| Upstream outcome | Breaker records | What the client gets |
| :--- | :--- | :--- |
| Connection error or timeout before headers (connect timeout 5 s, then 30 s for each read, write, or pool wait, with no deadline for the whole request) | failure | failover |
| HTTP 5xx | failure | failover |
| HTTP 429 | failure | failover |
| HTTP 401 or 403 | failure | failover, or 502 `upstream_auth_failed` when it is the last failure |
| Non-streaming HTTP 200 whose body is not JSON, has no list of choices with messages, or contains NaN or Infinity | failure | failover |
| Exception while reading or parsing a stream | failure | SSE error frame |
| Stream that ends with no `finish_reason` | failure | SSE error frame |
| Any other HTTP 4xx, such as 400, 404, or 422 | no outcome | the provider's status and body, with no failover |
| Client disconnect | no outcome | nothing, because the client is gone |

A 401 or 403 counts as a provider failure because the gateway calls providers with its own API key. Such a rejection means the gateway's credentials for that provider are wrong, so the gateway does not pass the provider's body to the caller.

### Failover
1. `ProviderRegistry.admit()` asks the primary's breaker for a permit. If the primary refuses, it asks the fallback's breaker. If both refuse, the client gets 503 `all_providers_unavailable`.
2. The gateway sends the request. If the primary fails before response headers, with any outcome marked "failover" in the table above, the gateway records the failure and tries the fallback once, provided the fallback's breaker grants a permit.
3. If the fallback succeeds, the client gets 200 with `X-Gateway-Fallback: true`, and `X-Gateway-Provider` names the fallback.
4. If the fallback refuses or also fails, the client gets 503 `all_providers_unavailable`. When the last failure was a 401 or 403, the client gets 502 `upstream_auth_failed` instead.
5. If the request went to the fallback in step 1 because the primary refused, and the fallback fails, the client gets 502 `upstream_error`. There is no second hop.

Runs against a mock upstream gave these results. A primary that returned 503, 429, 401, or a non-JSON 200, with a healthy fallback, gave 200 from the fallback. Both providers at 503 gave 503. Both at 401 gave 502. An OPEN primary with a fallback at 503 gave 502. A primary 400 came back to the client as 400, and the primary's breaker recorded no failure.

---

## 6. OpenTelemetry spans and Prometheus metrics

### OpenTelemetry span
Each request that passes the breaker gets one span named `chat {model}` (`gateway/telemetry/otel.py`). The name follows the spec's `{gen_ai.operation.name} {gen_ai.request.model}` pattern, but the span does not conform to the OpenTelemetry GenAI semantic conventions. Those conventions now live in the [open-telemetry/semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai) repository and have Development status. Compared with that repository at commit `e6658f1` (2026-11-06), the span differs in these ways:

- The span kind is `SERVER`. The spec says inference spans SHOULD be `CLIENT`.
- The spec requires `gen_ai.operation.name`. The code does not set it.
- Of the `gen_ai.*` attributes, only `gen_ai.request.model` is a current spec name.

| Attribute the code sets | Value | Spec status | Spec name |
| :--- | :--- | :--- | :--- |
| `gen_ai.system` | Provider name, changed to the fallback's name on failover | Deprecated | `gen_ai.provider.name` |
| `gen_ai.request.model` | Requested model | Current | `gen_ai.request.model` |
| `gen_ai.request.streaming` | Whether the request streams | Not in the spec | `gen_ai.request.stream` |
| `gen_ai.usage.prompt_tokens` | Prompt estimate (section 4) | Deprecated | `gen_ai.usage.input_tokens` |
| `gen_ai.usage.completion_tokens` | Content-event count for a stream, provider `usage.completion_tokens` otherwise | Deprecated | `gen_ai.usage.output_tokens` |
| `gen_ai.server.ttft` | TTFT in milliseconds, 0.0 for non-streaming requests | Not in the spec | Span attribute `gen_ai.response.time_to_first_chunk`, or histogram metric `gen_ai.server.time_to_first_token`, both in seconds |
| `gen_ai.server.tpot` | TPOT in milliseconds, 0.0 for non-streaming requests | Not in the spec | Histogram metric `gen_ai.server.time_per_output_token`, in seconds |
| `error.message` | Error text, set on failure | Deprecated in the core registry, which points to domain-specific message attributes | None |
| `gateway.request_id` | The `X-Gateway-Request-Id` value | Custom | None |
| `gateway.fallback_used` | Whether the request went to the fallback | Custom | None |
| `gateway.cached` | Always false | Custom | None |
| `gateway.client_disconnected` | True when the client left mid-stream | Custom | None |

The deprecations come from the deprecated-attributes table in the core [open-telemetry/semantic-conventions](https://github.com/open-telemetry/semantic-conventions) registry (`docs/registry/attributes/gen-ai.md` and `docs/registry/attributes/error.md`). This document does not rename anything. The table records what the code sets today.

`gateway.cached` is always false. A cache hit returns before the span starts, so no `chat` span exists for a hit. FastAPI 0.162.4, the version in the test environment, also emits its own spans, such as `POST /v1/chat/completions`, through the same tracer provider, and those do appear for cache hits.

Once the span starts, it ends once on each path out of the handler. A success ends it with status OK. A client disconnect ends it with no status set and `gateway.client_disconnected` true. Every other path ends it with status ERROR, including a failure before or after the headers, a relayed 4xx, and a cancelled request.

The gateway configures no span exporter. `OpenTelemetryGenAIManager` reuses the global SDK `TracerProvider` if one is installed, and otherwise installs one with no span processor. The `otel_exporter_endpoint` setting is never read. To export spans, the host process must install an SDK `TracerProvider` with a processor and exporter before it imports `gateway.main`, which builds the app at import time. It can also add a processor to `app.state.ctx.otel.tracer_provider`, which is what the tests do.

### Prometheus metrics
`GET /metrics` serves these metrics in Prometheus text format (`gateway/telemetry/prometheus.py`). Reading `/metrics` also reads every breaker's status, so a pending OPEN to HALF-OPEN change shows up in the gauge.

Histograms:
- `llm_ttft_seconds`, labelled `provider` and `model`, with buckets `[0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]`. The gateway observes it only for streams, and only when TTFT is above 0.
- `llm_tpot_seconds`, labelled `provider` and `model`, with buckets `[0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.25]`. The gateway observes it only for streams with at least two content events.
- `llm_gateway_latency_seconds`, labelled `provider` and `status`, with buckets `[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0]`. The gateway observes it with every `llm_requests_total` sample.

Counters:
- `llm_requests_total`, labelled `provider`, `model`, `status`, and `stream`. Every response the chat handler sends adds one sample, under the HTTP status the client received. A stream that fails after its headers counts as `502`, and a client disconnect as `499`. A rejection with no provider involved (413, 429, or every breaker open) uses `provider="none"`. Cache hits, and the 400 and 422 rejections that happen before the handler, add nothing.
- `llm_tokens_total`, labelled `provider`, `model`, and `type`, where `type` is `prompt` or `completion`. Prompt tokens are the estimate. Completion tokens are content events for a stream and the provider's `usage.completion_tokens` otherwise. The gateway records them with each `llm_requests_total` sample. A stream that fails after its headers records the prompt estimate and the content events already streamed, as a client disconnect does. Other failure samples add 0 tokens. For a 413, 429, 502, or 503 the limiter charges nothing either, so the two agree. For a relayed 4xx the limiter keeps the prompt charge.
- `circuit_breaker_trips_total`, labelled `provider`. It counts every change into OPEN, manual trips included.
- `llm_rate_limit_exceeded_total`, with no labels. It counts 429 responses from the limiter. A 413 does not count.
- `llm_cache_hits_total`, labelled `cache_type`, which is always `exact`.

Gauge:
- `circuit_breaker_state`, labelled `provider`, where `0` is CLOSED, `1` is HALF-OPEN, and `2` is OPEN. The breaker's transition listener is the only writer.

---

## 7. Exact-match response cache

Some requests repeat exactly, such as a fixed system prompt with the same question. The gateway keeps an in-memory cache of answers in `gateway/cache/semantic_cache.py`. Despite the file and class name, nothing in it is semantic. It matches requests exactly.

1. **Key.** The key is the SHA-256 of the request as JSON with sorted keys, after three changes. The gateway drops `stream` and `user`, and it strips leading and trailing whitespace from string message content. Every other request field is part of the key, including `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `stop`, and both penalties. Because `user` is not part of the key, every client shares one cache.
2. **Skipped requests.** The gateway neither reads nor writes the cache for a request with `n` greater than 1, or for one that carries any field the schema does not model, such as `tools`, `response_format`, or `logprobs`. A replay cannot reproduce those responses.
3. **What it stores.** It stores the answer text, the upstream's `finish_reason`, and a `usage` object. For a stream, it stores the joined content once the stream ends normally, with the prompt estimate and the content-event count as `usage`. For a non-streaming response, it stores `choices[0].message.content` and the provider's `usage`. The cache does not store an empty answer.
4. **Expiry and eviction.** An entry expires `cache_ttl_seconds` (300 s by default) after it was stored, and the cache checks expiry on read. When the cache holds `cache_max_size` (1000) entries, storing a new one evicts the oldest stored entry. Reads do not refresh an entry, so eviction is first in, first out.
5. **Replay.** A hit makes no provider call and carries `X-Cache: HIT`. A non-streaming hit returns a `chat.completion` object with an id that starts `chatcmpl-cache-` and the stored `usage`. A streaming hit sends the whole stored text as one SSE chunk, followed by `data: [DONE]`, with no separate role chunk. Both shapes report the stored `finish_reason`, so an answer the upstream cut off replays as `length`, not `stop`.

A miss carries `X-Cache: MISS` when the cache is enabled. A hit increments `llm_cache_hits_total` and nothing else. It creates no `chat` span, no `llm_requests_total` sample, and no latency sample. Nothing times the cache path.

---

## 8. What this gateway does not do

- **Share state.** Breakers, rate-limit buckets, and the cache live in memory in one process. Workers and replicas do not share them, and a restart resets them.
- **Authenticate clients.** The Bearer token, or the client IP, is only the rate-limit key. Anyone who can reach the gateway can call `/v1/circuit-breakers/{provider}/trip`, `/v1/circuit-breakers/{provider}/reset`, and `/v1/cache/clear`. CORS allows every origin, with credentials.
- **Export spans.** No span processor or exporter is configured, and `otel_exporter_endpoint` is unused (section 6).
- **Match similar prompts.** The cache matches exact requests only, evicts oldest first, and replays a stream as one chunk (section 7).
- **Record cache hits as requests.** A cache hit is charged against the rate limit but creates no span and no `llm_requests_total` sample.
- **Retry.** Failover is a single hop to one fallback, tried once, with no retries or backoff, and only before response headers are sent. After headers, a failure can only end in an SSE error frame. While a HALF-OPEN primary's probe is out, other requests go to the fallback.
- **Count tokens exactly.** Prompt tokens are an estimate. Streaming completion "tokens" are content events, and the gateway ignores upstream `usage` on streams. The rate limiter and the cached `usage` both use these numbers.
- **Accept non-text content.** Message content must be a string or a list of `text` parts. Image and audio parts fail validation (`gateway/schemas/openai.py`).
- **Change breaker state on a timer.** The OPEN to HALF-OPEN change happens on the next request or status read.
- **Report overhead under load.** The overhead benchmark runs one request at a time against a mock upstream, so no figure from it applies to concurrent traffic or a real network.

---

## 9. References

### Specifications and standards
- [WHATWG HTML Living Standard, §9.2 Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html). The event-stream format, its parsing rules, and the `EventSource` interface.
- [RFC 9113: HTTP/2](https://www.rfc-editor.org/rfc/rfc9113.html). Section 8.1 says HTTP/2 cannot use chunked transfer encoding.
- [OpenTelemetry GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai). The current home of the GenAI span attributes and metrics. The old page at opentelemetry.io/docs/specs/semconv/gen-ai/ now says the conventions have moved here.
- [OpenTelemetry semantic conventions](https://github.com/open-telemetry/semantic-conventions). The core registry, including the table of deprecated GenAI attributes.
- [OpenTelemetry Python API reference](https://opentelemetry-python.readthedocs.io/en/latest/). The landing page for the Python API and SDK reference.
- [OpenAI API reference: Chat Completions](https://developers.openai.com/api/reference/resources/chat). The request and response shape that `gateway/schemas/openai.py` models.
- [OpenMetrics specification](https://github.com/prometheus/OpenMetrics/blob/main/specification/OpenMetrics.md). The metric exposition format that Prometheus scrapes.

### Patterns
- [Michael Nygard, *Release It!* Second Edition](https://pragprog.com/titles/mnee2/release-it-second-edition/). The book that popularized the circuit breaker, which it lists under stability patterns.
- [Martin Fowler, CircuitBreaker](https://martinfowler.com/bliki/CircuitBreaker.html). A short essay on the pattern and its half-open state.
- [Leaky bucket (Wikipedia)](https://en.wikipedia.org/wiki/Leaky_bucket). Describes the meter and queue variants, and cites ITU-T Recommendation I.371 for the meter form used in ATM networks.

### Libraries
- [FastAPI](https://fastapi.tiangolo.com/). The web framework that serves the gateway.
- [Starlette StreamingResponse](https://starlette.dev/responses/#streamingresponse). The response class that streams the gateway's SSE body from an async generator.
- [HTTPX](https://www.python-httpx.org/). The async HTTP client the gateway uses for upstream calls, with connection pooling and streaming downloads.
- [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/). The validation library behind the request schemas and settings. Its core validation logic is written in Rust.
- [Locust](https://locust.io/). The load-testing tool behind `benchmarks/locustfile.py`. Tests are written in Python and can run across several machines.
