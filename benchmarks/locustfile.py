"""Locust load test: concurrent streaming chat completions against a running gateway.

Locust's own entry for `POST /v1/chat/completions` stops its clock when the response headers arrive,
because the request is made with stream=True, so it says nothing about how long the stream took.
Each streamed request therefore also reports two custom entries, timed on the client:
- `stream:ttft` times the request until the first event whose `choices[0].delta.content` is a non-empty
  string. The gateway's own TTFT uses the same rule, so the role-only first event does not stop the clock.
- `stream:full` times the request until `data: [DONE]`.
A non-200 response, or a stream that ends without `data: [DONE]`, is a failure.

What it measures: the gateway in front of the mock upstreams (canned tokens with sleeps, not a real LLM),
on whatever machine runs both, with Locust's single-process client. What it does not: a real provider,
a real network, or the gateway without the mocks' sleeps (30 ms before the first token, 10 ms per token).

All users send the same API key and so share one rate-limiter bucket. At the default limits (50,000
tokens, 2,000 tokens/s) 50 users drain it within seconds and the gateway answers 429. Those 429s are the
limiter doing its job, not a gateway failure, and they stay failures here. The response cache is on by
default and every user sends the same prompt, so after the first request most responses would be cache
hits; turn it off to load the proxy path. From the repository root, in three shells:

    PYTHONPATH=. .venv/bin/python scripts/run_mock_upstream.py

    CACHE_ENABLED=false RATE_LIMITER_CAPACITY_TOKENS=10000000 \\
    RATE_LIMITER_LEAK_RATE_TOKENS_PER_SECOND=1000000 \\
    PYTHONPATH=. .venv/bin/python scripts/run_gateway.py

    .venv/bin/locust -f benchmarks/locustfile.py --host http://127.0.0.1:8000 \\
        --users 50 --spawn-rate 10 --run-time 30s --headless

Settings reads these environment variables case-insensitively (pydantic-settings), so the uppercase
names map to the fields in config/settings.py.
"""

import json
import time

from locust import HttpUser, between, task

PAYLOAD = {
    "model": "llama3-70b-8192",
    "messages": [
        {"role": "system", "content": "You are a low-latency enterprise AI assistant."},
        {"role": "user", "content": "Explain how an async AI gateway optimizes streaming inference latency."},
    ],
    "stream": True,
    "temperature": 0.7,
}


def _has_content(data: bytes) -> bool:
    try:
        content = json.loads(data)["choices"][0]["delta"]["content"]
    except (ValueError, LookupError, TypeError):
        return False
    return isinstance(content, str) and content != ""


class AIGatewayStreamingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def _report(self, name: str, started: float, ended: float, length: int, error: Exception | None) -> None:
        self.environment.events.request.fire(
            request_type="SSE",
            name=name,
            response_time=(ended - started) * 1000.0,
            response_length=length,
            exception=error,
            context={},
        )

    @task(3)
    def stream_chat_completion(self):
        t_start = time.perf_counter()
        t_first = None
        length = 0
        done = False

        with self.client.post(
            "/v1/chat/completions",
            json=PAYLOAD,
            headers={"Authorization": "Bearer locust-stress-test-key"},
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Gateway returned status {response.status_code}")
                return

            for line in response.iter_lines():
                length += len(line)
                if not line.startswith(b"data: "):
                    continue
                if line[6:].strip() == b"[DONE]":
                    done = True
                    break
                if t_first is None and _has_content(line[6:]):
                    t_first = time.perf_counter()
            t_end = time.perf_counter()

            if not done:
                response.failure("stream ended without data: [DONE]")
            else:
                response.success()

        no_token = None if t_first is not None else RuntimeError("stream carried no content token")
        self._report("stream:ttft", t_start, t_first or t_end, 0, no_token)
        incomplete = None if done else RuntimeError("stream ended without data: [DONE]")
        self._report("stream:full", t_start, t_end, length, incomplete)

    @task(1)
    def check_health_and_metrics(self):
        self.client.get("/metrics")
        self.client.get("/health")
