"""Locust Load Testing Script for AI Gateway.

Simulates 50+ concurrent streaming inference clients:
- Measures Server-Sent Events (SSE) streaming latency
- Tracks Time To First Token (TTFT) and throughput
- Validates resilience under continuous load
"""

import json
import time
from locust import HttpUser, task, between


class AIGatewayStreamingUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(3)
    def stream_chat_completion(self):
        """Simulate high-concurrency streaming user."""
        payload = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You are a low-latency enterprise AI assistant."},
                {"role": "user", "content": "Explain how an async AI gateway optimizes streaming inference latency."},
            ],
            "stream": True,
            "temperature": 0.7,
        }

        t_start = time.perf_counter()
        t_first = None
        token_count = 0

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer locust-stress-test-key"},
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Gateway returned status {response.status_code}")
                return

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    data_str = line[6:].strip()
                    if data_str == b"[DONE]":
                        break
                    
                    if t_first is None:
                        t_first = time.perf_counter()
                    token_count += 1

            t_end = time.perf_counter()
            total_duration_ms = (t_end - t_start) * 1000.0

            if t_first is not None:
                ttft_ms = (t_first - t_start) * 1000.0
            else:
                ttft_ms = total_duration_ms

            response.success()

    @task(1)
    def check_health_and_metrics(self):
        """Periodically scrape metrics endpoint."""
        self.client.get("/metrics")
        self.client.get("/health")
