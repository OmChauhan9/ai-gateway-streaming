"""Interactive Demo Client for AI Gateway.

Connects to the gateway, streams completions via Server-Sent Events (SSE),
and measures real-time client-side latency:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Throughput (tokens/sec)
- Active upstream provider & circuit breaker header inspection
"""

import argparse
import asyncio
import json
import sys
import time
import httpx


async def stream_chat(
    gateway_url: str = "http://127.0.0.1:8000",
    model: str = "llama3-70b-8192",
    prompt: str = "Explain the importance of inference latency profiling in production AI.",
    api_key: str = "demo-client-key",
) -> None:
    endpoint = f"{gateway_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a senior AI infrastructure engineer."},
            {"role": "user", "content": prompt},
        ],
        "stream": True,
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"\n{'='*70}")
    print(f"  AI GATEWAY STREAMING CLIENT")
    print(f"  Target: {endpoint}")
    print(f"  Model:  {model}")
    print(f"  Prompt: {prompt}")
    print(f"{'='*70}\n")
    print("Response Stream: ", end="", flush=True)

    t_start = time.perf_counter()
    t_first = None
    t_last = None
    token_count = 0
    received_chunks = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            async with client.stream("POST", endpoint, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    print(f"\n[ERROR {response.status_code}] {body.decode('utf-8', errors='ignore')}")
                    return

                provider_used = response.headers.get("X-Gateway-Provider", "unknown")
                fallback_used = response.headers.get("X-Gateway-Fallback", "unknown")
                cache_status = response.headers.get("X-Cache", "MISS")
                req_id = response.headers.get("X-Gateway-Request-Id", "unknown")

                async for line in response.aiter_lines():
                    now = time.perf_counter()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            choices = chunk.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    if t_first is None:
                                        t_first = now
                                    t_last = now
                                    token_count += 1
                                    sys.stdout.write(content)
                                    sys.stdout.flush()
                        except json.JSONDecodeError:
                            pass

        except Exception as exc:
            print(f"\n[CLIENT EXCEPTION] {str(exc)}")
            return

    t_end = time.perf_counter()
    print("\n")

    # Metrics Summary
    total_latency_ms = (t_end - t_start) * 1000.0
    ttft_ms = ((t_first - t_start) * 1000.0) if t_first else 0.0
    tpot_ms = (((t_last - t_first) / (token_count - 1)) * 1000.0) if (t_first and token_count > 1) else 0.0
    throughput = (token_count / ((t_end - t_first))) if (t_first and (t_end - t_first) > 0) else 0.0

    print(f"{'='*70}")
    print(f"                       TELEMETRY PROFILE")
    print(f"{'='*70}")
    print(f"  Request ID:          {req_id}")
    print(f"  Active Provider:     {provider_used} (Fallback Diverted: {fallback_used})")
    print(f"  Cache Status:        {cache_status}")
    print(f"  Tokens Generated:    {token_count} tokens")
    print(f"  TTFT (First Token):  {ttft_ms:.2f} ms")
    print(f"  TPOT (Per Token):    {tpot_ms:.2f} ms / token")
    print(f"  Throughput:          {throughput:.1f} tokens / second")
    print(f"  Total Wall Time:     {total_latency_ms:.2f} ms")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Gateway Demo Client")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Gateway URL")
    parser.add_argument("--model", default="llama3-70b-8192", help="Model name")
    parser.add_argument("--prompt", default="Explain the importance of latency profiling in production AI.", help="User prompt")
    parser.add_argument("--key", default="demo-client-key", help="Client API key")
    args = parser.parse_args()

    asyncio.run(stream_chat(args.url, args.model, args.prompt, args.key))
