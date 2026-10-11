"""Runner script to start synthetic multi-provider mock upstreams.

Runs:
- Primary Provider (Groq / Ollama simulation) on port 8001
- Fallback Provider (Together / OpenAI simulation) on port 8002
"""

import multiprocessing
import uvicorn
from gateway.upstream.mock_server import create_mock_upstream_app


def start_primary():
    app = create_mock_upstream_app("groq-primary")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")


def start_fallback():
    app = create_mock_upstream_app("together-fallback")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")


def main():
    print(f"\n{'='*65}")
    print("  Starting Synthetic Upstream LLM Providers")
    print("  Primary Provider (groq-primary):      http://127.0.0.1:8001")
    print("  Fallback Provider (together-fallback): http://127.0.0.1:8002")
    print(f"{'='*65}\n")

    p1 = multiprocessing.Process(target=start_primary, name="mock-primary")
    p2 = multiprocessing.Process(target=start_fallback, name="mock-fallback")

    p1.start()
    p2.start()

    try:
        p1.join()
        p2.join()
    except KeyboardInterrupt:
        print("\nStopping mock upstreams...")
        p1.terminate()
        p2.terminate()


if __name__ == "__main__":
    main()
