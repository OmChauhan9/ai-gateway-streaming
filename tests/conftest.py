"""Pytest configuration and shared fixtures for AI Gateway tests.

Uses in-memory ASGI transports to test streaming SSE, circuit breakers,
and failover without needing live external networks or hardcoded ports.
"""

import asyncio
import contextlib
from typing import AsyncIterator, Tuple
import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from config.settings import Settings
from gateway.main import create_app
from gateway.upstream.client import AsyncUpstreamClient
from gateway.upstream.mock_server import create_mock_upstream_app


class StreamingASGITransport(httpx.AsyncBaseTransport):
    """In-memory ASGI transport that returns once the app sends response headers
    and streams body chunks as the app sends them.

    httpx.ASGITransport runs the app to completion and returns the body as one
    chunk, so an app failure mid-body surfaces before headers. Here it surfaces
    while the body is read, as httpx.RemoteProtocolError, the way a dropped
    socket does.
    """

    def __init__(self, app: FastAPI):
        self.app = app

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": request.method,
            "headers": [(k.lower(), v) for (k, v) in request.headers.raw],
            "scheme": request.url.scheme,
            "path": request.url.path,
            "raw_path": request.url.raw_path.split(b"?")[0],
            "query_string": request.url.query,
            "server": (request.url.host, request.url.port),
            "client": ("127.0.0.1", 123),
            "root_path": "",
        }
        request_body = request.stream.__aiter__()
        request_complete = False
        disconnected = asyncio.Event()
        sent: asyncio.Queue = asyncio.Queue()
        app_done = object()

        async def receive():
            nonlocal request_complete
            if request_complete:
                await disconnected.wait()
                return {"type": "http.disconnect"}
            try:
                body = await request_body.__anext__()
            except StopAsyncIteration:
                request_complete = True
                return {"type": "http.request", "body": b"", "more_body": False}
            return {"type": "http.request", "body": body, "more_body": True}

        async def run_app():
            try:
                await self.app(scope, receive, sent.put)
            finally:
                sent.put_nowait(app_done)

        task = asyncio.create_task(run_app())

        start = await sent.get()
        if start is app_done:
            await task
            raise RuntimeError("ASGI app returned without starting a response")

        class BodyStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    message = await sent.get()
                    if message is app_done:
                        raise httpx.RemoteProtocolError(
                            "peer closed connection without sending complete message body", request=request
                        ) from task.exception()
                    if request.method != "HEAD" and message.get("body"):
                        yield message["body"]
                    if not message.get("more_body", False):
                        return

            async def aclose(self):
                disconnected.set()
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        return httpx.Response(start["status"], headers=start.get("headers", []), stream=BodyStream())


class MultiAppTransport(httpx.AsyncBaseTransport):
    """Routes HTTP requests to distinct in-memory ASGI apps based on port/URL.
    
    127.0.0.1:8001 -> Primary Mock Upstream (e.g. Groq)
    127.0.0.1:8002 -> Fallback Mock Upstream (e.g. Together)
    """
    def __init__(self, primary_app: FastAPI, fallback_app: FastAPI):
        self.primary_transport = StreamingASGITransport(primary_app)
        self.fallback_transport = StreamingASGITransport(fallback_app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if ":8001" in url_str:
            return await self.primary_transport.handle_async_request(request)
        elif ":8002" in url_str:
            return await self.fallback_transport.handle_async_request(request)
        else:
            return await self.primary_transport.handle_async_request(request)


@pytest.fixture
def mock_apps() -> Tuple[FastAPI, FastAPI]:
    """Return (primary_mock_app, fallback_mock_app)."""
    primary = create_mock_upstream_app("mock-groq-primary")
    fallback = create_mock_upstream_app("mock-together-fallback")
    return primary, fallback


@pytest.fixture
def test_settings() -> Settings:
    """Settings configured for test runs."""
    return Settings(
        app_name="AI Gateway Test",
        primary_provider_name="mock-groq-primary",
        primary_base_url="http://127.0.0.1:8001/v1",
        fallback_provider_name="mock-together-fallback",
        fallback_base_url="http://127.0.0.1:8002/v1",
        cb_failure_threshold=3,
        cb_failure_window_seconds=5.0,
        cb_recovery_timeout_seconds=0.2,  # fast recovery for tests
        cb_half_open_success_threshold=2,
        rate_limiter_capacity_tokens=1000,
        rate_limiter_leak_rate_tokens_per_second=200.0,
        cache_enabled=True,
    )


@pytest_asyncio.fixture
async def gateway_app(mock_apps, test_settings) -> FastAPI:
    primary_app, fallback_app = mock_apps
    transport = MultiAppTransport(primary_app, fallback_app)
    custom_client = AsyncUpstreamClient(transport=transport)
    app = create_app(settings=test_settings, client=custom_client)
    app.state.primary_mock_app = primary_app
    app.state.fallback_mock_app = fallback_app
    return app


@pytest_asyncio.fixture
async def client(gateway_app) -> AsyncIterator[httpx.AsyncClient]:
    transport = StreamingASGITransport(gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
