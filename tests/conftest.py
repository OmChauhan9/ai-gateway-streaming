"""Pytest configuration and shared fixtures for AI Gateway tests.

Uses in-memory ASGI transports to test streaming SSE, circuit breakers,
and failover without needing live external networks or hardcoded ports.
"""

import asyncio
from typing import AsyncIterator, Tuple
import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from config.settings import Settings
from gateway.main import create_app
from gateway.upstream.client import AsyncUpstreamClient
from gateway.upstream.mock_server import create_mock_upstream_app


class MultiAppTransport(httpx.AsyncBaseTransport):
    """Routes HTTP requests to distinct in-memory ASGI apps based on port/URL.
    
    127.0.0.1:8001 -> Primary Mock Upstream (e.g. Groq)
    127.0.0.1:8002 -> Fallback Mock Upstream (e.g. Together)
    """
    def __init__(self, primary_app: FastAPI, fallback_app: FastAPI):
        self.primary_transport = httpx.ASGITransport(app=primary_app)
        self.fallback_transport = httpx.ASGITransport(app=fallback_app)

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
    transport = httpx.ASGITransport(app=gateway_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
