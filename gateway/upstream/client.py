"""Async HTTP Client for Upstream LLM Providers.

Configured with high-performance connection pooling, keep-alives, and timeouts.
Dispatches requests to upstream endpoints and normalizes network/protocol errors.
"""

import time
from typing import Any, AsyncIterator, Dict, Optional, Tuple
import httpx
from gateway.circuit_breaker.provider_registry import ProviderInfo
from gateway.schemas.openai import ChatCompletionRequest


class UpstreamException(Exception):
    def __init__(self, provider: str, status_code: Optional[int], message: str):
        self.provider = provider
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{provider}] HTTP {status_code or 'ERR'}: {message}")


class UpstreamConnectionError(UpstreamException):
    """Network connection failure, DNS failure, or connection timeout."""
    def __init__(self, provider: str, message: str):
        super().__init__(provider, None, message)


class UpstreamRateLimitedError(UpstreamException):
    """HTTP 429 Too Many Requests from upstream."""
    def __init__(self, provider: str, message: str):
        super().__init__(provider, 429, message)


class UpstreamServerError(UpstreamException):
    """HTTP 5xx Server Error from upstream."""
    def __init__(self, provider: str, status_code: int, message: str):
        super().__init__(provider, status_code, message)


class UpstreamClientError(UpstreamException):
    """HTTP 4xx Client Error (excluding 429) from upstream."""
    def __init__(self, provider: str, status_code: int, message: str):
        super().__init__(provider, status_code, message)


class AsyncUpstreamClient:
    """Production async HTTP client managing connections to upstream providers."""

    def __init__(
        self,
        max_connections: int = 500,
        max_keepalive: int = 100,
        keepalive_expiry: float = 30.0,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=keepalive_expiry,
        )
        self.client = httpx.AsyncClient(
            limits=limits,
            transport=transport,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.client.aclose()

    def _prepare_headers(self, provider: ProviderInfo) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json",
        }
        if provider.api_key:
            headers["Authorization"] = f"Bearer {provider.api_key}"
        return headers

    async def send_streaming_request(
        self,
        provider: ProviderInfo,
        request: ChatCompletionRequest,
    ) -> Tuple[httpx.Response, float]:
        """Initiate upstream streaming connection.
        
        Returns:
            Tuple[httpx.Response, float]: (open_streaming_response, t_start_monotonic)
        """
        url = f"{provider.base_url}/chat/completions"
        headers = self._prepare_headers(provider)
        payload = request.model_dump(exclude_none=True)
        # Ensure stream is True
        payload["stream"] = True

        timeout = httpx.Timeout(
            timeout=provider.timeout_seconds,
            connect=5.0,
        )

        t_start = time.perf_counter()
        req = self.client.build_request(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        )

        try:
            resp = await self.client.send(req, stream=True)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
            raise UpstreamConnectionError(provider.name, f"Connection failed: {str(exc)}") from exc
        except httpx.TimeoutException as exc:
            raise UpstreamConnectionError(provider.name, f"Request timed out: {str(exc)}") from exc
        except Exception as exc:
            raise UpstreamConnectionError(provider.name, f"Transport failure: {str(exc)}") from exc

        # Check status code before streaming body
        if resp.status_code >= 400:
            try:
                body_bytes = await resp.aread()
                body_text = body_bytes.decode("utf-8", errors="ignore")
            finally:
                await resp.aclose()

            if resp.status_code == 429:
                raise UpstreamRateLimitedError(provider.name, body_text)
            elif resp.status_code >= 500:
                raise UpstreamServerError(provider.name, resp.status_code, body_text)
            else:
                raise UpstreamClientError(provider.name, resp.status_code, body_text)

        return resp, t_start

    async def send_non_streaming_request(
        self,
        provider: ProviderInfo,
        request: ChatCompletionRequest,
    ) -> Tuple[dict, float]:
        """Dispatch non-streaming completion call."""
        url = f"{provider.base_url}/chat/completions"
        headers = self._prepare_headers(provider)
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = False

        timeout = httpx.Timeout(timeout=provider.timeout_seconds, connect=5.0)
        t_start = time.perf_counter()

        try:
            resp = await self.client.post(url, json=payload, headers=headers, timeout=timeout)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.NetworkError) as exc:
            raise UpstreamConnectionError(provider.name, f"Connection failed: {str(exc)}") from exc
        except httpx.TimeoutException as exc:
            raise UpstreamConnectionError(provider.name, f"Timeout: {str(exc)}") from exc
        except Exception as exc:
            raise UpstreamConnectionError(provider.name, f"Transport error: {str(exc)}") from exc

        if resp.status_code >= 400:
            body_text = resp.text
            if resp.status_code == 429:
                raise UpstreamRateLimitedError(provider.name, body_text)
            elif resp.status_code >= 500:
                raise UpstreamServerError(provider.name, resp.status_code, body_text)
            else:
                raise UpstreamClientError(provider.name, resp.status_code, body_text)

        return resp.json(), t_start
