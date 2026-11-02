"""Tests for how the upstream client classifies provider responses."""

import httpx
import pytest

from gateway.circuit_breaker.provider_registry import ProviderInfo
from gateway.circuit_breaker.state_machine import CircuitBreaker
from gateway.schemas.openai import ChatCompletionRequest
from gateway.upstream.client import AsyncUpstreamClient, UpstreamServerError


class _BodyCutOff(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"err'
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


@pytest.mark.asyncio
async def test_5xx_whose_body_is_cut_off_is_still_a_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, headers={"content-length": "100"}, stream=_BodyCutOff())

    client = AsyncUpstreamClient(transport=httpx.MockTransport(handler))
    provider = ProviderInfo(
        name="p", base_url="http://p/v1", api_key="", circuit_breaker=CircuitBreaker("p")
    )
    payload = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}], stream=True)
    try:
        with pytest.raises(UpstreamServerError) as info:
            await client.send_streaming_request(provider, payload)
        assert info.value.status_code == 503, "classified by the status line, not the lost body"
    finally:
        await client.close()
