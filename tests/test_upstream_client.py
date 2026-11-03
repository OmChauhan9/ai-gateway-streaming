"""Tests for how the upstream client classifies provider responses."""

import json

import httpx
import pytest

from gateway.circuit_breaker.provider_registry import ProviderInfo
from gateway.circuit_breaker.state_machine import CircuitBreaker
from gateway.main import create_app
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


async def _bodies_received_by_upstream(settings, sent: dict) -> list:
    received = []

    def upstream(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        if received[-1]["stream"]:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})

    gateway = create_app(settings=settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway), base_url="http://test") as c:
        resp = await c.post("/v1/chat/completions", json=sent)
    assert resp.status_code == 200, resp.text
    return received


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
async def test_upstream_receives_exactly_the_fields_the_client_sent(test_settings, stream):
    sent = {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {"role": "tool", "content": "18C", "tool_call_id": "call_1"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "seed": 42,
        "logprobs": True,
        "temperature": 0.2,
        "stream": stream,
    }
    if stream:
        sent["stream_options"] = {"include_usage": True}

    received = await _bodies_received_by_upstream(test_settings, sent)

    assert received == [sent], "unmodeled fields forwarded unchanged, no defaults the client never set"


@pytest.mark.asyncio
async def test_upstream_body_for_a_request_without_stream_adds_only_stream_false(test_settings):
    sent = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}

    received = await _bodies_received_by_upstream(test_settings, sent)

    assert received == [{**sent, "stream": False}]


@pytest.mark.asyncio
async def test_message_without_role_reaches_the_upstream_as_a_user_message(test_settings):
    sent = {"model": "gpt-4o", "messages": [{"content": "hi"}]}

    received = await _bodies_received_by_upstream(test_settings, sent)

    assert received[0]["messages"] == [{"role": "user", "content": "hi"}], "the default role is always forwarded"


def _tool_loop_second_turn(stream: bool) -> dict:
    return {
        "model": "gpt-4o",
        "messages": [
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}],
            },
            {"role": "tool", "content": "18C", "tool_call_id": "call_1"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}],
        "stream": stream,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
async def test_tool_loop_second_turn_reaches_the_upstream_unchanged(test_settings, stream):
    sent = _tool_loop_second_turn(stream)

    received = await _bodies_received_by_upstream(test_settings, sent)

    assert received == [sent], "the assistant's content: null and tool_calls reach the upstream as sent"


@pytest.mark.asyncio
async def test_text_part_content_reaches_the_upstream_unchanged(test_settings):
    sent = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]}],
        "stream": False,
    }

    received = await _bodies_received_by_upstream(test_settings, sent)

    assert received == [sent], "list-of-text-parts content is forwarded as sent"


NON_FINITE_BODIES = {
    "top_level_extra": '{"model":"gpt-4o","messages":[{"role":"user","content":"x"}],"seed":NaN,"stream":%s}',
    "message_extra": '{"model":"gpt-4o","messages":[{"role":"user","content":"x","weight":Infinity}],"stream":%s}',
    "nested": '{"model":"gpt-4o","messages":[{"role":"user","content":"x"}],"metadata":{"w":[1,-Infinity]},"stream":%s}',
    "modeled_field": '{"model":"gpt-4o","messages":[{"role":"user","content":"x"}],"temperature":NaN,"stream":%s}',
    "overflowing_literal": '{"model":"gpt-4o","messages":[{"role":"user","content":"x"}],"seed":1e999,"stream":%s}',
    "text_part": '{"model":"gpt-4o","messages":[{"role":"user","content":[{"type":"text","text":"x","w":NaN}]}],"stream":%s}',
    "tool_calls": (
        '{"model":"gpt-4o","messages":[{"role":"user","content":"x"},{"role":"assistant","content":null,'
        '"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"},"w":Infinity}]}],"stream":%s}'
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["stream", "nonstream"])
@pytest.mark.parametrize("body", NON_FINITE_BODIES.values(), ids=NON_FINITE_BODIES.keys())
async def test_non_finite_number_is_a_client_error_with_no_upstream_call(test_settings, stream, body):
    calls = []

    def upstream(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]})

    gateway = create_app(settings=test_settings, client=AsyncUpstreamClient(transport=httpx.MockTransport(upstream)))
    transport = httpx.ASGITransport(app=gateway, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post(
            "/v1/chat/completions",
            content=body % ("true" if stream else "false"),
            headers={"Content-Type": "application/json"},
        )
        breakers = (await c.get("/v1/circuit-breakers")).json()

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]["error"]["type"] == "invalid_request_error"
    assert calls == [], "a body no upstream can accept is not sent"
    assert all(b["circuit"]["recent_failures_count"] == 0 for b in breakers.values()), "no provider is charged for it"
