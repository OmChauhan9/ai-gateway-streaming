"""Unit tests for OpenAI schemas and validation."""

import pytest
from pydantic import ValidationError
from gateway.schemas.openai import (
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
)


def test_valid_chat_request():
    req = ChatCompletionRequest(
        model="llama3-70b-8192",
        messages=[
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello!"),
        ],
        stream=True,
        temperature=0.7,
    )
    assert req.model == "llama3-70b-8192"
    assert len(req.messages) == 2
    assert req.stream is True
    assert req.temperature == 0.7


def test_invalid_chat_request_empty_messages():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="llama3-70b-8192",
            messages=[],  # min_length=1
        )


def test_chunk_serialization():
    chunk = ChatCompletionChunk(
        model="llama3-8b",
        choices=[
            {
                "index": 0,
                "delta": {"content": "world"},
                "finish_reason": None,
            }
        ],
    )
    dumped = chunk.model_dump()
    assert dumped["object"] == "chat.completion.chunk"
    assert dumped["choices"][0]["delta"]["content"] == "world"
