"""OpenAI-compatible Chat Completion API schemas.

Faithful implementation of the standard OpenAI v1 REST specification.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Individual message in a chat conversation."""
    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Inbound request model for POST /v1/chat/completions."""
    model: str = Field(..., description="Target model identifier (e.g., llama3-70b-8192, gpt-4o)")
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of messages comprising the conversation")
    stream: bool = Field(default=False, description="Whether to stream back partial progress using SSE")
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(default=None, description="End-user identifier for abuse detection")


class UsageInfo(BaseModel):
    """Token consumption metrics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChoiceMessage(BaseModel):
    """Assistant message in a non-streaming completion."""
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    """Single completion choice."""
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    """Non-streaming response model."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None


class DeltaMessage(BaseModel):
    """Incremental delta content within a streaming SSE chunk."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChoiceDelta(BaseModel):
    """Choice representation in a streaming chunk."""
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """SSE frame payload model for streaming chat completions."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChoiceDelta]
    system_fingerprint: Optional[str] = None
