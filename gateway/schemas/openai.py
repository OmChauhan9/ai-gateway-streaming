"""OpenAI-compatible Chat Completion API schemas.

Faithful implementation of the standard OpenAI v1 REST specification.
"""

import time
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextContentPart(BaseModel):
    """One `{"type": "text"}` entry of a message's list-form content."""
    model_config = ConfigDict(extra="allow")

    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    """Individual message in a chat conversation."""
    model_config = ConfigDict(extra="allow")

    role: Literal["developer", "system", "user", "assistant", "function", "tool"] = "user"
    # Null or absent for an assistant turn that only carries tool_calls. Non-text parts (images, audio)
    # stay rejected because the token estimator has no way to count them.
    content: Optional[Union[str, List[TextContentPart]]] = None
    name: Optional[str] = None

    @model_validator(mode="after")
    def _content_required_unless_assistant_calls_a_tool(self) -> "ChatMessage":
        calls_a_tool = any(self.model_extra.get(field) is not None for field in ("tool_calls", "function_call"))
        if "content" not in self.model_fields_set and not (self.role == "assistant" and calls_a_tool):
            raise ValueError("content is required unless an assistant message carries tool_calls")
        return self

    @model_validator(mode="after")
    def _always_forward_role(self) -> "ChatMessage":
        # The upstream dump keeps only fields the client set, but upstreams reject a
        # message with no role, so the "user" default must still be sent.
        self.model_fields_set.add("role")
        return self


class ChatCompletionRequest(BaseModel):
    """Inbound request model for POST /v1/chat/completions."""
    model_config = ConfigDict(extra="allow")

    model: str = Field(..., description="Target model identifier (e.g., llama3-70b-8192, gpt-4o)")
    messages: List[ChatMessage] = Field(..., min_length=1, description="List of messages comprising the conversation")
    stream: bool = Field(default=False, description="Whether to stream back partial progress using SSE")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=None, ge=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
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
