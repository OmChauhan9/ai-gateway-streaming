"""High-performance token estimation heuristic.

In production gateways, running full BPE tokenization (tiktoken / tokenizers) on every
request CPU-blocks the asyncio event loop under thousands of concurrent requests.
Top-tier gateways use calibrated character/subword heuristics for prompt admission:
~4 characters per token for English text, with formatting framing adjustments.
"""

import json
from typing import List, Union

from pydantic import BaseModel

from gateway.schemas.openai import ChatCompletionRequest, ChatMessage


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for a raw string.
    
    Formula:
    tokens = max(1, ceil(len(text) / 4))
    Adjusted slightly for whitespace density.
    """
    if not text:
        return 0
    char_len = len(text)
    # Average 4.0 chars per token for English/code
    # A minimum of 1 token if non-empty
    estimated = max(1, (char_len + 3) // 4)
    return estimated


def estimate_prompt_tokens(messages: Union[List[ChatMessage], List[dict]]) -> int:
    """Estimate total prompt tokens for an OpenAI chat completion payload.
    
    Mirrors standard OpenAI chat formatting overhead:
    - Every message is wrapped in <|im_start|>{role}\n{content}<|im_end|>\n (approx 3-4 tokens)
    - 3 tokens added to prime the assistant response
    """
    num_tokens = 0
    for message in messages:
        num_tokens += 3  # Message framing overhead (<|im_start|>{role}...\n)
        if isinstance(message, ChatMessage):
            role = message.role or ""
            if isinstance(message.content, list):
                content = "".join(part.text for part in message.content)
            else:
                content = message.content or ""
            name = message.name or ""
            num_tokens += _forwarded_field_tokens(message)
        elif isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
            name = message.get("name", "")
        else:
            role = getattr(message, "role", "")
            content = getattr(message, "content", "")
            name = getattr(message, "name", "")
            
        num_tokens += estimate_text_tokens(role)
        num_tokens += estimate_text_tokens(content)
        if name:
            num_tokens += estimate_text_tokens(name) + 1

    num_tokens += 3  # Priming tokens for assistant response
    return max(1, num_tokens)


def _forwarded_field_tokens(model: BaseModel) -> int:
    # Fields the schema does not model (tools, tool_calls, response_format...) reach the
    # upstream verbatim, so their JSON is prompt input like any message text.
    return estimate_text_tokens(json.dumps(model.model_extra)) if model.model_extra else 0


def estimate_request_tokens(request: ChatCompletionRequest) -> int:
    """Estimate prompt tokens for everything the upstream receives: messages and forwarded fields."""
    return estimate_prompt_tokens(request.messages) + _forwarded_field_tokens(request)
