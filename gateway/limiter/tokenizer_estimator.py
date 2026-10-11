"""High-performance token estimation heuristic.

In production gateways, running full BPE tokenization (tiktoken / tokenizers) on every
request CPU-blocks the asyncio event loop under thousands of concurrent requests.
Top-tier gateways use calibrated character/subword heuristics for prompt admission:
~4 characters per token for English text, with formatting framing adjustments.
"""

from typing import List, Union
from gateway.schemas.openai import ChatMessage


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
            content = message.content or ""
            name = message.name or ""
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
