"""High-speed exact and normalized semantic cache for LLM requests.

Eliminates redundant upstream model invocations by hashing canonical request structures
(model, normalized messages, temperature, top_p) with time-to-live (TTL) expiration.
"""

import hashlib
import json
import time
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
from gateway.schemas.openai import ChatCompletionRequest, ChatCompletionResponse


class CacheEntry:
    def __init__(self, response_text: str, created_at: float, ttl: float, usage: dict):
        self.response_text = response_text
        self.created_at = created_at
        self.ttl = ttl
        self.usage = usage

    def is_expired(self, now: float) -> bool:
        return (now - self.created_at) > self.ttl


class SemanticCache:
    """In-memory LRU-like TTL cache for LLM completions."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000, enabled: bool = True):
        self.ttl = float(ttl_seconds)
        self.max_size = max_size
        self.enabled = enabled
        self._store: Dict[str, CacheEntry] = {}

    def _generate_key(self, request: ChatCompletionRequest) -> str:
        """Derive a deterministic SHA256 key from canonical request parameters."""
        normalized_messages = [
            {"role": m.role, "content": m.content.strip()}
            for m in request.messages
        ]
        key_payload = {
            "model": request.model,
            "messages": normalized_messages,
            "temperature": request.temperature if request.temperature is not None else 1.0,
            "top_p": request.top_p if request.top_p is not None else 1.0,
        }
        serialized = json.dumps(key_payload, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    async def get(self, request: ChatCompletionRequest) -> Optional[Tuple[str, dict]]:
        """Look up cached completion for request.
        
        Returns:
            Optional[Tuple[str, dict]]: (response_text, usage_dict) if found and not expired.
        """
        if not self.enabled:
            return None

        key = self._generate_key(request)
        entry = self._store.get(key)
        if not entry:
            return None

        now = time.monotonic()
        if entry.is_expired(now):
            del self._store[key]
            return None

        return entry.response_text, entry.usage

    async def set(
        self,
        request: ChatCompletionRequest,
        response_text: str,
        usage: Optional[dict] = None,
    ) -> None:
        """Store completion in cache."""
        if not self.enabled:
            return

        if len(self._store) >= self.max_size:
            # Simple eviction of oldest item
            oldest_key = next(iter(self._store))
            del self._store[oldest_key]

        key = self._generate_key(request)
        now = time.monotonic()
        self._store[key] = CacheEntry(
            response_text=response_text,
            created_at=now,
            ttl=self.ttl,
            usage=usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def clear(self) -> None:
        self._store.clear()
