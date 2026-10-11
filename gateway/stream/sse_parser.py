"""High-performance Server-Sent Events (SSE) byte chunk parser.

Optimized to process streaming tokens without JSON deserialization bottlenecks:
Instead of calling json.loads() on every SSE chunk (which consumes significant CPU at high QPS),
we use zero-copy byte searching (`find(b'"content"')`) to detect and count delta tokens.
"""

import json
from typing import Optional, Tuple


def extract_content_from_sse_chunk(chunk: bytes) -> Optional[str]:
    """Extract content string from raw SSE chunk using fast byte indexing.
    
    Expected format:
    data: {"choices": [{"delta": {"content": "..."}}]}
    """
    if not chunk:
        return None

    # Check if this is a standard SSE data frame
    data_idx = chunk.find(b"data: ")
    if data_idx == -1:
        return None

    payload = chunk[data_idx + 6:].strip()
    if payload == b"[DONE]":
        return None

    # Fast path: search for "content":
    content_marker = b'"content":'
    content_pos = payload.find(content_marker)
    if content_pos == -1:
        return None

    # Locate the value after "content":
    val_start = content_pos + len(content_marker)
    # Skip leading whitespace
    while val_start < len(payload) and payload[val_start : val_start + 1] in b" \t":
        val_start += 1

    if val_start >= len(payload):
        return None

    # If content is null (e.g. role announcement chunk)
    if payload.startswith(b"null", val_start):
        return None

    # If quoted string
    if payload[val_start : val_start + 1] == b'"':
        val_start += 1
        # Find closing unescaped quote
        idx = val_start
        found_end = -1
        while idx < len(payload):
            if payload[idx : idx + 1] == b'"':
                # Check escape count before this quote
                backslashes = 0
                k = idx - 1
                while k >= val_start and payload[k : k + 1] == b'\\':
                    backslashes += 1
                    k -= 1
                if backslashes % 2 == 0:
                    found_end = idx
                    break
            idx += 1

        if found_end != -1:
            raw_slice = payload[val_start:found_end]
            # Quick decode: if no escapes, direct utf-8 decode
            if b"\\" not in raw_slice:
                try:
                    return raw_slice.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            # Fallback for unicode escapes
            try:
                return json.loads(b'"' + raw_slice + b'"')
            except Exception:
                return raw_slice.decode("utf-8", errors="ignore")

    # Resilient fallback: standard JSON load if edge case encountered
    try:
        data = json.loads(payload)
        choices = data.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content")
    except Exception:
        pass

    return None


def parse_sse_event(raw_event: bytes) -> Tuple[Optional[str], Optional[bytes]]:
    """Parse raw SSE block into event name and data payload."""
    event_type = None
    data_lines = []

    for line in raw_event.split(b"\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(b"event: "):
            event_type = line[7:].decode("utf-8", errors="ignore")
        elif line.startswith(b"data: "):
            data_lines.append(line[6:])

    data_payload = b"\n".join(data_lines) if data_lines else None
    return event_type, data_payload
