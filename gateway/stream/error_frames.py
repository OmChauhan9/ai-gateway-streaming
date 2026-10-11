"""Standardized Server-Sent Events (SSE) error frames.

When an upstream LLM connection drops or aborts mid-stream AFTER HTTP 200 headers
have already been committed to the client, the gateway cannot change the HTTP status code.
Instead, it emits a standardized SSE error event frame before terminating the stream.
"""

import json


def generate_sse_error_frame(
    message: str = "Upstream provider connection interrupted mid-stream",
    error_type: str = "upstream_stream_interrupted",
    code: int = 502,
    provider: str = "unknown",
) -> bytes:
    """Format an SSE error event and [DONE] termination chunk."""
    error_payload = {
        "error": {
            "message": message,
            "type": error_type,
            "code": code,
            "provider": provider,
        }
    }
    json_bytes = json.dumps(error_payload).encode("utf-8")
    frame = (
        b"event: error\n"
        b"data: " + json_bytes + b"\n\n"
        b"data: [DONE]\n\n"
    )
    return frame
