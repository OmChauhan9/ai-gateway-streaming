"""Incremental Server-Sent Events (SSE) parser for upstream completion streams.

Network reads do not line up with SSE events: one read can carry several events,
and one event can span several reads. The parser buffers bytes across reads and
frames them per the WHATWG event-stream rules (HTML Living Standard, 9.2.6).
"""

import json
import re
from typing import AsyncIterator, Optional, Tuple

_LINE_END = re.compile(rb"\r\n|\r|\n")


def _data_value(line: bytes) -> Optional[bytes]:
    field, _, value = line.partition(b":")
    if field != b"data":
        return None
    return value[1:] if value.startswith(b" ") else value


def _is_whole(data: bytes) -> bool:
    if data == b"[DONE]":
        return True
    try:
        json.loads(data)
    except ValueError:
        return False
    return True


def _first_choice(data: bytes) -> Tuple[Optional[str], Optional[str]]:
    try:
        choice = json.loads(data)["choices"][0]
        finish_reason = choice.get("finish_reason")
    except (ValueError, LookupError, TypeError, AttributeError):
        return None, None
    try:
        content = choice["delta"]["content"]
    except (LookupError, TypeError):
        content = None
    return (content if isinstance(content, str) and content else None), finish_reason


async def iter_sse_events(chunks: AsyncIterator[bytes]) -> AsyncIterator[Tuple[bytes, Optional[str], Optional[str]]]:
    """Regroup raw byte chunks into complete SSE events.

    Yields `(raw_event, content, finish_reason)` for each event that a blank line terminates,
    where `raw_event` is the event's exact upstream bytes, `content` is its non-empty
    `choices[0].delta.content` or None, and `finish_reason` is `choices[0].finish_reason`
    or None. When `chunks` ends, an event still missing its blank line is
    yielded with the blank line appended if its data is whole JSON or `[DONE]`, and is
    dropped as a cut-off fragment otherwise.
    """
    buf = b""
    pos = 0
    data_lines = []
    pending_cr = False
    async for chunk in chunks:
        if not chunk:
            continue
        buf += chunk
        # A CR that ended the previous read may be the first half of a CRLF.
        if pending_cr and buf[pos : pos + 1] == b"\n":
            pos += 1
        pending_cr = False
        while (match := _LINE_END.search(buf, pos)) is not None:
            line = buf[pos : match.start()]
            pos = match.end()
            pending_cr = match.group() == b"\r" and pos == len(buf)
            if line:
                if (value := _data_value(line)) is not None:
                    data_lines.append(value)
                continue
            content, finish_reason = _first_choice(b"\n".join(data_lines)) if data_lines else (None, None)
            yield buf[:pos], content, finish_reason
            buf, pos, data_lines = buf[pos:], 0, []
    if pos < len(buf):
        if (value := _data_value(buf[pos:])) is not None:
            data_lines.append(value)
        buf += b"\n"
    data = b"\n".join(data_lines)
    if data_lines and _is_whole(data):
        content, finish_reason = _first_choice(data)
        # The appended blank line lets clients dispatch it and keeps a following error frame a separate event.
        yield buf + (b"\r" if pending_cr else b"\n"), content, finish_reason
