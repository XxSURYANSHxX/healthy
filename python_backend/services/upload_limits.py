"""How large an upload may be, decided before the upload is read.

The size limits were enforced after the fact. A request declaring ten
gigabytes was accepted, received in full, spooled, parsed, and only then
measured and refused - so the cheapest way to occupy the service was to send
something it was always going to reject. Worse, the per-format check read
``limit + 1`` bytes into one buffer first, which allocated 256 MB to decide
that 256 MB was too much.

Two bounds, in the order a request meets them:

1. **The declared length**, before any of the body is read. A request whose
   ``Content-Length`` exceeds the ceiling is answered 413 with nothing read.
2. **The bytes actually delivered**, counted as the framework pulls them. A
   request that declares nothing, declares a lie, or arrives chunked is cut
   off the moment the running total passes the ceiling.

The ceiling here is one number for the whole service, sized to the largest
format it accepts. The per-format limits are stricter and still apply, but
they are decisions about a file whose type is known, and the type is not known
until some of it has been read; this is the bound that holds before then.

Nothing in the refusal names a path, a limit source or an internal component.
"""

from __future__ import annotations

import threading
from typing import Awaitable, Callable, Dict, Optional

#: The largest body any route accepts, whatever it claims to be. Sized to the
#: largest per-format limit plus room for multipart framing: a file exactly at
#: its format's limit must not be refused for the boundary lines around it.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024
MAX_REQUEST_BYTES = 256 * 1024 * 1024 + MULTIPART_OVERHEAD_BYTES

#: What a refused request is told. Fixed strings: a caller learns that the
#: upload was too large, not how large the service can take, which component
#: decided, or where anything is stored.
TOO_LARGE_CODE = "upload_too_large"
TOO_LARGE_MESSAGE = "The file is too large to upload."
MALFORMED_LENGTH_CODE = "invalid_content_length"
MALFORMED_LENGTH_MESSAGE = "The request could not be read."


def declared_length(headers: Dict[bytes, bytes]) -> Optional[int]:
    """The declared body length, or None when there is not a usable one.

    A header that is present but not a number is a malformed request rather
    than an absent length: accepting it would mean falling back to counting,
    which is what a caller sending ``Content-Length: 99999999999999999999``
    would like us to do.
    """
    raw = headers.get(b"content-length")
    if raw is None:
        return None
    try:
        value = int(raw.decode("ascii").strip())
    except (UnicodeDecodeError, ValueError):
        raise ValueError("content_length_not_a_number") from None
    if value < 0:
        raise ValueError("content_length_negative")
    return value


class _TooLarge(Exception):
    """Raised inside the receive wrapper to stop pulling the body."""


class RequestSizeLimitMiddleware:
    """Refuse an oversized request before, or while, its body arrives.

    Pure ASGI rather than ``BaseHTTPMiddleware``: the body has to be bounded
    where the framework reads it, which is below anything that has already
    parsed a form.
    """

    def __init__(self, app, max_bytes: int = MAX_REQUEST_BYTES):
        self.app = app
        self.max_bytes = max_bytes
        self._stream_uploads = threading.BoundedSemaphore(2)

    async def __call__(self, scope, receive, send) -> None:
        streamed = (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/api/v1/ingest/jobs"
        )
        if streamed and not self._stream_uploads.acquire(blocking=False):
            await _refuse(
                scope,
                send,
                "upload_capacity_busy",
                "Uploads are busy. Try again shortly.",
                429,
            )
            return
        try:
            await self._limited(scope, receive, send)
        finally:
            if streamed:
                self._stream_uploads.release()

    async def _limited(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.app(scope, receive, send)
            return

        max_bytes = self.max_bytes
        if scope.get("path") == "/api/v1/ingest/jobs":
            from services.stream_routes import capacity_policy
            from starlette.concurrency import run_in_threadpool

            policy = await run_in_threadpool(capacity_policy)
            max_bytes = (
                max(
                    policy.capacity(name).effective_max_bytes
                    for name in ("csv", "text", "workbook")
                )
                + MULTIPART_OVERHEAD_BYTES
            )
            if self.max_bytes != MAX_REQUEST_BYTES:
                max_bytes = min(max_bytes, self.max_bytes)
        headers = dict(scope.get("headers") or [])
        try:
            declared = declared_length(headers)
        except ValueError:
            await _refuse(
                scope, send, MALFORMED_LENGTH_CODE, MALFORMED_LENGTH_MESSAGE, 400
            )
            return
        if declared is not None and declared > max_bytes:
            # Nothing of the body has been read, and nothing will be.
            await _refuse(scope, send, TOO_LARGE_CODE, TOO_LARGE_MESSAGE, 413)
            return

        counted = _CountingReceive(receive, max_bytes)
        try:
            await self.app(scope, counted, send)
        except _TooLarge:
            # The declared length was absent or untrue. Whatever the endpoint
            # was doing is abandoned; it had produced no artifact, because the
            # body it was reading never finished arriving.
            if not counted.response_started:
                await _refuse(scope, send, TOO_LARGE_CODE, TOO_LARGE_MESSAGE, 413)


class _CountingReceive:
    """Passes body chunks through, and stops when too many have gone by."""

    def __init__(self, receive: Callable[[], Awaitable[dict]], max_bytes: int):
        self._receive = receive
        self._max_bytes = max_bytes
        self.total = 0
        self.response_started = False

    async def __call__(self) -> dict:
        message = await self._receive()
        if message.get("type") != "http.request":
            return message
        self.total += len(message.get("body") or b"")
        if self.total > self._max_bytes:
            raise _TooLarge()
        return message


async def _refuse(scope, send, code: str, message: str, status: int) -> None:
    body = ('{"error_code":"%s","message":"%s"}' % (code, message)).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                # The connection is closed rather than kept alive: the rest of a
                # body nobody is going to read would otherwise still arrive.
                (b"connection", b"close"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})
