"""Reading an upload without holding more of it than necessary.

The route used to decide whether a file was too large by reading ``limit + 1``
bytes of it: to refuse a 256 MB upload it first allocated 256 MB. It then
hashed that buffer separately, so the bytes were walked twice and, for a
moment, held twice.

Reading here is chunked. The running total is checked against the format's
limit as it goes, so an oversized upload stops at the first chunk past the
limit and the buffer is discarded without ever reaching full size. The digest
is computed during the same pass, so nothing is walked a second time and the
signature check has a digest without a second copy to hash.

What is kept at the end is one copy of the bytes, because that is what the
format handlers take. Bounding *that* would mean giving every handler a file
path instead, which is a larger change than this one and is recorded as a
limitation rather than pretended away.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from typing import Optional, Protocol

#: How much is read at a time. Large enough that a 256 MB file is a few
#: hundred reads, small enough that overshooting a limit costs this much and
#: not the whole remainder.
CHUNK_BYTES = 1024 * 1024

#: Above this, a spooled read moves to a temporary file rather than staying in
#: memory. Matches the framework's own multipart threshold.
SPOOL_THRESHOLD_BYTES = 1024 * 1024


class TooLarge(Exception):
    """The upload passed its limit. Carries the limit, never the size read."""

    def __init__(self, limit: int):
        super().__init__("upload_too_large")
        self.limit = limit


class AsyncReader(Protocol):
    async def read(self, size: int = -1) -> bytes: ...

    async def seek(self, offset: int) -> None: ...


@dataclass(frozen=True)
class SpooledUpload:
    """One upload that fitted its limit, with the digest of what was read."""

    content: bytes
    sha256: str
    size_bytes: int


async def read_bounded(
    reader: AsyncReader, limit: int, chunk_bytes: int = CHUNK_BYTES
) -> SpooledUpload:
    """Read at most ``limit`` bytes, hashing as it goes.

    Raises :class:`TooLarge` at the first chunk that takes the total past the
    limit, before the remainder is read and before the buffer has grown to the
    size that was refused.
    """
    await reader.seek(0)
    digest = hashlib.sha256()
    parts = []
    total = 0
    while True:
        chunk = await reader.read(chunk_bytes)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            # Drop what was collected now rather than at the end of the
            # function: the caller is about to raise, and the partial buffer
            # is the largest thing in the request.
            parts.clear()
            raise TooLarge(limit)
        digest.update(chunk)
        parts.append(chunk)
    return SpooledUpload(b"".join(parts), digest.hexdigest(), total)


async def spool_to_file(
    reader: AsyncReader, limit: int, chunk_bytes: int = CHUNK_BYTES
) -> "tempfile.SpooledTemporaryFile":
    """The same bounded read, into a spool that moves to disk when it grows.

    For callers that need the bytes on a file rather than in one object. The
    caller closes it, which deletes the temporary file if one was created.
    """
    await reader.seek(0)
    spool = tempfile.SpooledTemporaryFile(max_size=SPOOL_THRESHOLD_BYTES)
    total = 0
    try:
        while True:
            chunk = await reader.read(chunk_bytes)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise TooLarge(limit)
            spool.write(chunk)
    except BaseException:
        # Every exit that is not a completed read closes the spool, so a
        # refusal, a cancellation and an error all leave nothing behind.
        spool.close()
        raise
    spool.seek(0)
    return spool


def digest_of(content: Optional[bytes]) -> Optional[str]:
    return hashlib.sha256(content).hexdigest() if content is not None else None
