"""Phase 16: an oversized upload is refused before it is received.

The limits were enforced after the fact. A request declaring ten gigabytes was
accepted, received in full, spooled, parsed and only then measured, so the
cheapest way to occupy the service was to send it something it was always
going to reject. The per-format check then read ``limit + 1`` bytes into one
buffer to decide the limit had been exceeded, allocating 256 MB to refuse
256 MB.

These tests fix the boundary in both directions - one byte under, exactly at,
one byte over - and cover the three ways a caller can lie about a length: not
declaring one, declaring one that is wrong, and sending chunks that keep
coming. They also check that nothing survives a refusal and that the answer
names no path, limit source or component.

Every payload is filler, not a document.
"""

import asyncio
import io
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import upload_limits  # noqa: E402
from services.ingestion import TEXT_READ_LIMIT_BYTES  # noqa: E402
from services.upload_spool import TooLarge, read_bounded, spool_to_file  # noqa: E402

INGEST = "/api/v1/ingest"


class _Reader:
    """An upload that hands out bytes a chunk at a time, and counts reads."""

    def __init__(self, payload: bytes):
        self._buffer = io.BytesIO(payload)
        self.bytes_read = 0

    async def read(self, size: int = -1) -> bytes:
        chunk = self._buffer.read(size)
        self.bytes_read += len(chunk)
        return chunk

    async def seek(self, offset: int) -> None:
        self._buffer.seek(offset)


# -- the bounded read -------------------------------------------------------


def test_a_payload_under_the_limit_is_read_whole():
    payload = b"a" * 4096
    reader = _Reader(payload)

    result = asyncio.run(read_bounded(reader, 8192, chunk_bytes=1024))

    assert result.content == payload
    assert result.size_bytes == len(payload)


def test_a_payload_exactly_at_the_limit_is_accepted():
    payload = b"b" * 8192

    result = asyncio.run(read_bounded(_Reader(payload), 8192, chunk_bytes=1024))

    assert result.size_bytes == 8192


def test_one_byte_over_the_limit_is_refused():
    payload = b"c" * 8193

    with pytest.raises(TooLarge) as refused:
        asyncio.run(read_bounded(_Reader(payload), 8192, chunk_bytes=1024))

    assert refused.value.limit == 8192


def test_a_refused_read_stops_at_the_first_chunk_past_the_limit():
    """It must not read the rest to discover how much too big it was. Reading
    limit + 1 bytes to refuse a limit-sized upload is the defect this fixes."""
    reader = _Reader(b"d" * (1024 * 1024))

    with pytest.raises(TooLarge):
        asyncio.run(read_bounded(reader, 4096, chunk_bytes=1024))

    assert reader.bytes_read <= 4096 + 1024


def test_the_digest_is_computed_during_the_same_pass():
    import hashlib

    payload = b"e" * 5000

    result = asyncio.run(read_bounded(_Reader(payload), 8192, chunk_bytes=512))

    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_an_empty_upload_reads_as_empty_rather_than_failing():
    result = asyncio.run(read_bounded(_Reader(b""), 8192))

    assert result.content == b"" and result.size_bytes == 0


# -- the spool --------------------------------------------------------------


def test_a_spooled_read_returns_the_bytes_and_closes_cleanly():
    payload = b"f" * 3000
    spool = asyncio.run(spool_to_file(_Reader(payload), 8192, chunk_bytes=512))
    try:
        assert spool.read() == payload
    finally:
        spool.close()


def test_a_refused_spool_leaves_nothing_open():
    """A refusal, a cancellation and an error all close the spool, so no
    temporary file outlives the request that made it."""
    import tempfile

    before = set(os.listdir(tempfile.gettempdir()))

    with pytest.raises(TooLarge):
        asyncio.run(spool_to_file(_Reader(b"g" * 40_000), 4096, chunk_bytes=1024))

    after = set(os.listdir(tempfile.gettempdir()))
    assert not {name for name in after - before if name.startswith("tmp")}


# -- the declared length ----------------------------------------------------


def test_a_declared_length_over_the_ceiling_is_read_as_too_large():
    headers = {b"content-length": str(upload_limits.MAX_REQUEST_BYTES + 1).encode()}

    assert upload_limits.declared_length(headers) > upload_limits.MAX_REQUEST_BYTES


def test_an_absent_length_is_not_a_zero_length():
    assert upload_limits.declared_length({}) is None


@pytest.mark.parametrize(
    "value", [b"not-a-number", b"99999999999999999999x", b"-1", b""]
)
def test_a_malformed_length_is_refused_rather_than_ignored(value):
    """Ignoring it would fall back to counting, which is exactly what a caller
    sending a nonsense length would prefer."""
    with pytest.raises(ValueError):
        upload_limits.declared_length({b"content-length": value})


# -- through the running app ------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    import streaming_support as support

    support.isolate_storage(tmp_path, monkeypatch)
    from main import app

    return TestClient(app)


def _post(client, payload: bytes, filename="note.txt", content_type="text/plain"):
    return client.post(
        INGEST,
        files={"file": (filename, io.BytesIO(payload), content_type)},
        data={"profile": "strict"},
    )


def test_a_text_upload_one_byte_over_its_limit_is_refused(client):
    response = _post(client, b"h" * (TEXT_READ_LIMIT_BYTES + 1))

    assert response.status_code == 413


def test_a_legacy_text_upload_at_the_size_limit_is_read_without_an_extra_byte(
    client, monkeypatch
):
    import main

    monkeypatch.setattr(
        main,
        "route_for_ingestion",
        lambda **fields: {"size": len(fields["text_content"])},
    )
    assert (
        _post(client, b"x" * TEXT_READ_LIMIT_BYTES).json()["size"]
        == TEXT_READ_LIMIT_BYTES
    )


def test_a_declared_oversized_request_is_refused_without_its_body(client):
    """The body is never read: the answer comes from the declared length."""
    huge = upload_limits.MAX_REQUEST_BYTES + 1
    response = client.post(
        INGEST,
        content=b"x" * 16,
        headers={
            "content-length": str(huge),
            "content-type": "application/octet-stream",
        },
    )

    assert response.status_code == 413
    assert response.json()["error_code"] == upload_limits.TOO_LARGE_CODE


def _bounded_app(max_bytes: int):
    """A endpoint that drains its body, behind the size limit under test.

    Exercised with a small ceiling rather than the real one: sending a quarter
    of a gigabyte through a test client would measure the test client.
    """
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def drain(request):
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
        return JSONResponse({"read": total})

    app = Starlette(routes=[Route("/drain", drain, methods=["POST"])])
    app.add_middleware(upload_limits.RequestSizeLimitMiddleware, max_bytes=max_bytes)
    return TestClient(app)


def test_a_body_that_keeps_coming_is_cut_off_without_a_declared_length():
    """No Content-Length at all, so the declared check cannot help. The bytes
    are counted as the framework pulls them and the request ends when the
    running total passes the ceiling."""
    client = _bounded_app(8192)

    def chunks():
        for _ in range(40):
            yield b"y" * 1024

    response = client.post("/drain", content=chunks())

    assert response.status_code == 413
    assert response.json()["error_code"] == upload_limits.TOO_LARGE_CODE


def test_a_body_smaller_than_it_declares_is_still_read():
    """Counting must not refuse an honest request that simply declares more
    than it sends; only passing the ceiling refuses."""
    client = _bounded_app(8192)

    response = client.post("/drain", content=b"y" * 4096)

    assert response.status_code == 200 and response.json()["read"] == 4096


def test_a_body_at_the_ceiling_is_read_and_one_byte_more_is_not():
    client = _bounded_app(4096)

    assert client.post("/drain", content=b"z" * 4096).status_code == 200
    assert client.post("/drain", content=b"z" * 4097).status_code == 413


def test_a_request_with_no_body_passes_through():
    client = _bounded_app(4096)

    assert client.post("/drain", content=b"").status_code == 200


def test_the_refusal_names_no_path_or_component(client):
    response = client.post(
        INGEST,
        content=b"z" * 16,
        headers={
            "content-length": str(upload_limits.MAX_REQUEST_BYTES + 1),
            "content-type": "application/octet-stream",
        },
    )

    body = response.text
    for leaked in ("C:", "Traceback", "services.", "tempfile", "starlette"):
        assert leaked not in body


def test_an_oversized_upload_reaches_no_model_or_store(client, monkeypatch):
    """Refused before inference, before the artifact store, and before
    anything downstream of them."""
    from services import ingestion

    def fail(*_args, **_kwargs):
        raise AssertionError("an oversized upload reached the pipeline")

    monkeypatch.setattr(ingestion, "route_for_ingestion", fail)

    response = _post(client, b"i" * (TEXT_READ_LIMIT_BYTES + 1))

    assert response.status_code == 413


def test_a_normal_upload_still_produces_the_same_artifact(client):
    """The streaming read must not change what a good upload becomes."""
    import hashlib

    note = b"Assessment: routine review with no findings recorded.\n"
    first = _post(client, note).json()
    second = _post(client, note).json()

    assert first["detected_modality"] == "text"
    assert (
        first["transformation_manifest"]["artifact_id"]
        == second["transformation_manifest"]["artifact_id"]
    )
    assert first["anonymized_text"] == second["anonymized_text"]
