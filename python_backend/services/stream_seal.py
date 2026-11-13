"""Sealing files too large to hold in memory, with the existing envelope scheme.

``publication.encrypt`` seals bytes in one AES-GCM call, which needs the whole
plaintext in memory. For a streamed review copy the same construction is used
incrementally instead: a random 256-bit data key, standard AES-GCM with a
random 96-bit nonce over the file in chunks, and the same associated data
(owner, digest, media type, context) as the one-shot envelope. The data key is
wrapped exactly as any artifact key is - by ``publication.encrypt`` under the
master key and the same context - so nothing here introduces a new primitive
or a new key hierarchy.

Opening writes the plaintext to a private temporary file and reports it only
after the GCM tag verified and the plaintext re-hashed to the recorded digest;
a failed tag or digest leaves nothing behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import tempfile
from pathlib import Path
from typing import Tuple

from services import stream_crypto as publication

MAGIC = b"BBSG1"
PREFIX = "stream-gcm-v1:"
NONCE_BYTES = 12
TAG_BYTES = 16
CHUNK = 1024 * 1024


class SealError(RuntimeError):
    def __init__(self, error_code: str):
        super().__init__(error_code)
        self.error_code = error_code


def is_streamed(wrapped_key: str) -> bool:
    return isinstance(wrapped_key, str) and wrapped_key.startswith(PREFIX)


def seal_file(
    source: Path, target: Path, owner: str, sha256: str, media_type: str, context: str
) -> Tuple[str, str, int]:
    """(key version, wrapped key, plaintext size). Writes ``target`` atomically."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    data_key = secrets.token_bytes(32)
    wrapped = publication.encrypt(data_key, owner, sha256, media_type, context=context)
    nonce = secrets.token_bytes(NONCE_BYTES)
    encryptor = Cipher(algorithms.AES(data_key), modes.GCM(nonce)).encryptor()
    encryptor.authenticate_additional_data(
        publication._aad(owner, sha256, media_type, context)
    )
    digest = hashlib.sha256()
    size = 0
    handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".sealing-")
    try:
        with os.fdopen(handle, "wb") as out, open(source, "rb") as stream:
            out.write(MAGIC + nonce)
            for block in iter(lambda: stream.read(CHUNK), b""):
                from services.request_deadline import check

                check("stream.seal")
                digest.update(block)
                size += len(block)
                out.write(encryptor.update(block))
            out.write(encryptor.finalize())
            out.write(encryptor.tag)
            out.flush()
            os.fsync(out.fileno())
        if digest.hexdigest() != sha256:
            raise SealError("stream_seal_digest_mismatch")
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    envelope = json.dumps(
        {"payload": wrapped["payload"].hex(), "wrapped_key": wrapped["wrapped_key"]},
        separators=(",", ":"),
    )
    return wrapped["key_version"], PREFIX + envelope, size


def open_file(
    source: Path,
    directory: Path,
    key_version: str,
    wrapped_key: str,
    owner: str,
    sha256: str,
    media_type: str,
    context: str,
    size: int,
) -> Path:
    """A private plaintext copy, verified by tag and digest; the caller deletes it."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    if not is_streamed(wrapped_key):
        raise SealError("stream_seal_not_streamed")
    try:
        envelope = json.loads(wrapped_key[len(PREFIX) :])
        data_key = publication.decrypt(
            bytes.fromhex(envelope["payload"]),
            envelope["wrapped_key"],
            key_version,
            owner,
            sha256,
            media_type,
            context=context,
        )
    except (ValueError, KeyError, publication.PublicationError):
        raise SealError("stream_seal_key_unreadable") from None
    total = source.stat().st_size
    body = total - len(MAGIC) - NONCE_BYTES - TAG_BYTES
    if body < 0:
        raise SealError("stream_seal_malformed")
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(directory, stat.S_IRWXU)
    except OSError:
        pass
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".open-")
    digest = hashlib.sha256()
    written = 0
    try:
        with open(source, "rb") as stream, os.fdopen(handle, "wb") as out:
            if stream.read(len(MAGIC)) != MAGIC:
                raise SealError("stream_seal_malformed")
            nonce = stream.read(NONCE_BYTES)
            stream.seek(total - TAG_BYTES)
            tag = stream.read(TAG_BYTES)
            stream.seek(len(MAGIC) + NONCE_BYTES)
            decryptor = Cipher(
                algorithms.AES(data_key), modes.GCM(nonce, tag)
            ).decryptor()
            decryptor.authenticate_additional_data(
                publication._aad(owner, sha256, media_type, context)
            )
            remaining = body
            while remaining:
                from services.request_deadline import check

                check("stream.open")
                block = stream.read(min(CHUNK, remaining))
                if not block:
                    raise SealError("stream_seal_malformed")
                remaining -= len(block)
                plain = decryptor.update(block)
                digest.update(plain)
                written += len(plain)
                out.write(plain)
            try:
                decryptor.finalize()
            except InvalidTag:
                raise SealError("stream_seal_not_authentic") from None
        if digest.hexdigest() != sha256 or written != size:
            raise SealError("stream_seal_digest_mismatch")
        return Path(temporary)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
