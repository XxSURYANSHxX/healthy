"""Source envelope encryption needed by private stream spill/review files only."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from typing import Any, Dict

PUBLICATION_VERSION = "publication-v1"

MASTER_KEY_ENV_PREFIX = "BIOBLOCK_PUBLICATION_MASTER_KEY_V"

ACTIVE_KEY_VERSION_ENV_VAR = "BIOBLOCK_PUBLICATION_KEY_VERSION"

KEY_BYTES = 32

NONCE_BYTES = 12

MAGIC = b"BBPUB1\0"


class PublicationError(RuntimeError):
    """A refused publication. Carries a code, never a value or a path."""

    def __init__(self, error_code: str, status_code: int = 409):
        super().__init__(error_code)
        self.error_code = error_code
        self.status_code = status_code


def active_key_version() -> str:
    return (os.getenv(ACTIVE_KEY_VERSION_ENV_VAR) or "1").strip() or "1"


def master_key(version: str) -> bytes:
    """The wrapping key for one version, or a refusal.

    A missing key is a refusal rather than a generated default: a default
    would encrypt an artifact under a key that disappears at restart, and the
    failure would only be discovered when someone tried to read it back.
    """
    raw = (os.getenv(f"{MASTER_KEY_ENV_PREFIX}{version}") or "").strip()
    if not raw:
        raise PublicationError("publication_key_not_configured", status_code=503)
    try:
        key = bytes.fromhex(raw)
    except ValueError:
        raise PublicationError("publication_key_malformed", status_code=503) from None
    if len(key) != KEY_BYTES:
        raise PublicationError("publication_key_malformed", status_code=503)
    return key


def _aad(
    owner: str,
    artifact_sha256: str,
    media_type: str,
    context: str = PUBLICATION_VERSION,
) -> bytes:
    """What the ciphertext is bound to.

    Authenticated as associated data, so a ciphertext cannot be moved to a
    different artifact, a different owner or a different media type and still
    decrypt: the tag covers all three. ``context`` names what the envelope is
    for, so a quarantined review copy and a publication can never be swapped
    for one another even under the same master key.
    """
    return json.dumps(
        {
            "v": context,
            "owner": owner.lower(),
            "artifact": artifact_sha256,
            "media_type": media_type,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def encrypt(
    plaintext: bytes,
    owner: str,
    artifact_sha256: str,
    media_type: str,
    context: str = PUBLICATION_VERSION,
) -> Dict[str, Any]:
    """Envelope-encrypt one artifact. Returns the payload and the wrapped key."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    version = active_key_version()
    wrapping = master_key(version)
    data_key = secrets.token_bytes(KEY_BYTES)
    nonce = secrets.token_bytes(NONCE_BYTES)
    aad = _aad(owner, artifact_sha256, media_type, context)
    body = AESGCM(data_key).encrypt(nonce, plaintext, aad)

    wrap_nonce = secrets.token_bytes(NONCE_BYTES)
    wrapped = AESGCM(wrapping).encrypt(wrap_nonce, data_key, aad)
    return {
        "payload": MAGIC + nonce + body,
        "key_version": version,
        "wrapped_key": (wrap_nonce + wrapped).hex(),
    }


def decrypt(
    payload: bytes,
    wrapped_key_hex: str,
    key_version: str,
    owner: str,
    artifact_sha256: str,
    media_type: str,
    context: str = PUBLICATION_VERSION,
) -> bytes:
    """The plaintext, or a refusal. A failed tag is never a partial answer."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if not payload.startswith(MAGIC) or len(payload) < len(MAGIC) + NONCE_BYTES:
        raise PublicationError("published_payload_malformed")
    wrapping = master_key(key_version)
    aad = _aad(owner, artifact_sha256, media_type, context)
    try:
        wrapped = bytes.fromhex(wrapped_key_hex)
    except ValueError:
        raise PublicationError("published_key_malformed") from None
    if len(wrapped) <= NONCE_BYTES:
        raise PublicationError("published_key_malformed")
    try:
        data_key = AESGCM(wrapping).decrypt(
            wrapped[:NONCE_BYTES], wrapped[NONCE_BYTES:], aad
        )
        nonce = payload[len(MAGIC) : len(MAGIC) + NONCE_BYTES]
        return AESGCM(data_key).decrypt(nonce, payload[len(MAGIC) + NONCE_BYTES :], aad)
    except InvalidTag:
        # Tampered bytes, the wrong key version, or a payload moved to another
        # artifact. All three are the same answer: nothing is returned.
        raise PublicationError("published_payload_not_authentic") from None
