"""Upload signature contract projected from source auth, without search routes."""

from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
import time
from typing import Dict, Optional

from fastapi import Request


class UploadError(RuntimeError):
    def __init__(self, status_code, error_code, message, headers=None):
        super().__init__(error_code)
        self.status_code, self.error_code, self.message = (
            status_code,
            error_code,
            message,
        )
        self.headers = headers or {}

    def to_public_dict(self):
        return {"error_code": self.error_code, "message": self.message}


def _failed():
    return UploadError(
        401, "authentication_failed", "The upload signature could not be verified."
    )


def _required():
    return UploadError(
        401, "authentication_required", "A complete upload signature is required."
    )


def _consume(address, nonce):
    from services.sanitized_artifacts import output_directory

    path = output_directory().with_name(
        output_directory().name + ".upload-nonces.sqlite3"
    )
    try:
        with sqlite3.connect(path, timeout=10) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS used_nonces (owner TEXT, nonce TEXT, expires REAL, PRIMARY KEY(owner,nonce))"
            )
            connection.execute("BEGIN IMMEDIATE")
            now = time.time()
            connection.execute("DELETE FROM used_nonces WHERE expires < ?", (now,))
            connection.execute(
                "INSERT INTO used_nonces VALUES (?,?,?)",
                (address.lower(), nonce, now + 2 * MAX_CLOCK_SKEW_SECONDS),
            )
    except sqlite3.IntegrityError:
        raise _failed() from None
    except sqlite3.Error:
        raise UploadError(
            503,
            "authentication_unavailable",
            "Upload authentication is temporarily unavailable.",
        ) from None


ADDRESS_HEADER = "X-BioBlock-Address"

TIMESTAMP_HEADER = "X-BioBlock-Timestamp"

NONCE_HEADER = "X-BioBlock-Nonce"

SIGNATURE_HEADER = "X-BioBlock-Signature"

MAX_CLOCK_SKEW_SECONDS = 300

_ADDRESS = re.compile(r"^0x[0-9a-fA-F]{40}$")

_NONCE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_SIGNATURE = re.compile(r"^(?:0x)?[0-9a-fA-F]{130}$")

_TIMESTAMP = re.compile(r"^[0-9]{1,12}$")


def verify_signature(address: str, signature: str, message: str) -> bool:
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct

        recovered = Account.recover_message(
            encode_defunct(text=message), signature=signature
        )
    except Exception:
        return False
    return recovered.lower() == address.lower()


def _signed_headers(request: Request, required: bool):
    """(address, timestamp, nonce, signature) after format checks, or None if unsigned."""
    headers = request.headers
    values = tuple(
        headers.get(name, "")
        for name in (ADDRESS_HEADER, TIMESTAMP_HEADER, NONCE_HEADER, SIGNATURE_HEADER)
    )
    if not any(values):
        if required:
            raise _required()
        return None
    address, timestamp, nonce, signature = values
    if not all(values):
        raise _required() if required else _failed()
    if not (
        _ADDRESS.fullmatch(address)
        and _TIMESTAMP.fullmatch(timestamp)
        and _NONCE.fullmatch(nonce)
        and _SIGNATURE.fullmatch(signature)
    ):
        raise _failed()
    if abs(time.time() - int(timestamp)) > MAX_CLOCK_SKEW_SECONDS:
        raise _failed()
    return values


UPLOAD_MESSAGE_TITLE = "BioBlock upload request"


def canonical_upload_message(
    address: str,
    method: str,
    path: str,
    query: str,
    timestamp: int,
    nonce: str,
    file_sha256: str,
    profile: str,
    study_ref: str = "",
    tabular_policy_sha256: str = "",
) -> str:
    """The exact bytes a wallet signs for one upload.

    The study reference and the tabular policy digest are always lines, empty
    when none was declared, rather than lines that appear only when there is
    something to put in them. A message whose shape depends on its content is a
    message two different uploads can produce, and the signature then covers
    less than it looks like.
    """
    return "\n".join(
        (
            UPLOAD_MESSAGE_TITLE,
            f"address:{address.lower()}",
            f"method:{method.upper()}",
            f"path:{path}",
            f"query:{query}",
            f"timestamp:{timestamp}",
            f"nonce:{nonce}",
            f"file-sha256:{file_sha256}",
            f"profile:{profile}",
            f"study-ref:{study_ref}",
            f"tabular-policy-sha256:{tabular_policy_sha256}",
        )
    )


def verify_upload_owner(
    request: Request,
    file_sha256: str,
    profile: str,
    study_ref: str = "",
    tabular_policy_sha256: str = "",
) -> Optional[str]:
    """The wallet that signed this upload, None for an unsigned one; refuse a bad signature.

    A multipart body is serialised by the browser, so the signature covers the
    SHA-256 of the uploaded file's bytes and the requested profile instead of
    the raw body. It must be checked before the file is processed.
    """
    if request is None:
        # Called directly, outside HTTP: there is no signature to verify.
        return None
    values = _signed_headers(request, required=False)
    if values is None:
        return None
    address, timestamp, nonce, signature = values
    message = canonical_upload_message(
        address,
        request.method,
        request.url.path,
        request.url.query,
        int(timestamp),
        nonce,
        file_sha256,
        profile,
        study_ref,
        tabular_policy_sha256,
    )
    if not verify_signature(address, signature, message):
        raise _failed()
    _consume(address, nonce)
    return address.lower()


def signed_upload_headers(
    private_key: str,
    path: str,
    file_bytes: bytes,
    profile: str,
    query: str = "",
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
    file_sha256: Optional[str] = None,
    study_ref: str = "",
    tabular_policy: str = "",
) -> Dict[str, str]:
    """Client-side reference: the headers for one signed upload.

    ``tabular_policy`` is the exact form-field text sent beside the file; the
    wallet signs its SHA-256, or an empty line when there is none.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct

    account = Account.from_key(private_key)
    stamp = int(time.time()) if timestamp is None else int(timestamp)
    once = nonce or secrets.token_urlsafe(24)
    message = canonical_upload_message(
        account.address,
        "POST",
        path,
        query,
        stamp,
        once,
        file_sha256 or hashlib.sha256(file_bytes).hexdigest(),
        profile,
        study_ref,
        (
            hashlib.sha256(tabular_policy.encode("utf-8")).hexdigest()
            if tabular_policy
            else ""
        ),
    )
    signature = Account.sign_message(
        encode_defunct(text=message), private_key=private_key
    )
    return {
        ADDRESS_HEADER: account.address,
        TIMESTAMP_HEADER: str(stamp),
        NONCE_HEADER: once,
        SIGNATURE_HEADER: signature.signature.hex(),
    }


INGESTION_MESSAGE_TITLE = "BioBlock upload session request"
MAX_BODY_BYTES = 16 * 1024
MESSAGE_TITLE = INGESTION_MESSAGE_TITLE


async def authenticated_ingestion_owner(request: Request) -> str:
    address, timestamp, nonce, signature = _signed_headers(request, required=True)
    body = bytearray()
    async for block in request.stream():
        if len(body) + len(block) > MAX_BODY_BYTES:
            raise UploadError(413, "request_too_large", "The request is too large.")
        body.extend(block)
    message = canonical_message(
        address,
        request.method,
        request.url.path,
        request.url.query,
        int(timestamp),
        nonce,
        hashlib.sha256(body).hexdigest(),
        INGESTION_MESSAGE_TITLE,
    )
    if not verify_signature(address, signature, message):
        raise _failed()
    _consume(address, nonce)
    return address.lower()


def canonical_message(
    address: str,
    method: str,
    path: str,
    query: str,
    timestamp: int,
    nonce: str,
    body_sha256: str,
    title: str = MESSAGE_TITLE,
) -> str:
    return "\n".join(
        (
            title,
            f"address:{address.lower()}",
            f"method:{method.upper()}",
            f"path:{path}",
            f"query:{query}",
            f"timestamp:{timestamp}",
            f"nonce:{nonce}",
            f"body-sha256:{body_sha256}",
        )
    )


def signed_headers(
    private_key: str,
    method: str,
    path: str,
    body: bytes = b"",
    query: str = "",
    timestamp: Optional[int] = None,
    nonce: Optional[str] = None,
    title: str = MESSAGE_TITLE,
) -> Dict[str, str]:
    """Client-side reference: the headers for one signed request.

    Used by tests, the evaluation and the benchmark, and documented for the
    frontend. A browser signs the same message with ``personal_sign``.
    """
    from eth_account import Account
    from eth_account.messages import encode_defunct

    account = Account.from_key(private_key)
    stamp = int(time.time()) if timestamp is None else int(timestamp)
    once = nonce or secrets.token_urlsafe(24)
    message = canonical_message(
        account.address,
        method,
        path,
        query,
        stamp,
        once,
        hashlib.sha256(body).hexdigest(),
        title,
    )
    signature = Account.sign_message(
        encode_defunct(text=message), private_key=private_key
    )
    return {
        ADDRESS_HEADER: account.address,
        TIMESTAMP_HEADER: str(stamp),
        NONCE_HEADER: once,
        SIGNATURE_HEADER: signature.signature.hex(),
    }


def signed_ingestion_headers(
    private_key: str, method: str, path: str, **options
) -> Dict[str, str]:
    """Client-side reference: the headers for one signed upload-session request."""
    return signed_headers(
        private_key, method, path, title=INGESTION_MESSAGE_TITLE, **options
    )


def request_is_signed(request: Request) -> bool:
    """Whether a request carries any wallet signature header at all."""
    return any(
        request.headers.get(name)
        for name in (ADDRESS_HEADER, TIMESTAMP_HEADER, NONCE_HEADER, SIGNATURE_HEADER)
    )
