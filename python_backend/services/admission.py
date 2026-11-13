"""Scoped source projection for CSV/TXT/XLSX streaming and verification."""

from __future__ import annotations

import zipfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple

_KB = 1024

_MB = 1024 * 1024

MAX_ENCODED_BYTES: Dict[str, int] = {
    "text": 256 * _KB,
    "csv": 32 * _MB,
    "pdf": 32 * _MB,
    "image": 32 * _MB,
    "workbook": 32 * _MB,
    "dicom": 128 * _MB,
    "nifti": 256 * _MB,
    "wsi": 256 * _MB,
}

MAX_ZIP_MEMBERS = 5_000

MAX_ZIP_UNCOMPRESSED_BYTES = 256 * _MB

MAX_ZIP_RATIO = 200

MAX_WORKBOOK_SHEETS = 100

MAX_WORKBOOK_CELLS = 200_000

UPLOAD_TOO_LARGE = "upload_too_large"

SHEET_LIMIT_EXCEEDED = "sheet_limit_exceeded"

CELL_LIMIT_EXCEEDED = "cell_limit_exceeded"

CONTAINER_UNPARSEABLE = "container_unparseable"

CONTAINER_MEMBER_LIMIT = "container_member_limit_exceeded"

CONTAINER_DECOMPRESSION_LIMIT = "container_decompression_limit_exceeded"

CONTAINER_ENCRYPTED = "container_encrypted"

CONTAINER_PATH_TRAVERSAL = "container_unsafe_member_path"


class AdmissionRefused(Exception):
    """An upload the service will not open. Carries a code, never content."""

    def __init__(self, reason_code: str, status_code: int = 413):
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.status_code = status_code


def encoded_limit(modality: str) -> Optional[int]:
    return MAX_ENCODED_BYTES.get(modality)


def check_encoded_size(modality: str, size_bytes: int) -> None:
    limit = encoded_limit(modality)
    if limit is not None and size_bytes > limit:
        raise AdmissionRefused(UPLOAD_TOO_LARGE)


def check_workbook_shape(sheets: int, cells: int) -> None:
    if sheets > MAX_WORKBOOK_SHEETS:
        raise AdmissionRefused(SHEET_LIMIT_EXCEEDED)
    if cells > MAX_WORKBOOK_CELLS:
        raise AdmissionRefused(CELL_LIMIT_EXCEEDED)


def unsafe_member_path(name: str) -> bool:
    """Whether a zip member names somewhere outside the archive.

    An OOXML part is a relative path inside the package. A member called
    ``../../etc/passwd`` or ``C:\\Windows\\x`` is naming the host, and the only
    safe response is to refuse the container rather than to sanitise the name
    and hope.
    """
    if not name or name.startswith(("/", "\\")):
        return True
    if ":" in name.split("/", 1)[0]:
        return True
    normalized = name.replace("\\", "/")
    return any(part == ".." for part in normalized.split("/"))


def container_problems(
    content: bytes,
    max_members: int = MAX_ZIP_MEMBERS,
    max_uncompressed_bytes: int = MAX_ZIP_UNCOMPRESSED_BYTES,
    max_ratio: int = MAX_ZIP_RATIO,
) -> List[str]:
    """Reason codes for a zip container that is unsafe to open, else nothing."""
    try:
        with zipfile.ZipFile(BytesIO(content)) as archive:
            members = archive.infolist()
    except (zipfile.BadZipFile, ValueError):
        return [CONTAINER_UNPARSEABLE]
    if len(members) > max_members:
        return [CONTAINER_MEMBER_LIMIT]
    if any(unsafe_member_path(member.filename) for member in members):
        return [CONTAINER_PATH_TRAVERSAL]
    total = 0
    for member in members:
        # A declared size can be nonsense. Refuse rather than trusting it into
        # an allocation.
        if member.file_size < 0 or member.compress_size < 0:
            return [CONTAINER_UNPARSEABLE]
        total += member.file_size
        if total > max_uncompressed_bytes:
            return [CONTAINER_DECOMPRESSION_LIMIT]
    compressed = sum(member.compress_size for member in members) or 1
    if total / compressed > max_ratio:
        return [CONTAINER_DECOMPRESSION_LIMIT]
    if any(member.flag_bits & 0x1 for member in members):
        return [CONTAINER_ENCRYPTED]
    return []
