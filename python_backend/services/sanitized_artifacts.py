"""Scoped source projection for CSV/TXT/XLSX streaming and verification."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

OUTPUT_DIR_ENV_VAR = "BIOBLOCK_SANITIZED_OUTPUT_DIR"


def output_directory() -> Path:
    configured = os.getenv(OUTPUT_DIR_ENV_VAR)
    directory = (
        Path(configured)
        if configured and configured.strip()
        else (Path(tempfile.gettempdir()) / "bioblock-sanitized-artifacts")
    )
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def file_sha256(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            from services.request_deadline import check

            check("stream.digest")
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size
