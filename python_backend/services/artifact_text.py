"""Scoped source projection for CSV/TXT/XLSX streaming and verification."""

from __future__ import annotations

import re

from services.surrogates import HASH_SURROGATE_PATTERN

_PLACEHOLDER = re.compile(
    r"<[A-Z][A-Z_]*_[1-9][0-9]*>|<REDACTED_[A-Z_]+>|\[REDACTED(?:_[A-Z_]+)?\]"
    r"|\b[A-Z][A-Z]*(?:_[A-Z]+)*_\d{3,}\b|\b90\+"
)


def strip_placeholders(text: str) -> str:
    return HASH_SURROGATE_PATTERN.sub(" ", _PLACEHOLDER.sub(" ", text))


PLACEHOLDER_FILLER = "█"


def fill_placeholders(text: str) -> str:
    """Each generated placeholder becomes one filler token, not a vanished gap.

    Blanking a placeholder deletes a token, which pulls a label ("MRN") that
    the placeholder already satisfied closer to an unrelated later number and
    lets a token-distance scan announce it. The filler keeps the distance.
    """
    # No added spaces: the punctuation that followed the placeholder ("<X>;")
    # must stay attached, because it is what ends the labelled field.
    return HASH_SURROGATE_PATTERN.sub(
        PLACEHOLDER_FILLER, _PLACEHOLDER.sub(PLACEHOLDER_FILLER, text)
    )
