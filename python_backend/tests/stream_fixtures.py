"""Deterministic large CSV and text fixtures, and stand-in models, for streaming tests.

Nothing here is real: identifiers are generated UUIDs, names are invented,
and the stand-in models find exactly the planted values. The fixtures are
written to disk in bounded chunks; nothing builds a whole file in memory.
"""

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

HEADER = [
    "PATIENT",
    "ENCOUNTER",
    "DATE",
    "CODE",
    "DESCRIPTION",
    "analyte",
    "value",
    "unit",
    "note",
]
TERMS = [
    ("419474003", "Allergy to mould"),
    ("300916003", "Latex allergy"),
    ("424213003", "Allergy to bee venom"),
    ("91935009", "Allergy to peanuts"),
    ("0232", "DTaP"),
    ("08", "Hep B  adult"),
    ("140", "Influenza  seasonal  injectable"),
    ("00123", "Hep A  ped"),
]
ANALYTES = [
    ("haemoglobin", "g/L", "132"),
    ("serum creatinine", "umol/L", "88"),
    ("glucose", "mmol/L", "5.4"),
    ("white blood cell count", "10^9/L", "6.1"),
]
#: Invented people planted near the start, unit boundaries, the middle and the end.
PLANTED_NAMES = [
    "Zelquin Vantreske",
    "Marbrint Morrowgale",
    "Quenby Ashfordale",
    "Talvio Brenwick",
    "Oriel Pemberfast",
    "Ysolde Crandmere",
]


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def write_csv(
    path: Path, target_bytes: int, seed: int = 7, patients: int = 5_000
) -> Dict[str, object]:
    """A Synthea-shaped table of ``target_bytes`` or a little more. Returns its facts."""
    rng = random.Random(seed)
    people = [_uuid(rng) for _ in range(patients)]
    planted_rows: List[int] = []
    rows = 0
    size = 0
    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8", newline="") as stream:

        def put(line: str) -> None:
            nonlocal size
            data = line.encode("utf-8")
            size += len(data)
            digest.update(data)
            stream.write(line)

        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        put(",".join(HEADER) + "\n")
        estimate = max(1, target_bytes // 170)
        marks = sorted(
            {
                1,
                2,
                estimate // 3,
                estimate // 2,
                estimate - 3,
                *(estimate * k // 7 for k in range(1, 7)),
            }
        )
        while size < target_bytes:
            rows += 1
            code, description = TERMS[rows % len(TERMS)]
            analyte, unit, value = ANALYTES[rows % len(ANALYTES)]
            note = "Stable, no change"
            if rows in marks:
                note = (
                    f"Seen with {PLANTED_NAMES[len(planted_rows) % len(PLANTED_NAMES)]}"
                )
                planted_rows.append(rows)
            date = f"20{10 + rows % 10:02d}-{1 + rows % 12:02d}-{1 + rows % 28:02d}"
            buffer.seek(0)
            buffer.truncate(0)
            writer.writerow(
                [
                    people[rows % patients],
                    _uuid(rng),
                    date,
                    code,
                    description,
                    analyte,
                    value,
                    unit,
                    note,
                ]
            )
            put(buffer.getvalue())
    return {
        "rows": rows,
        "bytes": size,
        "sha256": digest.hexdigest(),
        "planted_rows": planted_rows,
        "patients": patients,
    }


NARRATIVE = (
    "Assessment: hypertension controlled on amlodipine 5 mg daily; creatinine 88 "
    "umol/L. Plan: continue current dose and repeat renal function in three months. "
)
PLANTED_TEXT = [
    "Dr Zelquin Vantreske",
    "zv.contact@example.invalid",
    "(555) 014-2277",
    "14 Harrow Lane",
    "2019-03-14",
    "MRN-4821337",
    "Ysolde Crandmère",
]


def write_text(
    path: Path, target_bytes: int, window: int = 32 * 1024, seed: int = 11
) -> Dict[str, object]:
    """Repeated narrative with planted values straddling window boundaries."""
    rng = random.Random(seed)
    size = 0
    planted = 0
    position = 0
    digest = hashlib.sha256()
    with open(path, "w", encoding="utf-8", newline="") as stream:

        def put(text: str) -> None:
            nonlocal size, position
            data = text.encode("utf-8")
            size += len(data)
            position += len(text)
            digest.update(data)
            stream.write(text)

        boundary = window
        while size < target_bytes:
            if position + len(NARRATIVE) + 40 >= boundary:
                value = PLANTED_TEXT[planted % len(PLANTED_TEXT)]
                # The planted value straddles the boundary by a varying amount.
                lead = max(
                    0, boundary - position - rng.randint(1, max(1, len(value) - 1))
                )
                put(NARRATIVE[:lead] if lead < len(NARRATIVE) else NARRATIVE)
                put(" " + value + (" \r\n" if planted % 3 == 0 else " "))
                planted += 1
                boundary += window
                continue
            put(NARRATIVE)
    return {"bytes": size, "sha256": digest.hexdigest(), "planted": planted}


# -- stand-in models ------------------------------------------------------------------------------

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\(\d{3}\) \d{3}-\d{4}")
_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_MRN = re.compile(r"\bMRN-\d{7}\b")
_ADDRESS = re.compile(r"\b\d{1,5} [A-Z][a-z]+ (?:Lane|Street|Road)\b")
_NAMES = re.compile(
    "|".join(
        re.escape(name)
        for name in sorted(
            {*PLANTED_NAMES, "Zelquin Vantreske", "Ysolde Crandmère"},
            key=len,
            reverse=True,
        )
    )
)
_TITLE = re.compile(r"\bDr (?=" + _NAMES.pattern + ")")


def _spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    for pattern, kind in (
        (_EMAIL, "EMAIL_ADDRESS"),
        (_PHONE, "PHONE_NUMBER"),
        (_DATE, "DATE"),
        (_MRN, "MEDICAL_RECORD_NUMBER"),
        (_ADDRESS, "ADDRESS"),
        (_NAMES, "PERSON"),
    ):
        for match in pattern.finditer(text):
            start = match.start()
            if kind == "PERSON":
                title = text.rfind("Dr ", 0, start)
                if title >= 0 and title + 3 == start:
                    start = title
            if not any(start < end and begin < match.end() for begin, end, _k in spans):
                spans.append((start, match.end(), kind))
    return sorted(spans)


def fake_window_anonymizer(text: str, profile: str, study_salt, allocator):
    """Finds exactly the planted shapes, replaces them with shared surrogates."""
    applied = []
    pieces = []
    cursor = 0
    for start, end, kind in _spans(text):
        value = text[start:end]
        token = allocator.surrogate_for(kind, value) or f"<REDACTED_{kind}>"
        applied.append((start, end, kind, token))
        pieces.append(text[cursor:start])
        pieces.append(token)
        cursor = end
    pieces.append(text[cursor:])
    result = {
        "anonymization_status": "completed",
        "anonymized_text": "".join(pieces),
        "residual_phi_categories": {},
        "review_required_reasons": [],
        "utility_metrics": {},
        "date_strategy": "redact",
        "text_identifier_strategy": "redact",
        "detected_entities": {},
        "entity_count": len(applied),
        "detection_sources": {},
        "ner_model": "stand-in",
        "trained_ner_active": False,
        "surrogate_counts": {},
    }
    return result, applied


def fake_cell_anonymizer(text: str, profile: str, allocator):
    """The CSV cell anonymizer stand-in: the same planted shapes, per joined batch."""
    result, applied = fake_window_anonymizer(text, profile, None, allocator)
    return result, applied


def fake_residual(text: str, provenance=None) -> List[Dict[str, object]]:
    """A residual detector that finds any planted value still present."""
    found = []
    for _start, _end, kind in _spans(text):
        found.append(
            {
                "blocking": True,
                "category": kind,
                "detector": "stand_in",
                "evidence_type": "model",
            }
        )
    return found


def fake_categories(text: str) -> Iterable[str]:
    return [finding["category"] for finding in fake_residual(text)]


def stream_options() -> Dict[str, object]:
    return {
        "text_anonymizer": fake_cell_anonymizer,
        "verify_scanner": fake_residual,
        "audit_text_scanner": fake_categories,
        "audit_finding_scanner": fake_residual,
        "text_window_anonymizer": fake_window_anonymizer,
        "text_residual": fake_residual,
    }


def planted_values() -> Sequence[str]:
    return [*PLANTED_NAMES, *PLANTED_TEXT]
