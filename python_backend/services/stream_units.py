"""The work-unit contract shared by the streaming CSV and text engines.

A streaming job reads its input in bounded units, in a fixed order, and each
unit is accounted for: what it read (a byte and record range, and its share of
the input digest), what it wrote, whether it verified, and how it ended. A
file is releasable only when every unit of every pass completed and verified;
one missing, failed, cancelled or timed-out unit refuses release.

Units are bounded on every axis at once - bytes, cells or text segments,
estimated model tokens, and distinct model-bound values - so a single huge
record or field cannot slip through a byte limit alone. Cancellation and the
job deadline are checked before a unit is parsed, before anything is sent to
a model and before its output is committed. Unit errors carry codes and
positions, never content.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from services import request_deadline

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"

VERIFY_PENDING = "pending"
VERIFY_PASSED = "passed"
VERIFY_HELD = "held"

#: Rough characters per model token, for the token budget of a unit.
CHARS_PER_TOKEN = 4

_LIMITS = {
    # name: (environment variable, default, low, high)
    "max_bytes": (
        "BIOBLOCK_STREAM_UNIT_BYTES",
        4 * 1024 * 1024,
        64 * 1024,
        64 * 1024 * 1024,
    ),
    "max_cells": ("BIOBLOCK_STREAM_UNIT_CELLS", 100_000, 1_000, 2_000_000),
    "max_tokens": ("BIOBLOCK_STREAM_UNIT_TOKENS", 1_500_000, 10_000, 20_000_000),
    "max_model_values": ("BIOBLOCK_STREAM_UNIT_MODEL_VALUES", 2_048, 16, 100_000),
    "max_record_chars": (
        "BIOBLOCK_STREAM_MAX_RECORD_CHARS",
        1024 * 1024,
        32 * 1024,
        16 * 1024 * 1024,
    ),
    # A window plus both overlaps must stay within full_v1's 256 KiB per call
    # even at four bytes a character: at most 48 Ki + 2 x 8 Ki characters.
    "text_window_chars": (
        "BIOBLOCK_STREAM_TEXT_WINDOW_CHARS",
        32 * 1024,
        4 * 1024,
        48 * 1024,
    ),
    "text_overlap_chars": ("BIOBLOCK_STREAM_TEXT_OVERLAP_CHARS", 1024, 256, 8 * 1024),
}


class UnitError(RuntimeError):
    """A unit that cannot complete. A code and a position, never content."""

    def __init__(self, error_code: str, unit_index: int = -1, status_code: int = 422):
        super().__init__(error_code)
        self.error_code = error_code
        self.unit_index = unit_index
        self.status_code = status_code


@dataclass(frozen=True)
class UnitBudget:
    """How large one unit may be, on every axis. Conservative laptop defaults."""

    max_bytes: int = _LIMITS["max_bytes"][1]
    max_cells: int = _LIMITS["max_cells"][1]
    max_tokens: int = _LIMITS["max_tokens"][1]
    max_model_values: int = _LIMITS["max_model_values"][1]
    max_record_chars: int = _LIMITS["max_record_chars"][1]
    text_window_chars: int = _LIMITS["text_window_chars"][1]
    text_overlap_chars: int = _LIMITS["text_overlap_chars"][1]

    @classmethod
    def from_environment(cls) -> "UnitBudget":
        from services.resource_policy import CapacityConfigurationError

        values: Dict[str, int] = {}
        for name, (variable, default, low, high) in _LIMITS.items():
            raw = (os.getenv(variable) or "").strip()
            if not raw:
                values[name] = default
                continue
            if not raw.isdigit() or not low <= int(raw) <= high:
                raise CapacityConfigurationError(f"invalid_{variable.lower()}")
            values[name] = int(raw)
        budget = cls(**values)
        if budget.text_overlap_chars * 4 > budget.text_window_chars:
            raise CapacityConfigurationError(
                "invalid_bioblock_stream_text_overlap_chars"
            )
        return budget

    def full(self, units_bytes: int, cells: int, chars: int, model_values: int) -> bool:
        """Whether a unit holding this much must close before taking more."""
        return (
            units_bytes >= self.max_bytes
            or cells >= self.max_cells
            or chars // CHARS_PER_TOKEN >= self.max_tokens
            or model_values >= self.max_model_values
        )


@dataclass
class WorkUnit:
    job_id: str
    format: str
    phase: str
    index: int
    record_start: int
    record_end: int = 0
    byte_start: int = 0
    byte_end: int = 0
    input_sha256: str = ""
    cells: int = 0
    model_values: int = 0
    model_call_budget: int = 0
    output_bytes: int = 0
    state: str = PENDING
    verification: str = VERIFY_PENDING
    error_code: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def public(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "index": self.index,
            "records": [self.record_start, self.record_end],
            "bytes": [self.byte_start, self.byte_end],
            "state": self.state,
            "verification": self.verification,
            "error_code": self.error_code or None,
        }


class UnitLedger:
    """Every unit of every pass, in order. The only authority on completeness."""

    def __init__(
        self,
        job_id: str,
        file_format: str,
        progress: Optional[Callable[[str, float], None]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.job_id = job_id
        self.format = file_format
        self.units: List[WorkUnit] = []
        self._progress = progress
        #: Phase -> share of the job's work; progress is the weighted sum of
        #: bytes actually processed in each phase.
        self.weights = weights or {}
        self._done: Dict[str, float] = {}
        self.total_bytes = 0
        self._reported = 0.0
        self.completed_units = 0
        self.failed = False

    def checkpoint(self, stage: str) -> None:
        """Cancellation and the deadline, before parsing, model calls and commits."""
        request_deadline.check(stage)

    def open(self, phase: str, record_start: int, byte_start: int) -> WorkUnit:
        self.checkpoint(f"stream.{self.format}.{phase}.unit")
        index = len([unit for unit in self.units if unit.phase == phase])
        unit = WorkUnit(
            self.job_id,
            self.format,
            phase,
            index,
            record_start,
            byte_start=byte_start,
            state=RUNNING,
        )
        self.units.append(unit)
        return unit

    def close(
        self,
        unit: WorkUnit,
        record_end: int,
        byte_end: int,
        digest_part: bytes = b"",
        output_bytes: int = 0,
        verified: Optional[bool] = None,
    ) -> None:
        self.checkpoint(f"stream.{self.format}.{unit.phase}.commit")
        unit.record_end = record_end
        unit.byte_end = byte_end
        unit.output_bytes = output_bytes
        unit.input_sha256 = (
            hashlib.sha256(digest_part).hexdigest() if digest_part else ""
        )
        if verified is not None:
            unit.verification = VERIFY_PASSED if verified else VERIFY_HELD
        unit.state = COMPLETED
        self.completed_units += 1
        self.advance(unit.phase, byte_end)

    def fail(
        self, unit: Optional[WorkUnit], code: str, cancelled: bool = False
    ) -> None:
        self.failed = True
        if unit is not None:
            unit.state = CANCELLED if cancelled else FAILED
            unit.error_code = code

    def advance(self, phase: str, byte_position: int) -> None:
        self._done[phase] = max(self._done.get(phase, 0.0), float(byte_position))
        if self._progress is None or not self.total_bytes:
            return
        share = 0.0
        for name, weight in self.weights.items():
            share += weight * min(1.0, self._done.get(name, 0.0) / self.total_bytes)
        share = min(0.99, share)
        # Monotonic: a later phase never reports less than was already shown.
        if share > self._reported:
            self._reported = share
            self._progress(phase, share)

    def complete(self, phases: List[str]) -> bool:
        """Every unit of every named phase completed, and none failed."""
        if self.failed:
            return False
        for phase in phases:
            units = [unit for unit in self.units if unit.phase == phase]
            if not units or any(unit.state != COMPLETED for unit in units):
                return False
            # Contiguous: each unit starts where the previous one ended.
            for previous, current in zip(units, units[1:]):
                if current.record_start != previous.record_end or (
                    current.byte_start != previous.byte_end
                ):
                    return False
        return True

    def summary(self) -> Dict[str, Any]:
        phases: Dict[str, Dict[str, int]] = {}
        for unit in self.units:
            entry = phases.setdefault(
                unit.phase, {"units": 0, "records": 0, "bytes": 0, "output_bytes": 0}
            )
            entry["units"] += 1
            entry["records"] = max(entry["records"], unit.record_end)
            entry["bytes"] = max(entry["bytes"], unit.byte_end)
            entry["output_bytes"] += unit.output_bytes
        return {"units_completed": self.completed_units, "phases": phases}
