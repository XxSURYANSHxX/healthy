"""Bounded-memory text anonymization: overlapping windows, each character owned once.

A note is decoded incrementally and read in windows. Each window is an owned
region plus a bounded overlap on both sides; full_v1 reads the whole window,
so a name, a title and a name, an e-mail address, a phone number, an address,
a date or an identifier that crosses a boundary is read whole by the window
that owns its first character. A replacement is kept only by the window owning
its start; one that runs past the owned region extends it, and the next window
begins where it ended. Every character is emitted exactly once, in order, so
nothing is dropped or duplicated, and a window never splits a code point.

One surrogate allocator (spilled, keyed by digest) serves every window, so
the same value receives the same surrogate however far apart it recurs.

A text that fits one window is anonymized exactly as the whole-text path
always did, with the same residual validation. A longer text is then verified
by an independent pass over the written output, again in overlapping windows,
with the residual detectors; it is releasable only if every window of both
passes completed and nothing survived.
"""

from __future__ import annotations

import codecs
import hashlib
import io
from dataclasses import dataclass, field
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple

from services.multiformat import generated_token_provenance
from services.occurrence import merge_summaries
from services.spill_store import (
    EmittedLedger,
    JobSpill,
    SpillUnavailable,
    StreamingAllocator,
)
from services.stream_units import UnitBudget, UnitLedger

REASON_UNITS_INCOMPLETE = "stream_units_incomplete"
REASON_SPILL_UNAVAILABLE = "stream_spill_key_unavailable"
READ_BYTES = 256 * 1024
_BOUNDARY = "\n \t.,;:)]}"
#: Where a review reason was raised, by character range and never by value,
#: so a reviewer can find it. Bounded: a note full of them says so in count.
MAX_REVIEW_LOCATIONS = 64


class TextStreamError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass
class TextOutcome:
    handler_result: Dict[str, Any]
    output_sha256: str = ""
    output_size: int = 0
    windows: int = 0
    ledger: Optional[UnitLedger] = None
    residual: Dict[str, int] = field(default_factory=dict)


class _Decoded:
    """The source as text, decoded incrementally and never split inside a code point."""

    def __init__(self, raw: BinaryIO):
        self._raw = raw
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.digest = hashlib.sha256()
        self.bytes_read = 0
        self.done = False

    def read(self) -> str:
        data = self._raw.read(READ_BYTES)
        if data:
            self.digest.update(data)
            self.bytes_read += len(data)
        try:
            text = self._decoder.decode(data, final=not data)
        except UnicodeDecodeError as exc:
            raise TextStreamError("Text uploads must be UTF-8 encoded") from exc
        if "\x00" in text:
            raise TextStreamError("Text uploads must not contain NUL bytes")
        if not data:
            self.done = True
        return text


def _cut(text: str, start: int, end: int, overlap: int) -> int:
    """Where an owned region should end: at a boundary near ``end``, never before it by much."""
    if end >= len(text):
        return len(text)
    for index in range(end, max(start + 1, end - overlap), -1):
        if text[index - 1] in _BOUNDARY:
            return index
    return end


def window_bounds(buffer: str, span: int, overlap: int, final: bool) -> Tuple[int, int]:
    """Where a verification window ends, and where the next one starts.

    Both fall at a word boundary. A window sliced at a raw offset hands the
    detectors a word cut in half - "ssessment", "potassi", the "y" of "by
    Dr" - and a fragment at the edge of a window reads as a name to the
    proper-noun rule and the model alike. That was the verifier's whole share
    of the 25 KB boundary-note hold. Nothing about the detectors changes: every character
    is still read, the next window still starts at least ``overlap``
    characters before this one ends, and a token too long to hold a boundary
    falls back to the raw offset, where a cut still holds rather than passes.
    """
    if final and len(buffer) <= span:
        return len(buffer), len(buffer)
    end = min(len(buffer), span)
    if not final or end < len(buffer):
        end = _cut(buffer, 0, end, overlap)
    floor = max(1, end - 2 * overlap)
    start = end - overlap
    for index in range(end - overlap, floor - 1, -1):
        if index > 0 and buffer[index - 1] in _BOUNDARY:
            start = index
            break
    return end, max(1, start)


def _default_anonymize(text: str, profile: str, study_salt, allocator):
    from services.text_anonymization import anonymize_clinical_text_with_replacements

    return anonymize_clinical_text_with_replacements(
        text, profile=profile, study_salt=study_salt, allocator=allocator
    )


def _default_residual(text: str, provenance) -> List[Dict[str, Any]]:
    from services.text_anonymization import residual_findings

    return residual_findings(text, provenance)


def _sum_utility(total: Dict[str, float], part: Dict[str, float]) -> None:
    for key in (
        "clinical_terms_in_input",
        "clinical_terms_preserved",
        "content_tokens_in_input",
        "content_tokens_preserved",
        "clinical_numbers_in_input",
        "clinical_numbers_preserved",
    ):
        total[key] = total.get(key, 0) + part.get(key, 0)


def _ratios(total: Dict[str, float]) -> Dict[str, float]:
    from services.text_utility import _ratio

    total = dict(total)
    total["clinical_term_preservation"] = _ratio(
        total.get("clinical_terms_preserved", 0),
        total.get("clinical_terms_in_input", 0),
    )
    total["content_token_preservation"] = _ratio(
        total.get("content_tokens_preserved", 0),
        total.get("content_tokens_in_input", 0),
    )
    total["numeric_preservation"] = _ratio(
        total.get("clinical_numbers_preserved", 0),
        total.get("clinical_numbers_in_input", 0),
    )
    return total


def anonymize_text_stream(
    source_open: Callable[[], BinaryIO],
    size: int,
    sink_factory: Callable[[], BinaryIO],
    reopen_sink: Callable[[], BinaryIO],
    profile: str = "strict",
    study_salt: Optional[str] = None,
    job_id: str = "",
    owner: str = "",
    budget: Optional[UnitBudget] = None,
    window_chars: Optional[int] = None,
    anonymize: Optional[Callable] = None,
    residual: Optional[Callable] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    audit: Optional[Callable[[Any], None]] = None,
) -> TextOutcome:
    """Anonymize a UTF-8 text from a re-readable source into a sink, and verify the sink.

    ``audit``, when given, is called with the ledger of tokens this run wrote,
    while that ledger is still open: the release audit reads the written file
    with the same provenance the verifier had, not with the tokens masked out.
    """
    import secrets

    from services.text_utility import measure_text_utility

    budget = budget or UnitBudget.from_environment()
    window = window_chars or budget.text_window_chars
    overlap = min(budget.text_overlap_chars, max(64, window // 4))
    anonymize = anonymize or _default_anonymize
    residual = residual or _default_residual
    spill = JobSpill(job_id or secrets.token_hex(12), owner)
    allocator = StreamingAllocator(spill)
    emitted = EmittedLedger(spill, allocator)
    ledger = UnitLedger(
        spill.job_id, "text", progress, {"anonymize": 0.6, "verify_text": 0.4}
    )
    ledger.total_bytes = max(1, size)
    try:
        # Every token-shaped string the input already carries keeps its number,
        # whichever window it sits in, before the first replacement is numbered.
        _reserve_input_tokens(source_open, allocator)
        outcome = _transform(
            source_open,
            size,
            sink_factory,
            profile,
            study_salt,
            window,
            overlap,
            anonymize,
            allocator,
            emitted,
            ledger,
            measure_text_utility,
        )
        # A result with no single whole-file window skipped its own residual
        # pass, so the stream verifier is what validates it - whatever the count.
        if outcome.windows > 1 or "surrogate_counts" not in outcome.handler_result:
            outcome.residual = _verify(
                reopen_sink,
                outcome.output_size,
                window,
                overlap,
                residual,
                emitted,
                ledger,
                outcome.handler_result["review_locations"],
            )
            outcome.handler_result["residual_phi_categories"] = dict(
                sorted(outcome.residual.items())
            )
            phases = ["anonymize", "verify_text"]
        else:
            phases = ["anonymize"]
        if not ledger.complete(phases):
            outcome.handler_result["review_required_reasons"] = sorted(
                set(outcome.handler_result.get("review_required_reasons", []))
                | {REASON_UNITS_INCOMPLETE}
            )
            outcome.handler_result["anonymization_status"] = "manual_review_required"
        if outcome.windows > 1 or "surrogate_counts" not in outcome.handler_result:
            outcome.handler_result["surrogate_counts"] = allocator.counts()
        outcome.handler_result["streaming"] = {
            **ledger.summary(),
            "windows": outcome.windows,
            "window_chars": window,
            "overlap_chars": overlap,
            "spill_peak_bytes": spill.peak_bytes,
        }
        outcome.ledger = ledger
        if audit is not None and outcome.output_size:
            audit(emitted)
        return outcome
    except SpillUnavailable:
        raise TextStreamError(REASON_SPILL_UNAVAILABLE, 503) from None
    finally:
        spill.close()


#: A token such as ``<PROVIDER_1234567>`` is shorter than this, so a token cut by
#: a chunk edge is whole in the carried tail of the next chunk.
_TOKEN_CARRY = 64


def _reserve_input_tokens(source_open, allocator) -> None:
    """Read the input once, in bounded chunks, reserving the numbers of its own tokens.

    Windows are processed one at a time with one allocator. Reserving per
    window means a token in window five is reserved only after window one may
    already have written the same token for a real value; reading the whole
    input first closes that, at the price of one extra sequential read.
    """
    import codecs

    raw = source_open()
    try:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        carry = ""
        while True:
            data = raw.read(256 * 1024)
            text = carry + decoder.decode(data, final=not data)
            allocator.reserve_existing(text)
            if not data:
                return
            carry = text[-_TOKEN_CARRY:]
    finally:
        raw.close()


def _transform(
    source_open,
    size,
    sink_factory,
    profile,
    study_salt,
    window,
    overlap,
    anonymize,
    allocator,
    emitted,
    ledger,
    measure,
) -> TextOutcome:
    raw = source_open()
    decoded = _Decoded(raw)
    sink = sink_factory()
    out_digest = hashlib.sha256()
    out_size = 0
    buffer = ""
    base = 0  # global offset of buffer[0]
    own_start = 0  # global offset of the next character to emit
    windows = 0
    single: Optional[Dict[str, Any]] = None
    detected: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    reviews: set = set()
    decisions: Dict[str, Any] = {}
    locations: List[Dict[str, Any]] = []
    residual: Dict[str, int] = {}
    utility: Dict[str, float] = {}
    entity_count = 0
    bytes_emitted = 0

    def emit(text: str) -> None:
        nonlocal out_size
        data = text.encode("utf-8")
        out_digest.update(data)
        out_size += len(data)
        sink.write(data)

    try:
        while True:
            while (
                not decoded.done and len(buffer) - (own_start - base) < window + overlap
            ):
                buffer += decoded.read()
            local = own_start - base
            if windows == 0 and decoded.done and not buffer.strip():
                raise TextStreamError("Text input is empty")
            if local >= len(buffer) and decoded.done:
                break
            unit = ledger.open("anonymize", windows, bytes_emitted)
            end = _cut(buffer, local, min(len(buffer), local + window), overlap)
            if decoded.done and len(buffer) - end <= overlap // 2:
                end = len(buffer)
            left = max(0, local - overlap)
            # Start the window on a word, as the verifier's windows do: a
            # window opened mid-word hands the detectors "gittal" for
            # "sagittal", which reads as a name.
            for index in range(left, local):
                if index == 0 or buffer[index - 1] in _BOUNDARY:
                    left = index
                    break
            right = min(len(buffer), end + overlap)
            # The right edge is aligned the same way: a window that ends mid-word
            # hands the detectors "Nutriti" for "Nutrition", which reads as a name.
            # It never ends before the owned region does.
            if right < len(buffer):
                for index in range(right, max(end, right - overlap), -1):
                    if buffer[index - 1] in _BOUNDARY:
                        right = index
                        break
            text = buffer[left:right]
            whole = left == 0 and right == len(buffer) and decoded.done and windows == 0
            if not text.strip():
                owned = buffer[local:end]
                emit(owned)
                bytes_emitted += len(owned.encode("utf-8"))
                own_start = base + end
                windows += 1
                ledger.close(unit, windows, bytes_emitted, output_bytes=out_size)
                continue
            ledger.checkpoint("stream.text.anonymize.model")
            # Only a window that is the whole file validates its own output; any
            # other is read again from the written file by ``_verify``, which
            # supplies the residual result for the whole document.
            from services.text_anonymization import deferred_residual

            with deferred_residual(not whole):
                result, applied = anonymize(text, profile, study_salt, allocator)
            windows += 1
            reviews.update(result.get("review_required_reasons") or [])
            for reason in sorted(result.get("review_required_reasons") or []):
                _locate(
                    locations,
                    reason,
                    "anonymize",
                    windows - 1,
                    base + left,
                    base + right,
                )
            for name, count in (result.get("detection_sources") or {}).items():
                sources[name] = sources.get(name, 0) + count
            merge_summaries(decisions, result.get("occurrence_decisions") or {})
            if whole:
                single = result
                output = result["anonymized_text"]
                emit(output)
                bytes_emitted += len(text.encode("utf-8"))
                own_start = base + len(buffer)
                for _start, _end, token_type, replacement in applied:
                    emitted.add(replacement.strip())
                ledger.close(unit, windows, bytes_emitted, output_bytes=out_size)
                continue
            own_left, own_right = local - left, end - left
            kept: List[Tuple[int, int, str, str]] = []
            for start, stop, entity_type, replacement in sorted(applied):
                if own_left <= start < own_right and (not kept or start >= kept[-1][1]):
                    kept.append((start, stop, entity_type, replacement))
                    own_right = max(own_right, stop)
            pieces: List[str] = []
            cursor = own_left
            removed: List[Tuple[int, int]] = []
            for start, stop, entity_type, replacement in kept:
                pieces.append(text[cursor:start])
                pieces.append(replacement)
                removed.append((start - own_left, stop - own_left))
                emitted.add(replacement.strip())
                detected[entity_type] = detected.get(entity_type, 0) + 1
                entity_count += 1
                cursor = stop
            pieces.append(text[cursor:own_right])
            owned_output = "".join(pieces)
            owned_source = text[own_left:own_right]
            _sum_utility(
                utility, measure(owned_source, owned_output, redacted_spans=removed)
            )
            emit(owned_output)
            bytes_emitted += len(owned_source.encode("utf-8"))
            own_start = base + left + own_right
            ledger.close(unit, windows, bytes_emitted, output_bytes=out_size)
            # Keep only what the next window needs: its left overlap onward.
            drop = max(0, (own_start - base) - overlap)
            # The buffer starts on a word. A raw cut would leave its first character
            # in the middle of one, and the next window, which treats the start of the
            # buffer as a boundary, would open on "rotein" for "protein". A word too
            # long to hold a boundary falls back to the raw offset.
            for index in range(drop, max(0, drop - overlap), -1):
                if buffer[index - 1] in _BOUNDARY:
                    drop = index
                    break
            if drop:
                buffer = buffer[drop:]
                base += drop
        if raw.read(1):
            raise TextStreamError("Text input changed while it was read")
    finally:
        raw.close()
        sink.flush()
        sink.close()
    if decoded.bytes_read != size:
        raise TextStreamError("Text input changed while it was read")
    if single is not None:
        handler = {
            "residual_phi_categories": single["residual_phi_categories"],
            "utility_metrics": single.get("utility_metrics", {}),
            "review_required_reasons": single.get("review_required_reasons", []),
            "review_locations": locations,
            "anonymization_status": single["anonymization_status"],
            "date_strategy": single["date_strategy"],
            "text_identifier_strategy": single["text_identifier_strategy"],
            "detected_entities": single["detected_entities"],
            "entity_count": single["entity_count"],
            "detection_sources": single["detection_sources"],
            "ner_model": single["ner_model"],
            "trained_ner_active": single["trained_ner_active"],
            "surrogate_counts": single.get("surrogate_counts", {}),
            "occurrence_decisions": single.get("occurrence_decisions", {}),
        }
    else:
        from services.text_anonymization import _profile_settings, configured_model_name

        _profile, settings = _profile_settings(profile)
        handler = {
            "residual_phi_categories": residual,
            "utility_metrics": _ratios(utility),
            "review_required_reasons": sorted(reviews),
            "review_locations": locations,
            "anonymization_status": "completed",
            "date_strategy": settings["date_strategy"],
            "text_identifier_strategy": settings["text_identifier_strategy"],
            "detected_entities": dict(sorted(detected.items())),
            "entity_count": entity_count,
            "detection_sources": dict(sorted(sources.items())),
            "ner_model": configured_model_name(),
            "trained_ner_active": True,
            "occurrence_decisions": decisions,
        }
    handler["input_sha256"] = decoded.digest.hexdigest()
    return TextOutcome(handler, out_digest.hexdigest(), out_size, windows)


def _locate(
    locations: List[Dict[str, Any]],
    reason: str,
    pass_name: str,
    window: int,
    start: int,
    end: int,
) -> None:
    if len(locations) < MAX_REVIEW_LOCATIONS:
        locations.append(
            {
                "reason": reason,
                "pass": pass_name,
                "window": window,
                "char_start": start,
                "char_end": end,
            }
        )


def _verify(
    reopen_sink,
    size,
    window,
    overlap,
    residual,
    emitted,
    ledger,
    locations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Every output character read again by the residual detectors, windows overlapping."""
    from services.text_anonymization import _residual_summary

    raw = reopen_sink()
    decoded = _Decoded(raw)
    categories: Dict[str, int] = {}
    buffer = ""
    position = 0
    chars = 0
    index = 0
    try:
        while True:
            while not decoded.done and len(buffer) < window + overlap:
                buffer += decoded.read()
            if not buffer.strip() and decoded.done:
                if buffer:
                    unit = ledger.open("verify_text", index, position)
                    position += len(buffer.encode("utf-8"))
                    index += 1
                    ledger.close(unit, index, position, verified=True)
                break
            end, advance = window_bounds(
                buffer, window + overlap, overlap, decoded.done
            )
            text = buffer[:end]
            unit = ledger.open("verify_text", index, position)
            ledger.checkpoint("stream.text.verify.model")
            found = residual(text, generated_token_provenance(text, emitted))
            blocking = _residual_summary(found)
            for category, count in blocking.items():
                categories[category] = categories.get(category, 0) + count
                if locations is not None:
                    _locate(
                        locations,
                        f"residual_{category.lower()}",
                        "verify",
                        index,
                        chars,
                        chars + end,
                    )
            if decoded.done and end == len(buffer):
                advance = end
            position += len(buffer[:advance].encode("utf-8"))
            chars += advance
            index += 1
            ledger.close(unit, index, position, verified=not blocking)
            buffer = buffer[advance:]
            if decoded.done and not buffer:
                break
    finally:
        raw.close()
    if position != size:
        ledger.fail(None, "verify_incomplete")
    return categories


def output_bytes_of(sink: io.BytesIO) -> bytes:
    return sink.getvalue()


def iter_chunks(path, size: int = 1 << 20) -> Iterator[bytes]:
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(size), b""):
            yield block
