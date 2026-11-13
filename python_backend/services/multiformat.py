"""Scoped source projection for CSV/TXT/XLSX streaming and verification."""

from __future__ import annotations

import hashlib
import re
import struct
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AbstractSet, Any, Dict, Iterator, List, Optional, Sequence, Tuple

from services.latency_metrics import stage as _timed_stage
from services.request_deadline import check as _deadline_check
from services.surrogates import (
    LEGACY_SURROGATE_PATTERN,
    SURROGATE_PATTERN,
    SurrogateAllocator,
)
from services.transformation_provenance import (
    KIND_GENERALIZED,
    KIND_PLACEHOLDER,
    KIND_SURROGATE,
    ProvenanceBuilder,
)

FORMAT_TEXT = "text"

FORMAT_CSV = "csv"

FORMAT_PDF = "pdf"

FORMAT_DICOM = "dicom"

FORMAT_NIFTI = "nifti"

FORMAT_WSI = "wsi"

FORMAT_WORKBOOK = "workbook"

FORMAT_IMAGE = "image"

SIGNATURE_PDF = "pdf_header"

SIGNATURE_DICOM = "dicom_preamble"

SIGNATURE_ZIP = "zip_container"

SIGNATURE_GZIP = "gzip_stream"

SIGNATURE_NIFTI1 = "nifti1_header"

SIGNATURE_NIFTI2 = "nifti2_header"

SIGNATURE_TIFF = "tiff_header"

SIGNATURE_BIGTIFF = "bigtiff_header"

SIGNATURE_PNG = "png_header"

SIGNATURE_JPEG = "jpeg_header"

SIGNATURE_TEXT = "utf8_text"

SIGNATURE_UNKNOWN = "unrecognised"

EXPECTED_SIGNATURES: Dict[str, frozenset] = {
    FORMAT_TEXT: frozenset({SIGNATURE_TEXT}),
    FORMAT_CSV: frozenset({SIGNATURE_TEXT}),
    FORMAT_PDF: frozenset({SIGNATURE_PDF}),
    FORMAT_DICOM: frozenset({SIGNATURE_DICOM}),
    FORMAT_WORKBOOK: frozenset({SIGNATURE_ZIP}),
    FORMAT_NIFTI: frozenset({SIGNATURE_NIFTI1, SIGNATURE_NIFTI2}),
    FORMAT_WSI: frozenset({SIGNATURE_TIFF, SIGNATURE_BIGTIFF}),
    # A TIFF may be a slide or an ordinary picture. Both signatures are
    # accepted here and the handler decides which by opening it; routing sends
    # a TIFF to the slide path first and here only when that declines it.
    FORMAT_IMAGE: frozenset(
        {SIGNATURE_PNG, SIGNATURE_JPEG, SIGNATURE_TIFF, SIGNATURE_BIGTIFF}
    ),
}

TEXT_BATCH_BYTES = 32 * 1024

TEXT_SEPARATOR = ".\n\n"


class MultiformatError(ValueError):
    """A request-level failure: bad input, never a hold on a parsed file."""

    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def _nifti_signature(header: bytes) -> Optional[str]:
    if len(header) < 348:
        return None
    for order in ("<", ">"):
        size = struct.unpack(order + "i", header[:4])[0]
        if size == 348 and header[344:348] in (b"n+1\x00", b"ni1\x00"):
            return SIGNATURE_NIFTI1
        if size == 540 and header[4:8] in (b"n+2\x00", b"ni2\x00"):
            return SIGNATURE_NIFTI2
    return None


def gunzip_head(content: bytes, limit: int = 1024) -> bytes:
    """The first ``limit`` decompressed bytes of a gzip stream, or nothing."""
    try:
        return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(content, limit)
    except zlib.error:
        return b""


def _looks_like_text(head: bytes) -> bool:
    if not head or b"\x00" in head:
        return False
    for trim in range(4):
        try:
            head[: len(head) - trim].decode("utf-8")
            return True
        except UnicodeDecodeError:
            continue
    return False


def sniff_signature(head: bytes) -> str:
    """What the leading bytes say the file is, independent of its name."""
    if len(head) >= 132 and head[128:132] == b"DICM":
        return SIGNATURE_DICOM
    if head[:1024].lstrip().startswith(b"%PDF-"):
        return SIGNATURE_PDF
    if head.startswith(b"PK\x03\x04"):
        return SIGNATURE_ZIP
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return SIGNATURE_PNG
    if head[:3] == b"\xff\xd8\xff":
        return SIGNATURE_JPEG
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return SIGNATURE_TIFF
    if head[:4] in (b"II+\x00", b"MM\x00+"):
        return SIGNATURE_BIGTIFF
    if head[:2] == b"\x1f\x8b":
        return SIGNATURE_GZIP
    nifti = _nifti_signature(head)
    if nifti:
        return nifti
    if _looks_like_text(head):
        return SIGNATURE_TEXT
    return SIGNATURE_UNKNOWN


def signature_for(format_name: str, head: bytes, extension: str = "") -> str:
    """The signature that counts for ``format_name``.

    A gzip-compressed NIfTI is judged by the header inside the stream, so a
    ``.nii.gz`` that decompresses to something else is still a mismatch.
    """
    signature = sniff_signature(head)
    if format_name == FORMAT_NIFTI and signature == SIGNATURE_GZIP:
        inner = _nifti_signature(gunzip_head(head, 1024))
        return inner or SIGNATURE_GZIP
    return signature


def signature_matches(format_name: str, head: bytes, extension: str = "") -> bool:
    return signature_for(format_name, head, extension) in EXPECTED_SIGNATURES.get(
        format_name, frozenset()
    )


@dataclass
class MultiformatResult:
    """One handler's verdict. Bytes stay internal; the public view has none."""

    input_format: str
    detected_mime: str
    signature: str
    categories_detected: Dict[str, int] = field(default_factory=dict)
    residual_verification_passed: bool = False
    reason_codes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    duration_ms: float = 0.0
    output_bytes: Optional[bytes] = field(default=None, repr=False)
    output_media_type: Optional[str] = None
    output_extension: Optional[str] = None
    validators: Tuple[str, ...] = ()
    # Exact tokens emitted by this request, retained only for the final-byte
    # release audit. They are never included in ``to_public_dict``.
    generated_tokens: AbstractSet[str] = field(default_factory=frozenset, repr=False)
    # A streamed output is never held in memory: its verified file, digest and
    # size stand in for ``output_bytes``.
    output_path: Optional[str] = field(default=None, repr=False)
    output_digest: Optional[str] = None
    output_size: int = 0

    def hold(self, *codes: str) -> None:
        for code in codes:
            if code and code not in self.reason_codes:
                self.reason_codes.append(code)

    def warn(self, *codes: str) -> None:
        for code in codes:
            if code and code not in self.warnings:
                self.warnings.append(code)

    def error(self, *codes: str) -> None:
        for code in codes:
            if code and code not in self.errors:
                self.errors.append(code)
        self.hold(*codes)

    def add_categories(self, counts: Dict[str, int]) -> None:
        merge_counts(self.categories_detected, counts)

    @property
    def releasable(self) -> bool:
        return (
            (
                bool(self.output_bytes)
                or bool(self.output_path and self.output_digest and self.output_size)
            )
            and self.residual_verification_passed
            and not self.reason_codes
            and not self.errors
        )

    @property
    def manual_review_required(self) -> bool:
        return not self.releasable

    def output_sha256(self) -> Optional[str]:
        if self.output_digest and not self.output_bytes:
            return self.output_digest
        if not self.output_bytes:
            return None
        return hashlib.sha256(self.output_bytes).hexdigest()

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "input_format": self.input_format,
            "detected_mime": self.detected_mime,
            "signature": self.signature,
            "categories_detected": dict(sorted(self.categories_detected.items())),
            "residual_verification_passed": self.residual_verification_passed,
            "manual_review_required": self.manual_review_required,
            "releasable": self.releasable,
            "reason_codes": sorted(self.reason_codes),
            "warnings": sorted(self.warnings),
            "errors": sorted(self.errors),
            "sanitized_output_produced": bool(self.output_bytes or self.output_path),
            "summary": self.summary,
            "stage_timings_ms": dict(self.stage_timings_ms),
            "duration_ms": self.duration_ms,
        }


STRUCTURAL_ERROR = re.compile(
    r"(?:write_failed|reopen_failed|unparseable|undecodable|render_failed|xref_unreadable|"
    r"page_count_changed|pixels_changed|shape_changed|not_reopenable|structure_invalid|"
    r"has_no_pages|empty)"
)


def reviewable_hold(result: "MultiformatResult", reviewable: AbstractSet[str]) -> bool:
    """Whether a held result's artifact is a valid candidate for a person to review.

    Only when processing completed without an error and every reason is an
    uncertainty the handler lists - plus the residual and ambiguity codes the
    text pipeline attaches to them. A surface nothing processed, a value that
    survived, or a failed write makes the artifact something other than an
    anonymized candidate, and it is withheld.
    """
    # The universal review contract: a valid, admitted, supported file that
    # was held for privacy keeps the best-effort artifact produced so far, for
    # its owner to review behind a warning - it may still contain identifying
    # information and is never released, published or indexed. Only an
    # artifact that is not structurally a file of its kind (a failed write,
    # an output that does not reopen or decode) is withheld.
    if not result.reason_codes or not (result.output_bytes or result.output_path):
        return False
    return not any(STRUCTURAL_ERROR.search(code) for code in result.errors)


def merge_counts(target: Dict[str, int], addition: Dict[str, int]) -> None:
    for key, value in (addition or {}).items():
        target[key] = target.get(key, 0) + int(value)


class StageClock:
    """Stage timings for one result, each stage also a deadline checkpoint."""

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.timings: Dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        label = f"{self.prefix}.{name}"
        _deadline_check(label)
        started = time.perf_counter()
        try:
            with _timed_stage(label):
                yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000
            self.timings[name] = round(self.timings.get(name, 0.0) + elapsed, 3)

    def finish(self, result: MultiformatResult) -> MultiformatResult:
        result.stage_timings_ms = dict(self.timings)
        result.duration_ms = round((time.perf_counter() - self._started) * 1000, 3)
        return result


@dataclass(frozen=True)
class SanitizedText:
    """Anonymized text plus the input offsets of every replaced span.

    The offsets exist so a handler can place a redaction on a page or an
    image. They are never serialized.
    """

    text: str
    spans: Tuple[Tuple[int, int, str], ...] = ()
    review_reasons: Tuple[str, ...] = ()
    residual_categories: Tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.spans)


class TextChannelSanitizer:
    """Runs extracted strings through the full_v1 text anonymizer in batches.

    Strings are joined with ``TEXT_SEPARATOR``, a full stop and a blank line,
    which the anonymizer does not remove or introduce. If a detected span
    crosses a boundary anyway, or a batch comes back with the wrong number of
    parts, every string in it is anonymized on its own instead.
    """

    def __init__(
        self, profile: str = "strict", allocator: Optional[SurrogateAllocator] = None
    ):
        self.profile = profile
        #: One allocator for every string of one file, so a value keeps one
        #: token across batches and single retries alike.
        self.allocator = allocator if allocator is not None else SurrogateAllocator()
        self.categories: Dict[str, int] = {}
        self.residual: Dict[str, int] = {}
        self.review_reasons: set = set()
        self.calls = 0

    def _anonymize(self, text: str):
        from services.text_anonymization import anonymize_clinical_text_with_spans

        # StageClock publishes the workbook's coarse stage. Keep the
        # per-cell cancellation/deadline checkpoint without a progress write.
        _deadline_check()
        self.calls += 1
        return anonymize_clinical_text_with_spans(
            text, profile=self.profile, allocator=self.allocator
        )

    def _absorb(self, result: Dict[str, Any]) -> None:
        merge_counts(self.categories, result.get("detected_entities", {}))
        merge_counts(self.residual, result.get("residual_phi_categories", {}))
        self.review_reasons.update(result.get("review_required_reasons", []))

    def _single(self, text: str) -> SanitizedText:
        result, spans = self._anonymize(text)
        self._absorb(result)
        return SanitizedText(
            result["anonymized_text"],
            tuple(spans),
            tuple(sorted(set(result.get("review_required_reasons") or ()))),
            tuple(sorted((result.get("residual_phi_categories") or {}).keys())),
        )

    def sanitize(self, texts: Sequence[str]) -> List[SanitizedText]:
        from services.text_anonymization import MAX_TEXT_BYTES, TextAnonymizationError

        for text in texts:
            self.allocator.reserve_existing(text or "")
        results: List[Optional[SanitizedText]] = [None] * len(texts)
        batch: List[int] = []
        batch_bytes = 0

        def flush() -> None:
            nonlocal batch, batch_bytes
            if batch:
                self._flush(texts, batch, results)
            batch, batch_bytes = [], 0

        for index, text in enumerate(texts):
            if not text or not text.strip():
                results[index] = SanitizedText(text or "")
                continue
            size = len(text.encode("utf-8"))
            if size > MAX_TEXT_BYTES:
                raise TextAnonymizationError(
                    "text_channel_exceeds_scan_limit", status_code=413
                )
            if TEXT_SEPARATOR in text or size >= TEXT_BATCH_BYTES:
                flush()
                results[index] = self._single(text)
                continue
            if batch_bytes + size + len(TEXT_SEPARATOR) > TEXT_BATCH_BYTES:
                flush()
            batch.append(index)
            batch_bytes += size + len(TEXT_SEPARATOR)
        flush()
        return [item if item is not None else SanitizedText("") for item in results]

    def sanitize_isolated(
        self, texts: Sequence[str], surfaces: Optional[Sequence[str]] = None
    ) -> List[SanitizedText]:
        """Keep each surface independent while prefetching two model inputs.

        The model worker receives separate indexed strings. Its proposals are
        cached for this file only; _single still applies the full, separate
        occurrence policy, transformation, residual pass, and verification.
        No delimiter or narrative joining is used.
        """
        from services.local_model_detectors import LocalModelError
        from services.model_client import get_client, worker_enabled
        from services.request_deadline import model_evidence_cache
        from services.text_anonymization import MAX_TEXT_BYTES, TextAnonymizationError

        if surfaces is not None and len(surfaces) != len(texts):
            raise ValueError("surface count does not match text count")
        for value in texts:
            text = value or ""
            self.allocator.reserve_existing(text)
            if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
                raise TextAnonymizationError(
                    "text_channel_exceeds_scan_limit", status_code=413
                )
        results: List[SanitizedText] = []
        prefetch = worker_enabled() and model_evidence_cache() is not None
        client = get_client() if prefetch else None
        index = 0
        while index < len(texts):
            batch = [index]
            first = texts[index] or ""
            if prefetch and first.strip() and index + 1 < len(texts):
                second = texts[index + 1] or ""
                sizes = [len(first.encode("utf-8")), len(second.encode("utf-8"))]
                if (
                    second.strip()
                    and sum(sizes) <= 128 * 1024
                    and sum((size + 3) // 4 for size in sizes) <= 8192
                ):
                    batch.append(index + 1)
            if prefetch and all((texts[item] or "").strip() for item in batch):
                requests = [
                    {
                        "surface": (
                            str(surfaces[item])
                            if surfaces is not None
                            else f"text:{item}"
                        ),
                        "text": texts[item],
                    }
                    for item in batch
                ]
                if (
                    sum(len(item["text"].encode("utf-8")) for item in requests)
                    <= 128 * 1024
                    and sum(
                        (len(item["text"].encode("utf-8")) + 3) // 4
                        for item in requests
                    )
                    <= 8192
                ):
                    for detector in ("stanford", "gliner"):
                        try:
                            client.detect_many(detector, requests)
                        except LocalModelError as exc:
                            raise TextAnonymizationError(
                                exc.error_code, status_code=exc.status_code
                            ) from exc
            for item in batch:
                value = texts[item] or ""
                results.append(
                    self._single(value) if value.strip() else SanitizedText(value)
                )
            index += len(batch)
        return results

    def _flush(self, texts: Sequence[str], batch: List[int], results: List) -> None:
        joined = TEXT_SEPARATOR.join(texts[index] for index in batch)
        result, spans = self._anonymize(joined)
        parts = result["anonymized_text"].split(TEXT_SEPARATOR)
        bounds = []
        cursor = 0
        for index in batch:
            bounds.append((cursor, cursor + len(texts[index])))
            cursor += len(texts[index]) + len(TEXT_SEPARATOR)
        per_item: List[List[Tuple[int, int, str]]] = [[] for _ in batch]
        crossing = False
        for start, end, category in spans:
            owner = next(
                (
                    k
                    for k, (low, high) in enumerate(bounds)
                    if low <= start and end <= high
                ),
                None,
            )
            if owner is None:
                crossing = True
                break
            low = bounds[owner][0]
            per_item[owner].append((start - low, end - low, category))
        if crossing or len(parts) != len(batch):
            for index in batch:
                results[index] = self._single(texts[index])
            return
        self._absorb(result)
        for position, index in enumerate(batch):
            results[index] = SanitizedText(parts[position], tuple(per_item[position]))


_HASH_SURROGATE = re.compile(
    r"\b(?:PERSON|MRN|PATIENT_ID|HEALTH_PLAN|ACCESSION|DEVICE)_[0-9A-F]{8}\b"
)

_PLACEHOLDER = re.compile(r"<REDACTED_[A-Z0-9_]+>")

_AGE_AGGREGATE = re.compile(r"(?<!\d)90\+")

from services.transformation_provenance import generated_token_provenance


@dataclass(frozen=True)
class VerificationOutcome:
    passed: bool
    blocking_categories: Dict[str, int]
    texts_inspected: int


def verify_extracted_text(
    texts: Sequence[str],
    emitted: AbstractSet[str],
    *,
    isolate: bool = False,
) -> VerificationOutcome:
    """Re-inspect text read back from a written output, independently.

    This is the residual validator's own pass - the full_v1 detector chain and
    the separate structured-identifier scan - run over what the file now says,
    not over what the anonymizer believes it wrote.

    ``emitted`` is the mandatory set of replacement tokens this transformation
    actually wrote. A caller with no emitted tokens passes an empty set.
    """
    from services.text_anonymization import residual_findings

    blocking: Dict[str, int] = {}
    pending: List[str] = []
    pending_bytes = 0
    inspected = 0
    seen: set[str] = set()

    def run(chunk: str) -> None:
        if not chunk.strip():
            return
        _deadline_check()
        for finding in residual_findings(
            chunk, generated_token_provenance(chunk, emitted)
        ):
            if finding["blocking"]:
                category = finding["category"]
                blocking[category] = blocking.get(category, 0) + 1

    for text in texts:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        inspected += 1
        if isolate:
            # A workbook cell/comment/formula literal is its own attribution
            # boundary. Identical strings are context-free detector inputs,
            # so scanning each distinct value once preserves parity.
            if cleaned not in seen:
                run(cleaned)
                seen.add(cleaned)
            continue
        size = len(cleaned.encode("utf-8"))
        if size >= TEXT_BATCH_BYTES:
            run(cleaned)
            continue
        if pending_bytes + size + len(TEXT_SEPARATOR) > TEXT_BATCH_BYTES:
            run(TEXT_SEPARATOR.join(pending))
            pending, pending_bytes = [], 0
        pending.append(cleaned)
        pending_bytes += size + len(TEXT_SEPARATOR)
    if pending:
        run(TEXT_SEPARATOR.join(pending))
    return VerificationOutcome(not blocking, blocking, inspected)


def zip_container_problems(
    content: bytes,
    max_members: int = 5000,
    max_uncompressed_bytes: int = 256 * 1024 * 1024,
    max_ratio: int = 200,
) -> List[str]:
    """Reason codes for a zip container that is unsafe to open, else nothing.

    Delegates to ``services.admission``, which owns the limits and also
    refuses a member that names somewhere outside the archive. The signature
    stays for the callers that pass their own ceilings.
    """
    from services import admission

    return admission.container_problems(
        content,
        max_members=max_members,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_ratio=max_ratio,
    )


def sanitized_filename(extension: str, sha256: str) -> str:
    """An output name derived from the output's digest, never the upload's name."""
    return f"sanitized_{sha256[:16]}{extension}"
