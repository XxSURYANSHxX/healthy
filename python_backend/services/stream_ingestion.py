"""Single-file CSV/TXT/XLSX jobs and private, digest-bound candidate delivery.

The format engines come from the development implementation. This adapter keeps
their streaming and artifact contracts independent of batch, search and manual
approval workflows. A held copy is encrypted and is never an automatic release.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

from services import (
    csv_stream,
    detector_capability,
    release_audit,
    request_deadline,
    resource_policy,
    stream_seal,
    tabular_privacy,
    text_anonymization,
    text_stream,
    workbook_sanitizer,
)
from services.ingest_jobs import JobManager, WorkOutcome
from services.multiformat import MultiformatError
from services.privacy_contracts import (
    StreamedArtifact,
    expert_determination_decision,
    issue_release,
    manual_review_decision,
)
from services.sanitized_artifacts import file_sha256, output_directory
from services.upload_auth import UploadError
from services.utility_contract import contract_for

CHUNK = 1024 * 1024
MEDIA = {
    "csv": "text/csv",
    "text": "text/plain",
    "workbook": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
EXTENSIONS = {"csv": ".csv", "text": ".txt", "workbook": ".xlsx"}

_SAFE_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{2,63}")


def _ingestion_error_code(detail, status_code):
    """Source ingestion error classification, without exposing document details."""
    if _SAFE_ERROR_CODE.fullmatch(detail):
        return detail
    lowered = detail.casefold()
    if status_code == 413:
        return "upload_too_large"
    if status_code == 415:
        return "unsupported_file_variant"
    if any(
        word in lowered
        for word in ("malformed", "invalid", "corrupt", "empty", "nul", "utf-8")
    ):
        return "malformed_file"
    return "internal_processing_failure" if status_code >= 500 else "upload_rejected"


def audit_scanner(scanner, emitted):
    """Bind a supplied model scanner to this run's authenticated token ledger."""
    if scanner is None:
        return release_audit.artifact_scanner(emitted)
    if hasattr(scanner, "emitted_tokens"):
        return scanner

    def bound(text):
        return scanner(text)

    setattr(
        bound,
        release_audit.READS_EXACT_TEXT,
        getattr(scanner, release_audit.READS_EXACT_TEXT, False),
    )
    bound.emitted_tokens = emitted
    return bound


class CandidateDownload:
    """Bounded iterator with idempotent cleanup, including an unopened response."""

    def __init__(self, stream, cleanup):
        self.stream, self.cleanup = stream, cleanup
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._closed:
                raise StopIteration
            try:
                request_deadline.check("stream.download")
                block = self.stream.read(CHUNK)
            except BaseException:
                self._closed = True
                self.cleanup()
                raise
        if not block:
            self.close()
            raise StopIteration
        return block

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self.cleanup()


@dataclass
class Candidate:
    directory: Path
    path: Path
    sha256: str
    size: int
    modality: str
    key_version: str = ""
    wrapped_key: str = ""
    leases: int = 0
    expired: bool = False


class Pipeline:
    def __init__(self, **manager_options):
        self._candidates = {}
        self._lock = threading.Lock()
        self._reservations = {}
        self.manager = JobManager(
            quarantine=self.quarantine, discard=self.discard, **manager_options
        )

    def has_candidate(self, job_id):
        with self._lock:
            candidate = self._candidates.get(job_id)
            return candidate is not None and not candidate.expired

    def shutdown(self):
        self.manager.shutdown()
        for job_id in list(self._candidates):
            self._discard_id(job_id)

    def discard(self, job):
        self._discard_id(job.job_id)

    def reserve(self, modality, size, policy):
        """Reserve future working space as well as checking today's free space."""
        with self._lock:
            live = resource_policy.detect(output_directory(), gpu=False)
            refusal = policy.admit(
                modality, size, live, sum(self._reservations.values())
            )
            if refusal:
                raise UploadError(
                    413 if refusal == "file_exceeds_capacity" else 503,
                    refusal,
                    "There is not enough capacity for this upload.",
                )
            token = secrets.token_hex(16)
            self._reservations[token] = size
            return token

    def release_reservation(self, token):
        with self._lock:
            self._reservations.pop(token, None)

    def _discard_id(self, job_id):
        with self._lock:
            candidate = self._candidates.get(job_id)
            if candidate is None:
                return
            candidate.expired = True
            if not candidate.leases:
                shutil.rmtree(candidate.directory)
                del self._candidates[job_id]

    def quarantine(self, job, candidate):
        # The candidate is already sealed by work(), before its plaintext is removed.
        with self._lock:
            if (
                self._candidates.get(job.job_id) is not candidate
                or not candidate.wrapped_key
            ):
                raise ValueError("candidate_unavailable")
        return 900.0

    def work(self, job, directory, modality, profile, manifest=None, *, options=None):
        """Transform staged input, audit written output, and bind the release to it."""
        options = options or {}
        source, output = directory / "input", directory / (
            "output" + EXTENSIONS[modality]
        )
        retained = False
        try:
            if profile == "research":
                from services.ingestion import _expert_determination_response

                return _expert_determination_response(
                    "upload" + EXTENSIONS[modality], modality
                )
            request_deadline.check("stream.admitted")
            summary, reasons = {}, set()
            if modality == "csv":
                outcome = csv_stream.anonymize_csv_stream(
                    csv_stream.Source.of_path(source),
                    lambda: open(output, "wb"),
                    lambda: open(output, "rb"),
                    profile=profile,
                    limits=csv_stream.streaming_limits(source.stat().st_size),
                    job_id=job.job_id,
                    owner=job.owner or "",
                    tabular_manifest=manifest,
                    text_anonymizer=options.get("text_anonymizer"),
                    residual_scanner=options.get("residual_scanner"),
                )
                result = outcome.result
                if outcome.output_size:
                    result.output_path = str(output)
                    result.output_digest, result.output_size = (
                        outcome.output_sha256,
                        outcome.output_size,
                    )
                report = (
                    release_audit.audit_csv_file(
                        output,
                        text_scanner=audit_scanner(
                            options.get("audit_scanner"), result.generated_tokens
                        ),
                        finding_scanner=options.get(
                            "finding_scanner", text_anonymization.residual_findings
                        ),
                        provenance=result.summary,
                    )
                    if outcome.output_size
                    else None
                )
                summary = result.summary
                reasons.update(result.reason_codes)
                if not result.releasable and not reasons:
                    reasons.add("residual_verification_incomplete")
                size, digest = outcome.output_size, outcome.output_sha256
            elif modality == "workbook":
                # Admission caps this whole-file format at 32 MiB, unlike CSV/TXT.
                result = workbook_sanitizer.anonymize_workbook_bytes(
                    source.read_bytes(), profile=profile, tabular_manifest=manifest
                )
                if result.output_bytes:
                    output.write_bytes(result.output_bytes)
                report = release_audit.audit_released_artifact(
                    result.output_bytes or b"",
                    "workbook",
                    provenance=result.summary,
                    text_scanner=audit_scanner(
                        options.get("audit_scanner"), result.generated_tokens
                    ),
                    finding_scanner=options.get(
                        "finding_scanner", text_anonymization.residual_findings
                    ),
                )
                summary = result.summary
                reasons.update(result.reason_codes)
                if not result.releasable and not reasons:
                    reasons.add("residual_verification_incomplete")
                size = len(result.output_bytes or b"")
                digest = hashlib.sha256(result.output_bytes or b"").hexdigest()
            else:
                audits = []

                def audit(emitted):
                    audits.append(
                        release_audit.audit_text_file(
                            output,
                            text_scanner=audit_scanner(
                                options.get("audit_scanner"), emitted
                            ),
                        )
                    )

                with text_anonymization.replacing_foreign_tokens():
                    outcome = text_stream.anonymize_text_stream(
                        lambda: open(source, "rb"),
                        source.stat().st_size,
                        lambda: open(output, "wb"),
                        lambda: open(output, "rb"),
                        profile=profile,
                        job_id=job.job_id,
                        owner=job.owner or "",
                        anonymize=options.get("text_anonymizer"),
                        residual=options.get("residual_scanner"),
                        audit=audit,
                    )
                summary = outcome.handler_result
                reasons.update(summary.get("review_required_reasons") or [])
                if summary.get("residual_phi_categories"):
                    reasons.add("privacy_requirements_not_met")
                if summary.get("anonymization_status") != "completed":
                    reasons.add("residual_verification_incomplete")
                verdict = contract_for("text").evaluate(
                    summary.get("utility_metrics", {})
                )
                summary["utility_verdict"] = verdict.to_public_dict()
                reasons.update(verdict.reason_codes())
                report = audits[0] if audits else None
                size, digest = outcome.output_size, outcome.output_sha256
            summary["release_audit"] = (
                report.public_dict() if report else {"passed": False}
            )
            if report is None or not report.passed:
                reasons.add(modality + "_release_audit_failed")
            request_deadline.check("stream.store")
            if (
                not size
                or not output.is_file()
                or output.stat().st_size != size
                or file_sha256(output) != (digest, size)
            ):
                reasons.add("invalid_sanitized_candidate")
                size = 0
            request_deadline.check("stream.final_decision")
            blocker = detector_capability.release_blocker()
            if blocker:
                reasons.add(blocker)
            decision = (
                manual_review_decision(*sorted(reasons))
                if reasons
                else issue_release(
                    StreamedArtifact(
                        str(output),
                        digest,
                        size,
                        MEDIA[modality],
                        "sanitized" + EXTENSIONS[modality],
                        ("reopened_output_verification", "final_byte_release_audit"),
                    )
                )
            )
            from services.ingestion import _blocked_downstream, _manifest_for

            body = {
                "status": "success",
                "filename": "upload" + EXTENSIONS[modality],
                "detected_modality": modality,
                "privacy_profile": profile,
                "privacy_policy": decision.policy.value,
                "handler": "anonymize_"
                + ("workbook" if modality == "workbook" else modality),
                "routing_status": "handler_selected",
                "anonymization_status": (
                    "completed" if decision.releasable else "manual_review_required"
                ),
                "message": "The written candidate was inspected; the release decision reports the outcome.",
                "release_decision": decision.to_public_dict(),
                "summary": summary,
                "downstream": _blocked_downstream(
                    "pending" if decision.releasable else "blocked"
                ),
            }
            body[modality + "_summary"] = summary
            if modality == "csv":
                body["tabular_summary"] = {
                    "rows_in": summary.get("rows", 0),
                    "rows_out": summary.get("rows", 0) if size else 0,
                    "output_columns": summary.get("columns", 0),
                }
            body["transformation_manifest"] = _manifest_for(
                modality,
                {"unscannable_reasons": sorted(reasons)},
                decision,
                digest if size else "unidentified_artifact",
                decision.policy.value,
            )
            policy = summary.get("tabular_privacy")
            if isinstance(policy, dict):
                policy["candidate_state"] = (
                    "ready"
                    if decision.releasable
                    else "held" if size else "failed_no_output"
                )
            if size and (decision.releasable or job.owner):
                candidate = Candidate(directory, output, digest, size, modality)
                if not decision.releasable:
                    sealed = directory / "candidate.sealed"
                    try:
                        candidate.key_version, candidate.wrapped_key, _ = (
                            stream_seal.seal_file(
                                output,
                                sealed,
                                job.owner,
                                digest,
                                MEDIA[modality],
                                job.job_id,
                            )
                        )
                    except request_deadline.RequestDeadlineExceeded:
                        raise
                    except Exception:
                        if isinstance(policy, dict):
                            policy["candidate_state"] = "failed_no_output"
                        body["review_unavailable"] = "review_artifact_store_failed"
                        return WorkOutcome(body)
                    output.unlink()
                    candidate.path = sealed
                request_deadline.check("stream.retain")
                with self._lock:
                    self._candidates[job.job_id] = candidate
                retained = True
                return WorkOutcome(body, candidate if not decision.releasable else None)
            return WorkOutcome(body)
        except tabular_privacy.ManifestError:
            from services.ingest_jobs import JobRejected

            raise JobRejected(
                422,
                "tabular_policy_schema_mismatch",
                "The policy does not match the table schema.",
            ) from None
        except (
            text_stream.TextStreamError,
            text_anonymization.TextAnonymizationError,
            MultiformatError,
        ) as exc:
            from services.ingest_jobs import JobRejected

            raise JobRejected(
                exc.status_code,
                _ingestion_error_code(str(exc), exc.status_code),
                "The upload could not be processed.",
            ) from None
        finally:
            source.unlink(missing_ok=True)
            if not retained:
                shutil.rmtree(directory)

    def open_candidate(self, job, *, review=False):
        """Verify before sending headers; serve this same open file in bounded chunks."""
        from services.ingest_jobs import STATE_HELD, STATE_READY

        if job.state != (STATE_HELD if review else STATE_READY):
            raise UploadError(
                409, "candidate_not_available", "No eligible candidate is available."
            )
        with self._lock:
            candidate = self._candidates.get(job.job_id)
            if (
                candidate is None
                or candidate.expired
                or bool(candidate.wrapped_key) != review
            ):
                raise UploadError(
                    404,
                    "candidate_not_available",
                    "No eligible candidate is available.",
                )
            candidate.leases += 1
        temporary, stream = None, None
        try:
            path = candidate.path
            if review:
                temporary = stream_seal.open_file(
                    path,
                    candidate.directory,
                    candidate.key_version,
                    candidate.wrapped_key,
                    job.owner,
                    candidate.sha256,
                    MEDIA[candidate.modality],
                    job.job_id,
                    candidate.size,
                )
                path = temporary
            stream = open(path, "rb")
            digest, size = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(CHUNK), b""):
                request_deadline.check("stream.download.verify")
                digest.update(block)
                size += len(block)
            if digest.hexdigest() != candidate.sha256 or size != candidate.size:
                raise ValueError("candidate_changed")
            stream.seek(0)
        except BaseException as exc:
            self._close_candidate(job.job_id, candidate, stream, temporary)
            if isinstance(
                exc,
                (
                    KeyboardInterrupt,
                    SystemExit,
                    request_deadline.RequestDeadlineExceeded,
                ),
            ):
                raise
            raise UploadError(
                422,
                "invalid_sanitized_candidate",
                "The candidate could not be verified.",
            ) from None

        return candidate, CandidateDownload(
            stream,
            lambda: self._close_candidate(job.job_id, candidate, stream, temporary),
        )

    def _close_candidate(self, job_id, candidate, stream, temporary):
        if stream:
            stream.close()
        if temporary:
            temporary.unlink(missing_ok=True)
        with self._lock:
            candidate.leases -= 1
            if candidate.expired and not candidate.leases:
                shutil.rmtree(candidate.directory)
                self._candidates.pop(job_id, None)


def private_work_directory():
    root = output_directory() / "stream-jobs"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="job-", dir=root))
