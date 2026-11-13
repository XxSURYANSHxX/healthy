"""Background ingestion jobs: submit, poll, cancel.

A large upload can take longer than a client should hold a request open. A job
runs the same ingestion route as ``POST /api/v1/ingest`` on a bounded worker
pool, under its own deadline, and reports a coarse state a client can show:

    queued -> anonymizing -> verifying -> ready | held | failed | cancelled

* Bounded. At most ``max_active`` jobs are queued or running; another is
  refused with 429 rather than queued without limit. Finished jobs are kept
  for ``ttl_seconds`` and at most ``max_retained`` of them, then forgotten.
* Cancellable at checkpoints. Cancelling sets an event that every stage
  checkpoint - the deadline checks each sanitizer already passes through, and
  the one before an artifact is stored - raises on, so a cancelled job stops
  at its next stage and stores nothing after it. A finished job cannot be
  cancelled.
* Owner-bound. A job started by a signed upload is read or cancelled only by
  requests signed by the same wallet. Any other request, an unknown id and a
  malformed id all get the same not-found answer. An unsigned job's id is an
  unguessable capability.
* Polled without signatures. Submission issues a short-lived bearer
  capability, so a browser wallet is asked to sign once per upload rather than
  once per poll. The capability is 256 random bits, returned once and stored
  only as a digest bound to the job and its owner. It grants reading the job
  and downloading its held review copy - never cancelling it and never
  approving anything. It expires, it dies with the job, and a signed
  restoration replaces it, which invalidates the previous one.
* Content-free. The result is exactly the synchronous ingest response; the job
  itself records only states, stage names and codes. When a held upload
  produced complete output, the work hands that candidate over beside its
  result; the manager passes it to the review quarantine and never into the
  job, and discards it if the job ends cancelled.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

from services import request_deadline
from services.upload_auth import UploadError as SearchError

STATE_QUEUED = "queued"
STATE_ANONYMIZING = "anonymizing"
STATE_VERIFYING = "verifying"
STATE_READY = "ready"
STATE_HELD = "held"
#: Processed, not released, and no valid candidate to review: a failure.
ERROR_NO_OUTPUT = "failed_no_output"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"
TERMINAL_STATES = frozenset({STATE_READY, STATE_HELD, STATE_FAILED, STATE_CANCELLED})

WORKERS_ENV_VAR = "BIOBLOCK_INGEST_JOB_WORKERS"
DEFAULT_WORKERS = 2
MAX_ACTIVE_JOBS = 16
MAX_RETAINED_JOBS = 256
RESULT_TTL_SECONDS = 900.0

CAPABILITY_TTL_ENV_VAR = "BIOBLOCK_JOB_CAPABILITY_TTL_SECONDS"
DEFAULT_CAPABILITY_TTL_SECONDS = 30 * 60.0
SCOPE_JOB_READ = "job:read"
SCOPE_REVIEW_DOWNLOAD = "review-artifact:download"
#: Everything a polling capability can ever do. Cancelling needs a wallet
#: signature, and approving a held copy needs its own bound signature.
CAPABILITY_SCOPES: FrozenSet[str] = frozenset({SCOPE_JOB_READ, SCOPE_REVIEW_DOWNLOAD})
#: 32 random bytes, URL-safe base64 without padding.
_CAPABILITY = re.compile(r"[A-Za-z0-9_-]{43}")

ERROR_NOT_FOUND = "not_found"
ERROR_QUEUE_FULL = "ingest_queue_full"
ERROR_FINISHED = "job_already_finished"
ERROR_FAILED = "job_failed"
ERROR_CAPABILITY = "job_capability_invalid"

_JOB_ID = re.compile(r"[A-Za-z0-9_-]{32}")


def capability_ttl_seconds() -> float:
    raw = os.getenv(CAPABILITY_TTL_ENV_VAR, "").strip()
    try:
        value = float(raw) if raw else DEFAULT_CAPABILITY_TTL_SECONDS
    except ValueError:
        value = DEFAULT_CAPABILITY_TTL_SECONDS
    return min(max(value, 60.0), 24 * 60 * 60.0)


def capability_digest(job_id: str, owner: Optional[str], capability: str) -> str:
    """The only form of a capability the server keeps.

    Domain-separated and bound to the job and its owner, so a digest copied
    onto another job, or a capability presented for another job, never matches.
    """
    material = "\0".join(
        ("bioblock-job-capability-v1", job_id, owner or "", capability)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _capability_refused() -> SearchError:
    # One answer for every bad bearer: unknown job, wrong job, wrong token,
    # expired, rotated, or a scope it never had.
    return SearchError(
        401,
        ERROR_CAPABILITY,
        "This upload session has expired. Sign in again.",
        {"WWW-Authenticate": "Bearer"},
    )


class JobRejected(Exception):
    """Work that ended with a client-facing rejection: a status and a fixed message."""

    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(error_code)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


def _not_found() -> SearchError:
    return SearchError(404, ERROR_NOT_FOUND, "The job was not found.")


JOB_DEADLINE_ENV_VAR = "BIOBLOCK_INGEST_JOB_DEADLINE_SECONDS"
#: A background job's whole-run deadline. Longer than a synchronous request's,
#: because nobody holds a connection open for it, and still bounded: expiry
#: fails the job and never releases a partial result.
DEFAULT_JOB_DEADLINE_SECONDS = 900.0
MIN_JOB_DEADLINE_SECONDS = 60.0
MAX_JOB_DEADLINE_SECONDS = 1800.0


def configured_job_deadline() -> float:
    raw = os.getenv(JOB_DEADLINE_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_JOB_DEADLINE_SECONDS
    try:
        value = float(raw)
    except ValueError:
        raise ValueError("invalid_ingest_job_deadline") from None
    if not MIN_JOB_DEADLINE_SECONDS <= value <= MAX_JOB_DEADLINE_SECONDS:
        raise ValueError("invalid_ingest_job_deadline")
    return value


def configured_workers() -> int:
    raw = os.getenv(WORKERS_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_WORKERS
    if not raw.isdigit() or not 1 <= int(raw) <= 16:
        raise ValueError("invalid_ingest_job_workers")
    return int(raw)


@dataclass
class Job:
    job_id: str
    owner: Optional[str]
    created_at: float
    state: str = STATE_QUEUED
    stage: str = ""
    finished_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    error_code: Optional[str] = None
    message: Optional[str] = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    done: threading.Event = field(default_factory=threading.Event, repr=False)
    #: Digest of the current bearer capability, and when it stops working
    #: (wall-clock seconds). Never the capability itself.
    capability_sha256: str = field(default="", repr=False)
    capability_expires_at: float = field(default=0.0, repr=False)
    #: A held job whose exact output was quarantined for review is kept this
    #: long, so its owner can still reach the job while the copy exists.
    review_retention: float = field(default=0.0, repr=False)
    #: Why a held job has no review copy, as a code, when keeping one failed.
    review_unavailable: str = field(default="", repr=False)

    def public_dict(self, now: float) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.state,
            "stage": self.stage or None,
            "elapsed_ms": round(
                ((self.finished_at or now) - self.created_at) * 1000, 1
            ),
            "cancellable": self.state not in TERMINAL_STATES,
        }
        if self.state in (STATE_READY, STATE_HELD):
            # The automatic decision, fixed when the job finished. A later
            # manual review is a separate record and never changes this.
            body["automatic_verification_passed"] = self.state == STATE_READY
        if self.error_code:
            body["error_code"] = self.error_code
            body["message"] = self.message
            body["status_code"] = self.status_code
        if self.result is not None:
            body["result"] = self.result
        return body


@dataclass(frozen=True)
class WorkOutcome:
    """What a job's work returns when it has a held candidate to hand over.

    The candidate travels beside the result, never inside it, so the result a
    client reads can never carry held bytes.
    """

    result: Dict[str, Any]
    held_candidate: Any = field(default=None, repr=False)


class JobManager:
    def __init__(
        self,
        workers: Optional[int] = None,
        max_active: int = MAX_ACTIVE_JOBS,
        max_retained: int = MAX_RETAINED_JOBS,
        ttl_seconds: float = RESULT_TTL_SECONDS,
        deadline_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
        quarantine: Optional[Callable[[Job, Any], float]] = None,
        discard: Optional[Callable[[Job], None]] = None,
    ):
        """``quarantine`` keeps a held job's candidate and returns how long it
        is kept; ``discard`` removes it again if the job ends cancelled."""
        self._executor = ThreadPoolExecutor(
            max_workers=workers or configured_workers(), thread_name_prefix="ingest-job"
        )
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._max_active = max_active
        self._max_retained = max_retained
        self._ttl = ttl_seconds
        self._deadline_seconds = deadline_seconds
        self._clock = clock
        self._quarantine = quarantine
        self._discard = discard

    # -- client operations -----------------------------------------------------------

    def submit(
        self,
        work: Callable[[], Dict[str, Any]],
        owner: Optional[str],
        *,
        bind_job: bool = False,
        discard_unstarted=None,
    ) -> Job:
        with self._lock:
            self._sweep()
            active = sum(
                1 for job in self._jobs.values() if job.state not in TERMINAL_STATES
            )
            if active >= self._max_active:
                raise SearchError(
                    429,
                    ERROR_QUEUE_FULL,
                    "Too many uploads are being processed. Try again shortly.",
                )
            job = Job(secrets.token_urlsafe(24), owner, self._clock())
            self._jobs[job.job_id] = job
        self._executor.submit(
            self._run, job, (lambda: work(job)) if bind_job else work, discard_unstarted
        )
        return job

    def issue_capability(self, job: Job) -> str:
        """A new bearer for this job, returned once. Any earlier one stops working.

        Called once after a signed submission and once per signed restoration.
        The plaintext exists only in this return value.
        """
        capability = secrets.token_urlsafe(32)
        with self._lock:
            job.capability_sha256 = capability_digest(job.job_id, job.owner, capability)
            job.capability_expires_at = time.time() + capability_ttl_seconds()
        return capability

    def authorize_capability(self, job_id: str, capability: str, scope: str) -> Job:
        """The job a bearer names, if the bearer is current and has this scope."""
        with self._lock:
            self._sweep()
            job = self._jobs.get(job_id) if _JOB_ID.fullmatch(job_id or "") else None
            if (
                job is None
                or scope not in CAPABILITY_SCOPES
                or not _CAPABILITY.fullmatch(capability or "")
                or not job.capability_sha256
                or time.time() > job.capability_expires_at
                or not hmac.compare_digest(
                    capability_digest(job.job_id, job.owner, capability),
                    job.capability_sha256,
                )
            ):
                raise _capability_refused()
            return job

    def get(self, job_id: str, owner: Optional[str]) -> Job:
        with self._lock:
            self._sweep()
            job = self._jobs.get(job_id) if _JOB_ID.fullmatch(job_id or "") else None
            if job is None or job.owner != owner:
                raise _not_found()
            return job

    def cancel(self, job_id: str, owner: Optional[str]) -> Job:
        job = self.get(job_id, owner)
        with self._lock:
            if job.state in TERMINAL_STATES:
                raise SearchError(409, ERROR_FINISHED, "The job has already finished.")
            job.cancel.set()
        return job

    def wait(self, job_id: str, owner: Optional[str], timeout: float) -> Job:
        job = self.get(job_id, owner)
        job.done.wait(timeout)
        return job

    def status(self, job: Job) -> Dict[str, Any]:
        with self._lock:
            return job.public_dict(self._clock())

    def shutdown(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                job.cancel.set()
        self._executor.shutdown(wait=True)

    # -- the worker ------------------------------------------------------------------------

    def _run(
        self, job: Job, work: Callable[[], Dict[str, Any]], discard_unstarted=None
    ) -> None:
        with self._lock:
            if job.cancel.is_set():
                if discard_unstarted is not None:
                    discard_unstarted()
                self._finish(
                    job,
                    STATE_CANCELLED,
                    error_code=request_deadline.ERROR_CANCELLED,
                    message="The job was cancelled.",
                )
                return
            job.state = STATE_ANONYMIZING
        binding = request_deadline.bind_job(
            job.cancel, lambda stage: self._progress(job, stage)
        )
        try:
            request_deadline.start(self._deadline_seconds)
            result = work()
        except request_deadline.RequestCancelled:
            self._finish_locked(
                job,
                STATE_CANCELLED,
                error_code=request_deadline.ERROR_CANCELLED,
                message="The job was cancelled.",
            )
        except request_deadline.RequestDeadlineExceeded as exc:
            self._finish_locked(
                job,
                STATE_FAILED,
                status_code=504,
                error_code=exc.error_code,
                message="The upload ran out of time.",
            )
        except JobRejected as exc:
            self._finish_locked(
                job,
                STATE_FAILED,
                status_code=exc.status_code,
                error_code=exc.error_code,
                message=exc.message,
            )
        except Exception:
            # The exception could quote the document; only a code is kept.
            self._finish_locked(
                job,
                STATE_FAILED,
                status_code=500,
                error_code=ERROR_FAILED,
                message="The upload could not be processed.",
            )
        else:
            outcome = result if isinstance(result, WorkOutcome) else WorkOutcome(result)
            self._settle(job, outcome)
        finally:
            if (
                job.state in {STATE_CANCELLED, STATE_FAILED}
                and self._discard is not None
            ):
                self._discard(job)
            request_deadline.clear()
            request_deadline.unbind_job(binding)

    def _settle(self, job: Job, outcome: WorkOutcome) -> None:
        """Finish a job whose work returned, keeping a held candidate if there is one.

        The work finished, but the user may have asked to stop while it was
        finishing. Cancellation is only observed at checkpoints, so a job can
        complete between the last one and the return - and releasing then
        hands back an artifact somebody cancelled. The decision and the
        cancellation check are therefore made together, under the lock that
        ``cancel`` takes, and a candidate kept for a job that ends cancelled
        is discarded again.
        """
        result = outcome.result
        decision = (result or {}).get("release_decision") or {}
        state = STATE_READY if decision.get("releasable") is True else STATE_HELD
        # Only an owned job can hold a review copy at all; an unsigned upload
        # stays held with nothing kept, as before.
        no_output = (
            state == STATE_HELD
            and bool(job.owner)
            and outcome.held_candidate is None
            and decision.get("disposition") != "expert_determination_required"
        )
        retention = 0.0
        if (
            state == STATE_HELD
            and outcome.held_candidate is not None
            and job.owner
            and self._quarantine is not None
            and not job.cancel.is_set()
        ):
            try:
                retention = float(self._quarantine(job, outcome.held_candidate))
            except Exception as exc:
                # Keeping a review copy is never part of the automatic result:
                # a key or store failure must not release anything and must
                # not turn the hold into a failure. The job says only why.
                code = getattr(exc, "error_code", "")
                job.review_unavailable = (
                    code
                    if re.fullmatch(r"[a-z_]{3,64}", code or "")
                    else ("review_artifact_store_failed")
                )
        with self._lock:
            cancelled = job.cancel.is_set()
            if cancelled:
                self._finish(
                    job,
                    STATE_CANCELLED,
                    error_code=request_deadline.ERROR_CANCELLED,
                    message="The job was cancelled.",
                )
            else:
                job.review_retention = retention
                if no_output:
                    # Nothing reviewable was produced: a failure with its
                    # privacy report, never a hold with no copy behind it.

                    code, message = no_output_code(result)
                    self._finish(
                        job,
                        STATE_FAILED,
                        result=result,
                        status_code=200,
                        error_code=code,
                        message=message,
                    )
                else:
                    self._finish(job, state, result=result, status_code=200)
        if cancelled and self._discard is not None:
            try:
                self._discard(job)
            except Exception:
                # Discard failed: the copy still expires, and the job is
                # cancelled, so no route will ever serve it.
                pass

    def _progress(self, job: Job, stage: str) -> None:
        with self._lock:
            if job.state in TERMINAL_STATES:
                return
            job.stage = stage
            if stage.rsplit(".", 1)[-1].startswith("verify"):
                job.state = STATE_VERIFYING
            elif job.state != STATE_VERIFYING:
                job.state = STATE_ANONYMIZING

    def _finish_locked(self, job: Job, state: str, **fields: Any) -> None:
        with self._lock:
            self._finish(job, state, **fields)

    def _finish(self, job: Job, state: str, **fields: Any) -> None:
        # A job settles once. Without this, a worker returning after any
        # terminal decision would overwrite it, and the decision most worth
        # protecting is the one the user made.
        if job.state in TERMINAL_STATES:
            return
        job.state = state
        job.finished_at = self._clock()
        for name, value in fields.items():
            setattr(job, name, value)
        job.done.set()

    def _sweep(self) -> None:
        now = self._clock()
        finished = sorted(
            (job for job in self._jobs.values() if job.state in TERMINAL_STATES),
            key=lambda job: job.finished_at or 0.0,
        )
        for job in finished:
            if now - (job.finished_at or now) > max(self._ttl, job.review_retention):
                self._forget(job)
        finished = [job for job in finished if job.job_id in self._jobs]
        while len(finished) > self._max_retained:
            self._forget(finished.pop(0))

    def _forget(self, job: Job) -> None:
        if self._discard is not None:
            self._discard(job)
        del self._jobs[job.job_id]


def no_output_code(result: Dict[str, Any]) -> Tuple[str, str]:
    """Why a finished file has no artifact: an unsupported subtype, or no valid file.

    A valid, supported file held for privacy always keeps a review copy, so
    failed_no_output means no structurally valid artifact could be built.
    """
    summary = ((result or {}).get("multiformat") or {}).get("summary") or {}
    if summary.get("unsupported_subtype"):
        return ERROR_UNSUPPORTED_SUBTYPE, UNSUPPORTED_SUBTYPE_MESSAGE
    return ERROR_NO_OUTPUT, NO_OUTPUT_MESSAGE


ERROR_UNSUPPORTED_SUBTYPE = "unsupported_subtype"
UNSUPPORTED_SUBTYPE_MESSAGE = "This file variant cannot be safely verified."
NO_OUTPUT_MESSAGE = "No valid protected file was created, so there is no review copy."
