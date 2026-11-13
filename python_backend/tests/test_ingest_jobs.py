"""Background ingestion jobs: states, checkpoint cancellation, bounds and owners.

Stand-in work passes the same deadline checkpoints a sanitizer passes, so
states, progress and cancellation are checked exactly; the API tests run real
uploads through the job routes. Every value is invented.
"""

import io
import os
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streaming_support as support  # noqa: E402
from services import ingest_jobs as jobs  # noqa: E402
from services import request_deadline  # noqa: E402
from services import sanitized_artifacts as store  # noqa: E402
from services import upload_auth as auth  # noqa: E402
from services.upload_auth import UploadError as SearchError  # noqa: E402

OWNER = "0x" + "a1" * 20
OTHER = "0x" + "b2" * 20
JOBS = "/api/v1/ingest/jobs"
TABLE = b"diagnosis,systolic_bp\nI10,128\nE11.9,141\n"


@pytest.fixture
def manager():
    created = []

    def make(**options):
        options.setdefault("workers", 2)
        instance = jobs.JobManager(**options)
        created.append(instance)
        return instance

    yield make
    for instance in created:
        instance.shutdown()


def _released(releasable=True):
    return {"status": "success", "release_decision": {"releasable": releasable}}


def _finished(instance, job, owner=None, timeout=10.0):
    return instance.status(instance.wait(job.job_id, owner, timeout))


def _until(predicate, timeout=10.0):
    ends = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > ends:
            raise AssertionError("condition not reached")
        time.sleep(0.01)


def _error(call):
    with pytest.raises(SearchError) as caught:
        call()
    return caught.value.status_code, caught.value.error_code, caught.value.message


# -- the job manager ---------------------------------------------------------------------------


def test_a_releasable_result_is_ready_and_a_held_one_is_held(manager):
    instance = manager()

    ready = instance.submit(lambda: _released(True), OWNER)
    held = instance.submit(lambda: _released(False), OWNER)

    ready_status = _finished(instance, ready, OWNER)
    assert ready_status["status"] == "ready" and ready_status["result"] == _released(
        True
    )
    assert not ready_status["cancellable"]
    # Held with no candidate behind it: nothing to review, so a failure that
    # keeps its report - never a hold that offers nothing.
    no_output = _finished(instance, held, OWNER)
    assert (
        no_output["status"] == "failed"
        and no_output["error_code"] == "failed_no_output"
    )
    assert no_output["result"] == _released(False)


def test_a_held_result_with_a_candidate_is_held(manager):
    from services.ingest_jobs import WorkOutcome

    instance = manager()
    held = instance.submit(
        lambda: WorkOutcome(_released(False), held_candidate=object()), OWNER
    )

    assert _finished(instance, held, OWNER)["status"] == "held"


def test_an_unsigned_hold_stays_held_with_nothing_kept(manager):
    instance = manager()
    held = instance.submit(lambda: _released(False), None)

    assert _finished(instance, held, None)["status"] == "held"


def test_the_state_follows_the_stage_checkpoints(manager):
    instance = manager()
    gates = [threading.Event(), threading.Event()]

    def work():
        request_deadline.check("csv.anonymize_text")
        gates[0].wait(5)
        request_deadline.check("csv.verify")
        gates[1].wait(5)
        return _released()

    job = instance.submit(work, None)

    _until(lambda: instance.status(job)["stage"] == "csv.anonymize_text")
    assert instance.status(job)["status"] == "anonymizing"
    gates[0].set()
    _until(lambda: instance.status(job)["stage"] == "csv.verify")
    assert instance.status(job)["status"] == "verifying"
    gates[1].set()
    assert _finished(instance, job)["status"] == "ready"


def test_a_cancelled_job_stops_at_its_next_checkpoint(manager):
    instance = manager()
    started, gate, after = threading.Event(), threading.Event(), threading.Event()

    def work():
        request_deadline.check("dicom.decode_pixels")
        started.set()
        gate.wait(5)
        request_deadline.check("dicom.write")
        after.set()
        return _released()

    job = instance.submit(work, None)
    started.wait(5)
    instance.cancel(job.job_id, None)
    gate.set()

    status = _finished(instance, job)
    assert status["status"] == "cancelled"
    assert status["error_code"] == request_deadline.ERROR_CANCELLED
    assert "result" not in status and not after.is_set()


def test_a_queued_job_cancelled_before_it_starts_never_runs(manager):
    instance = manager(workers=1)
    block, ran = threading.Event(), threading.Event()

    first = instance.submit(lambda: (block.wait(5), _released())[1], None)
    second = instance.submit(lambda: (ran.set(), _released())[1], None)
    instance.cancel(second.job_id, None)
    block.set()

    assert _finished(instance, second)["status"] == "cancelled"
    assert _finished(instance, first)["status"] == "ready"
    assert not ran.is_set()


def test_a_finished_job_cannot_be_cancelled(manager):
    instance = manager()
    job = instance.submit(_released, None)
    _finished(instance, job)

    assert _error(lambda: instance.cancel(job.job_id, None))[:2] == (
        409,
        jobs.ERROR_FINISHED,
    )


def test_the_queue_is_bounded(manager):
    instance = manager(workers=1, max_active=1)
    block = threading.Event()
    instance.submit(lambda: (block.wait(5), _released())[1], None)

    try:
        assert _error(lambda: instance.submit(_released, None))[:2] == (
            429,
            jobs.ERROR_QUEUE_FULL,
        )
    finally:
        block.set()


def test_another_owner_an_unknown_id_and_a_malformed_id_get_the_same_answer(manager):
    instance = manager()
    job = instance.submit(_released, OWNER)
    _finished(instance, job, OWNER)

    answers = {
        _error(lambda: instance.get(job.job_id, OTHER)),
        _error(lambda: instance.get(job.job_id, None)),
        _error(lambda: instance.get("x" * 32, OWNER)),
        _error(lambda: instance.get("../jobs", OWNER)),
        _error(lambda: instance.cancel(job.job_id, OTHER)),
    }

    assert answers == {(404, jobs.ERROR_NOT_FOUND, "The job was not found.")}
    assert instance.get(job.job_id, OWNER) is job


def _refused(call):
    return _error(call)[:2] == (401, jobs.ERROR_CAPABILITY)


def test_a_capability_is_random_scoped_and_stored_only_as_a_bound_digest(manager):
    instance = manager()
    job = instance.submit(_released, OWNER)
    capability = instance.issue_capability(job)

    # 32 random bytes; the job holds a digest bound to its id and owner.
    assert len(capability) == 43
    assert capability not in repr(job) and capability not in repr(vars(job))
    assert job.capability_sha256 == jobs.capability_digest(
        job.job_id, OWNER, capability
    )
    assert job.capability_sha256 != jobs.capability_digest(
        job.job_id, OTHER, capability
    )
    for scope in (jobs.SCOPE_JOB_READ, jobs.SCOPE_REVIEW_DOWNLOAD):
        assert instance.authorize_capability(job.job_id, capability, scope) is job
    # Nothing beyond reading and review download, whatever is asked for.
    for scope in ("job:cancel", "manual-approval", "database:promote", ""):
        assert _refused(
            lambda: instance.authorize_capability(job.job_id, capability, scope)
        )


def test_a_capability_opens_only_its_own_job(manager):
    instance = manager()
    first, second = instance.submit(_released, OWNER), instance.submit(_released, OWNER)
    first_capability = instance.issue_capability(first)
    instance.issue_capability(second)

    answers = {
        _error(
            lambda: instance.authorize_capability(
                second.job_id, first_capability, jobs.SCOPE_JOB_READ
            )
        ),
        _error(
            lambda: instance.authorize_capability(
                first.job_id, "x" * 43, jobs.SCOPE_JOB_READ
            )
        ),
        _error(
            lambda: instance.authorize_capability(
                "y" * 32, first_capability, jobs.SCOPE_JOB_READ
            )
        ),
        _error(
            lambda: instance.authorize_capability(first.job_id, "", jobs.SCOPE_JOB_READ)
        ),
    }
    assert len(answers) == 1 and next(iter(answers))[:2] == (401, jobs.ERROR_CAPABILITY)


def test_a_capability_expires_and_a_replacement_revokes_it(manager):
    instance = manager()
    job = instance.submit(_released, OWNER)
    old = instance.issue_capability(job)
    new = instance.issue_capability(job)

    assert _refused(
        lambda: instance.authorize_capability(job.job_id, old, jobs.SCOPE_JOB_READ)
    )
    assert instance.authorize_capability(job.job_id, new, jobs.SCOPE_JOB_READ) is job
    job.capability_expires_at = time.time() - 1
    assert _refused(
        lambda: instance.authorize_capability(job.job_id, new, jobs.SCOPE_JOB_READ)
    )


def test_a_capability_dies_with_its_job(manager):
    now = [100.0]
    instance = manager(ttl_seconds=10, clock=lambda: now[0])
    job = instance.submit(_released, OWNER)
    capability = instance.issue_capability(job)
    _finished(instance, job, OWNER)

    now[0] = 111.0

    assert _refused(
        lambda: instance.authorize_capability(
            job.job_id, capability, jobs.SCOPE_JOB_READ
        )
    )


def test_the_capability_lifetime_is_bounded_whatever_is_configured(monkeypatch):
    for configured, expected in (
        ("", 1800.0),
        ("5", 60.0),
        ("999999", 86400.0),
        ("nonsense", 1800.0),
        ("900", 900.0),
    ):
        monkeypatch.setenv(jobs.CAPABILITY_TTL_ENV_VAR, configured)
        assert jobs.capability_ttl_seconds() == expected


def test_finished_jobs_are_forgotten_after_their_retention(manager):
    now = [100.0]
    instance = manager(ttl_seconds=10, clock=lambda: now[0])
    job = instance.submit(_released, None)
    _finished(instance, job)

    now[0] = 111.0

    assert _error(lambda: instance.get(job.job_id, None))[0] == 404


def test_only_a_bounded_number_of_finished_jobs_is_kept(manager):
    instance = manager(max_retained=2)
    finished = []
    for _ in range(4):
        job = instance.submit(_released, None)
        _finished(instance, job)
        finished.append(job)

    assert _error(lambda: instance.get(finished[0].job_id, None))[0] == 404
    assert instance.get(finished[-1].job_id, None) is finished[-1]


def test_a_job_that_runs_out_of_time_fails_with_the_deadline_code(manager):
    instance = manager(deadline_seconds=0.05)

    def work():
        time.sleep(0.2)
        request_deadline.check("pdf.ocr")
        return _released()

    status = _finished(instance, instance.submit(work, None))

    assert status["status"] == "failed"
    assert status["error_code"] == request_deadline.ERROR_REQUEST_TIMEOUT
    assert status["status_code"] == 504


def test_a_progressing_job_can_pass_the_old_120_second_limit_without_sleeping(
    manager, monkeypatch
):
    """Regression for large CSV jobs killed by the former model sub-limit."""
    now = [10.0]
    monkeypatch.setattr(request_deadline.time, "monotonic", lambda: now[0])
    instance = manager(deadline_seconds=300.0, clock=lambda: now[0])

    def work():
        request_deadline.check("csv.anonymize_text")
        now[0] = 131.0
        request_deadline.check("csv.anonymize_text")
        now[0] = 299.0
        request_deadline.check("csv.verify")
        return _released()

    status = _finished(instance, instance.submit(work, None))

    assert status["status"] == "ready"
    assert status["stage"] == "csv.verify"
    assert status["elapsed_ms"] == 289000.0


def test_a_failure_never_carries_the_exception_text(manager):
    instance = manager()

    def leaks():
        raise RuntimeError("Patient Zelquin Vantreske could not be parsed")

    def rejected():
        raise jobs.JobRejected(
            415, "unsupported_media_type", "This file type is not supported."
        )

    failed = _finished(instance, instance.submit(leaks, None))
    refused = _finished(instance, instance.submit(rejected, None))

    assert failed["status"] == "failed" and failed["error_code"] == jobs.ERROR_FAILED
    assert "Vantreske" not in repr(failed)
    assert (refused["status_code"], refused["error_code"]) == (
        415,
        "unsupported_media_type",
    )


def test_a_job_binding_stays_in_its_own_thread():
    cancelled = threading.Event()
    cancelled.set()
    elsewhere = []

    binding = request_deadline.bind_job(cancelled)
    try:
        thread = threading.Thread(
            target=lambda: elsewhere.append(
                request_deadline.check("other.request") is None
            )
        )
        thread.start()
        thread.join()
        with pytest.raises(request_deadline.RequestCancelled):
            request_deadline.check("this.job")
    finally:
        request_deadline.unbind_job(binding)

    assert elsewhere == [True]
    request_deadline.check("after.unbinding")


# -- the routes --------------------------------------------------------------------------------
