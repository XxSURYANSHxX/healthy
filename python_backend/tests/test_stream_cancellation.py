"""Cancellation and terminal-state races, using the source regressions."""


def test_a_late_result_cannot_release_a_cancelled_job():
    """The defect this file was written for.

    Cancellation is observed at checkpoints, so a job can finish between the
    last checkpoint and the worker's return. The success path then released
    it, because it never asked again whether the user had changed their mind.
    The work here is held open until the cancellation is recorded and then
    allowed to succeed, which is exactly that window.
    """
    import threading

    from services import ingest_jobs

    cancelled = threading.Event()
    manager = ingest_jobs.JobManager()

    def work():
        # Finish only once the cancellation has landed.
        cancelled.wait(timeout=10)
        return {
            "release_decision": {"releasable": True},
            "sanitized_output": {"artifact_sha256": "a" * 64},
        }

    job = manager.submit(work, "cancel-owner")
    for _ in range(200):
        if job.state == ingest_jobs.STATE_ANONYMIZING:
            break
        import time as _time

        _time.sleep(0.01)
    manager.cancel(job.job_id, "cancel-owner")
    cancelled.set()
    job.done.wait(timeout=10)

    assert job.state == ingest_jobs.STATE_CANCELLED
    assert not (job.result or {}).get("sanitized_output")


def test_a_settled_job_is_not_overwritten():
    """Whatever a job settled as, a later writer leaves it alone."""
    from services import ingest_jobs

    manager = ingest_jobs.JobManager()
    job = manager.submit(
        lambda: {"release_decision": {"releasable": True}}, "cancel-owner"
    )
    job.done.wait(timeout=10)
    settled = job.state

    manager._finish(job, ingest_jobs.STATE_FAILED, error_code="late_writer")

    assert job.state == settled
