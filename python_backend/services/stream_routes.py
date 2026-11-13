"""The minimum single-file routes for bounded CSV/TXT and shared XLSX policy."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from services import request_deadline, resource_policy, tabular_privacy, upload_auth
from services.ingest_jobs import SCOPE_JOB_READ, SCOPE_REVIEW_DOWNLOAD, TERMINAL_STATES
from services.ingestion import IngestionError, validate_privacy_profile
from services.multiformat import signature_matches
from services.sanitized_artifacts import output_directory
from services.stream_ingestion import (
    CHUNK,
    EXTENSIONS,
    MEDIA,
    Pipeline,
    private_work_directory,
)
from services.upload_auth import UploadError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

router = APIRouter()
_pipeline = None
_pipeline_lock = threading.Lock()


def pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = Pipeline()
        return _pipeline


def shutdown():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            _pipeline.shutdown()
            _pipeline = None


def tabular_manifest(raw, modality):
    if not raw:
        return None
    if modality not in {"csv", "workbook"}:
        raise HTTPException(422, "tabular_policy_not_applicable")
    try:
        if len(raw.encode("utf-8")) > 64 * 1024:
            raise ValueError()
        manifest = json.loads(raw)
        if (
            not isinstance(manifest, dict)
            or manifest.get("policy_version") != tabular_privacy.POLICY_VERSION
            or not manifest.get("profile")
        ):
            raise ValueError()
        tabular_privacy.policy_config(manifest, allow_environment_manifest=False)
    except (ValueError, TypeError, tabular_privacy.ManifestError):
        raise HTTPException(422, "tabular_policy_invalid") from None
    return manifest


def modality_for(filename):
    suffix = Path(filename or "").suffix.casefold()
    if suffix == ".xls":
        raise HTTPException(415, "legacy_xls_not_supported")
    modality = {
        ".csv": "csv",
        ".txt": "text",
        ".xlsx": "workbook",
        ".xlsm": "workbook",
    }.get(suffix)
    if modality is None:
        raise HTTPException(415, "unsupported_stream_format")
    return modality


def capacity_policy():
    return resource_policy.for_snapshot(
        resource_policy.detect(output_directory(), gpu=False)
    )


async def submit_upload(request, file, profile, raw_policy="", study_ref=""):
    try:
        profile = validate_privacy_profile(profile)
    except IngestionError as exc:
        raise HTTPException(
            exc.status_code,
            "Invalid privacy profile. Supported profiles: safe_harbor_v1, strict, research",
        ) from None
    modality = modality_for(file.filename)
    manifest = tabular_manifest(raw_policy, modality)
    policy = await run_in_threadpool(capacity_policy)
    capacity = policy.capacity(modality)
    size = file.size
    if not isinstance(size, int) or size <= 0:
        raise HTTPException(400, "Uploaded file is empty")
    current = pipeline()
    reservation = await run_in_threadpool(current.reserve, modality, size, policy)
    directory = None
    submitted = False
    try:
        directory = private_work_directory()
        digest, total, head = hashlib.sha256(), 0, b""
        await file.seek(0)
        with open(directory / "input", "wb") as output:
            while block := await file.read(CHUNK):
                request_deadline.check("upload.spool")
                total += len(block)
                if total > capacity.effective_max_bytes or total > size:
                    raise HTTPException(413, "file_exceeds_capacity")
                if len(head) < 4096:
                    head += block[: 4096 - len(head)]
                digest.update(block)
                await run_in_threadpool(output.write, block)
        if total != size or not signature_matches(modality, head):
            raise HTTPException(422, "malformed_file")
        owner = await run_in_threadpool(
            upload_auth.verify_upload_owner,
            request,
            digest.hexdigest(),
            profile,
            study_ref,
            (
                hashlib.sha256(raw_policy.encode("utf-8")).hexdigest()
                if raw_policy
                else ""
            ),
        )

        def work(job):
            try:
                return current.work(job, directory, modality, profile, manifest)
            finally:
                current.release_reservation(reservation)

        def discard_unstarted():
            try:
                shutil.rmtree(directory)
            finally:
                current.release_reservation(reservation)

        job = current.manager.submit(
            work, owner, bind_job=True, discard_unstarted=discard_unstarted
        )
        submitted = True
        capability = current.manager.issue_capability(job)
        body = current.manager.status(job)
        body.update(
            {
                "capability": capability,
                "status_path": f"/api/v1/ingest/jobs/{job.job_id}",
            }
        )
        return job, body
    finally:
        await file.close()
        if not submitted:
            if directory is not None:
                shutil.rmtree(directory)
            current.release_reservation(reservation)


@router.post("/api/v1/ingest/jobs", status_code=202)
async def submit_job(
    request: Request,
    file: UploadFile = File(...),
    profile: str = Form("strict"),
    tabular_policy: str = Form(""),
    study_ref: str = Form(""),
):
    _, body = await submit_upload(request, file, profile, tabular_policy, study_ref)
    return body


async def authorized_job(request, job_id, scope=SCOPE_JOB_READ):
    manager = pipeline().manager
    authorization = request.headers.get("authorization", "")
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.casefold() != "bearer" or not token:
            raise UploadError(
                401, "job_capability_invalid", "The job credential is invalid."
            )
        return manager.authorize_capability(job_id, token, scope)
    owner = (
        await upload_auth.authenticated_ingestion_owner(request)
        if upload_auth.request_is_signed(request)
        else None
    )
    return manager.get(job_id, owner)


@router.get("/api/v1/ingest/jobs/{job_id}")
async def status(job_id: str, request: Request):
    job = await authorized_job(request, job_id)
    body = pipeline().manager.status(job)
    if job.state in {"ready", "held"} and pipeline().has_candidate(job_id):
        body["download_path"] = f"/api/v1/ingest/jobs/{job_id}/" + (
            "artifact" if job.state == "ready" else "review-artifact"
        )
    return body


@router.post("/api/v1/ingest/jobs/{job_id}/capability")
async def restore(job_id: str, request: Request):
    owner = await upload_auth.authenticated_ingestion_owner(request)
    manager = pipeline().manager
    return {"capability": manager.issue_capability(manager.get(job_id, owner))}


@router.delete("/api/v1/ingest/jobs/{job_id}", status_code=202)
async def cancel(job_id: str, request: Request):
    # A polling bearer never grants cancellation, including an unsigned job's bearer.
    owner = await upload_auth.authenticated_ingestion_owner(request)
    manager = pipeline().manager
    return manager.status(manager.cancel(job_id, owner))


async def download(job_id, request, review):
    job = await authorized_job(
        request, job_id, SCOPE_REVIEW_DOWNLOAD if review else SCOPE_JOB_READ
    )
    candidate, chunks = await run_in_threadpool(
        pipeline().open_candidate, job, review=review
    )
    return StreamingResponse(
        chunks,
        media_type=MEDIA[candidate.modality],
        background=BackgroundTask(chunks.close),
        headers={
            "Content-Length": str(candidate.size),
            "Content-Disposition": 'attachment; filename="'
            + ("review-only" if review else "sanitized")
            + EXTENSIONS[candidate.modality]
            + '"',
            "X-BioBlock-Artifact-SHA256": candidate.sha256,
            "X-BioBlock-Releasable": "false" if review else "true",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/api/v1/ingest/jobs/{job_id}/artifact")
async def artifact(job_id: str, request: Request):
    return await download(job_id, request, False)


@router.get("/api/v1/ingest/jobs/{job_id}/review-artifact")
async def review_artifact(job_id: str, request: Request):
    return await download(job_id, request, True)


@router.get("/api/v1/ingest/capabilities")
async def capabilities():
    policy = await run_in_threadpool(capacity_policy)
    return {
        "schema_version": resource_policy.SCHEMA_VERSION,
        "anonymization_streaming": ["csv", "text"],
        "formats": [
            policy._format_entry(policy.capacity(name))
            for name in ("csv", "text", "workbook")
        ],
    }
