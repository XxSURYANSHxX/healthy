"""Single-file policy, job authorization and final candidate delivery regressions.

All inputs are invented. Detector capability is simulated explicitly by the
fixture; these tests qualify mechanics, not real-model detection quality.
"""

import csv
import hashlib
import io
import json
import secrets
import threading
import time
import zipfile
from collections import Counter

import openpyxl
import pytest
import streaming_support as support
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from services import dataset_privacy, stream_routes, tabular_privacy, upload_auth
from services.ingest_jobs import STATE_READY
from services.multiformat import TextChannelSanitizer

JOBS = "/api/v1/ingest/jobs"
HEADERS = ["patient_name", "contact", "age", "lab_value", "diagnosis", "sex"]


def table():
    rows = [
        [
            f"SyntheticPerson-{i:03d}",
            "4417",
            str(20 + i % 10),
            "4417",
            ("I10", "E11.9")[(i // 2) % 2],
            "FM"[i % 2],
        ]
        for i in range(20)
    ]
    return [HEADERS, *rows]


def policy(profile="safe_harbor_plus_qi"):
    return {
        "policy_version": "tabular-privacy-v1",
        "profile": profile,
        "k": 5,
        "l": 2,
        "t": None,
        "columns": [
            {
                "index": 0,
                "header": "patient_name",
                "role": "direct_identifier",
                "entity_type": "PERSON",
            },
            {
                "index": 1,
                "header": "contact",
                "role": "direct_identifier",
                "entity_type": "PHONE_NUMBER",
            },
            {"index": 2, "header": "age", "role": "quasi_identifier", "qi_kind": "age"},
            {"index": 3, "header": "lab_value", "role": "clinical_utility"},
            {"index": 4, "header": "diagnosis", "role": "sensitive_clinical"},
            {"index": 5, "header": "sex", "role": "quasi_identifier", "qi_kind": "sex"},
        ],
    }


def encode(rows, workbook=False):
    if workbook:
        book = openpyxl.Workbook()
        for row in rows:
            book.active.append(row)
        stream = io.BytesIO()
        book.save(stream)
        book.close()
        return stream.getvalue()
    stream = io.StringIO(newline="")
    csv.writer(stream).writerows(rows)
    return stream.getvalue().encode()


@pytest.fixture
def client(tmp_path, monkeypatch, simulated_production_detectors):
    support.isolate_storage(tmp_path, monkeypatch)
    monkeypatch.setenv("BIOBLOCK_PUBLICATION_MASTER_KEY_V1", secrets.token_hex(32))
    monkeypatch.setenv("BIOBLOCK_STUDY_SALT", "synthetic-stream-tests")
    stream_routes.shutdown()
    app = FastAPI()
    app.include_router(stream_routes.router)

    @app.exception_handler(upload_auth.UploadError)
    async def error(_request, exc):
        return JSONResponse(status_code=exc.status_code, content=exc.to_public_dict())

    yield TestClient(app)
    stream_routes.shutdown()


def submit(
    client,
    payload=None,
    manifest=None,
    *,
    workbook=False,
    owner="owner-a",
    signed_policy=None,
    name=None,
):
    payload = encode(table(), workbook) if payload is None else payload
    raw = json.dumps(manifest) if manifest is not None else ""
    headers = (
        upload_auth.signed_upload_headers(
            support.wallet_key(owner),
            JOBS,
            payload,
            "strict",
            tabular_policy=raw if signed_policy is None else signed_policy,
        )
        if owner
        else {}
    )
    response = client.post(
        JOBS,
        files={"file": (name or ("table.xlsx" if workbook else "table.csv"), payload)},
        data={"profile": "strict", "tabular_policy": raw},
        headers=headers,
    )
    return response


def finished(client, submission):
    assert submission.status_code == 202, submission.text
    body = submission.json()
    headers = {"Authorization": "Bearer " + body["capability"]}
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(body["status_path"], headers=headers)
        assert response.status_code == 200, response.text
        result = response.json()
        if result["status"] in {"ready", "held", "failed", "cancelled"}:
            return result, headers
        time.sleep(0.01)
    raise AssertionError("job did not finish")


@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("workbook", [False, True])
def test_matched_csv_xlsx_written_cells_satisfy_policy(client, workbook, collision):
    declared = policy()
    original_rows = table()
    if not collision:
        for row in original_rows[1:]:
            row[3] = "4418"
    state, headers = finished(
        client,
        submit(
            client,
            payload=encode(original_rows, workbook),
            manifest=declared,
            workbook=workbook,
        ),
    )
    if workbook and collision:
        assert (
            state["status"] == "held"
            and "tabular_identifier_clinical_collision"
            in state["result"]["release_decision"]["reason_codes"]
        )
    else:
        assert state["status"] == "ready", state
    download = client.get(state["download_path"], headers=headers)
    assert download.status_code == 200
    assert (
        hashlib.sha256(download.content).hexdigest()
        == download.headers["x-bioblock-artifact-sha256"]
    )
    if state["status"] == "ready":
        assert (
            download.headers["x-bioblock-artifact-sha256"]
            == state["result"]["release_decision"]["artifact_sha256"]
        )
    else:
        assert download.headers["x-bioblock-releasable"] == "false"
    if workbook:
        book = openpyxl.load_workbook(io.BytesIO(download.content))
        rows = [
            [str(cell or "") for cell in row]
            for row in book.active.iter_rows(values_only=True)
        ]
        book.close()
    else:
        rows = list(csv.reader(io.StringIO(download.content.decode(), newline="")))
    assert rows[0] == HEADERS and len(rows) == len(table())
    for original, written in zip(original_rows[1:], rows[1:]):
        assert written[0] != original[0] and written[1] != "4417"
        assert written[3:5] == original[3:5]
        assert written[2] in {"20–29", "20-29", "*", "<AGE_1>"}
    config = tabular_privacy.policy_config(declared)
    columns = tabular_privacy.classify_columns(rows[0], rows[1:], config)
    roles = tabular_privacy.dataset_roles(columns)
    actual = dataset_privacy.verify_dataset(
        rows[1:], roles, policy=tabular_privacy.dataset_policy(config)
    )
    assert actual.satisfied and actual.k >= 5 and actual.l >= 2
    groups = Counter((row[2], row[5]) for row in rows[1:])
    assert min(groups.values()) >= 5


def test_policy_signature_binds_exact_text_and_rejects_replay(client):
    declared = policy()
    refused = submit(
        client,
        manifest=declared,
        signed_policy=json.dumps(policy("identifier_removal")),
    )
    assert refused.status_code == 401
    payload, raw = encode(table()), json.dumps(declared)
    headers = upload_auth.signed_upload_headers(
        support.wallet_key("owner-a"), JOBS, payload, "strict", tabular_policy=raw
    )
    arguments = {
        "files": {"file": ("table.csv", payload)},
        "data": {"tabular_policy": raw},
        "headers": headers,
    }
    assert client.post(JOBS, **arguments).status_code == 202
    assert client.post(JOBS, **arguments).status_code == 401


def test_partial_policy_does_not_claim_full_application(client):
    declared = policy()
    declared["columns"] = declared["columns"][:1]
    state, headers = finished(client, submit(client, manifest=declared))
    metadata = state["result"]["summary"]["tabular_privacy"]
    assert metadata["declared_column_count"] < len(HEADERS)
    assert metadata["schema_coverage"] == "partial"
    assert metadata["full_declared_schema_applied"] is False
    assert metadata["inferred_column_count"] == len(HEADERS) - 1


def test_default_identifier_removal_is_not_selected_by_environment(client, monkeypatch):
    monkeypatch.setenv("BIOBLOCK_CSV_PRIVACY_PROFILE", "safe_harbor_plus_qi")
    state, _ = finished(
        client, submit(client, payload=b"age,diagnosis\n23,I10\n24,E11.9\n")
    )
    assert (
        state["result"]["summary"]["tabular_privacy"]["profile"] == "identifier_removal"
    )


def test_held_candidate_is_sealed_review_only_and_never_original_fallback(
    client, monkeypatch
):
    from services import detector_capability

    monkeypatch.setattr(
        detector_capability, "release_blocker", lambda: "non_production_detectors"
    )
    state, headers = finished(client, submit(client, manifest=policy()))
    assert state["status"] == "held", state
    current = stream_routes.pipeline()
    candidate = current._candidates[state["job_id"]]
    assert (
        candidate.path.name == "candidate.sealed"
        and not (candidate.directory / "input").exists()
    )
    assert not (candidate.directory / "output.csv").exists()
    assert (
        client.get(f"{JOBS}/{state['job_id']}/artifact", headers=headers).status_code
        == 409
    )
    response = client.get(state["download_path"], headers=headers)
    assert (
        response.status_code == 200
        and response.headers["x-bioblock-releasable"] == "false"
    )
    assert response.content != encode(table())
    rows = list(csv.reader(io.StringIO(response.content.decode())))
    assert all(row[2] not in {str(n) for n in range(20, 30)} for row in rows[1:])
    assert not list(candidate.directory.glob(".open-*"))
    candidate.path.write_bytes(b"invalid candidate")
    assert client.get(state["download_path"], headers=headers).status_code == 422
    candidate.path.unlink()
    assert client.get(state["download_path"], headers=headers).status_code == 422


def test_polling_capability_cannot_cross_jobs_cancel_or_restore(client):
    one = submit(client, manifest=policy()).json()
    two = submit(client, manifest=policy(), owner="owner-b").json()
    bearer = {"Authorization": "Bearer " + one["capability"]}
    assert client.get(two["status_path"], headers=bearer).status_code == 401
    assert client.delete(one["status_path"], headers=bearer).status_code == 401
    assert (
        client.post(one["status_path"] + "/capability", headers=bearer).status_code
        == 401
    )
    restore_path = one["status_path"] + "/capability"
    signed = upload_auth.signed_ingestion_headers(
        support.wallet_key("owner-a"), "POST", restore_path
    )
    restored = client.post(restore_path, headers=signed)
    assert restored.status_code == 200
    assert client.get(one["status_path"], headers=bearer).status_code == 401
    replacement = {"Authorization": "Bearer " + restored.json()["capability"]}
    assert client.get(one["status_path"], headers=replacement).status_code == 200


def test_unknown_workbook_part_has_no_downloadable_original(client):
    source, out = encode(table(), True), io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as original, zipfile.ZipFile(
        out, "w"
    ) as written:
        for name in original.namelist():
            written.writestr(name, original.read(name))
        written.writestr("xl/unknown-private.xml", "SyntheticPerson-999")
    state, headers = finished(
        client, submit(client, payload=out.getvalue(), workbook=True)
    )
    assert state["status"] != "ready", state
    response = client.get(f"{JOBS}/{state['job_id']}/artifact", headers=headers)
    assert response.status_code == 409
    assert "SyntheticPerson" not in json.dumps(state)


def test_cancel_during_transform_discards_output_and_spill(
    client, monkeypatch, tmp_path
):
    from services import csv_stream, request_deadline

    started, proceed = threading.Event(), threading.Event()
    original = csv_stream.anonymize_csv_stream

    def blocked(*args, **kwargs):
        started.set()
        assert proceed.wait(10)
        request_deadline.check("stream.cancel.test")
        return original(*args, **kwargs)

    monkeypatch.setattr(csv_stream, "anonymize_csv_stream", blocked)
    submission = submit(client, manifest=policy())
    assert started.wait(10)
    body = submission.json()
    headers = upload_auth.signed_ingestion_headers(
        support.wallet_key("owner-a"), "DELETE", body["status_path"]
    )
    try:
        assert client.delete(body["status_path"], headers=headers).status_code == 202
    finally:
        proceed.set()
    state, bearer = finished(client, submission)
    assert state["status"] == "cancelled"
    assert (
        client.get(body["status_path"] + "/artifact", headers=bearer).status_code == 409
    )
    assert not list((tmp_path / "artifacts" / "stream-jobs").iterdir())


def test_changed_or_missing_released_candidate_never_downloads_original(client):
    state, headers = finished(client, submit(client, manifest=policy()))
    assert state["status"] == "ready"
    candidate = stream_routes.pipeline()._candidates[state["job_id"]]
    candidate.path.write_bytes(b"tampered output")
    assert client.get(state["download_path"], headers=headers).status_code == 422
    candidate.path.unlink()
    assert client.get(state["download_path"], headers=headers).status_code == 422


def test_unstarted_review_download_releases_its_plaintext_and_lease(
    client, monkeypatch
):
    from services import detector_capability

    monkeypatch.setattr(
        detector_capability, "release_blocker", lambda: "non_production_detector_mode"
    )
    state, headers = finished(client, submit(client, manifest=policy()))
    current = stream_routes.pipeline()
    job = current.manager.get(state["job_id"], support.wallet_address("owner-a"))
    candidate, chunks = current.open_candidate(job, review=True)
    assert candidate.leases == 1 and list(candidate.directory.glob(".open-*"))
    current.discard(job)
    assert candidate.directory.exists()
    chunks.close()  # The body was never iterated, for example after a disconnect.
    chunks.close()
    assert candidate.leases == 0 and not candidate.directory.exists()


def test_signed_schema_mismatch_fails_without_a_downloadable_candidate(client):
    declared = policy()
    declared["columns"][0]["header"] = "different_schema"
    state, headers = finished(client, submit(client, manifest=declared))
    assert state["status"] == "failed" and state["status_code"] in {400, 422}
    assert (
        client.get(f"{JOBS}/{state['job_id']}/artifact", headers=headers).status_code
        == 409
    )
    assert state["job_id"] not in stream_routes.pipeline()._candidates


def test_text_job_runs_multiple_windows_and_downloads_written_text(client, monkeypatch):
    monkeypatch.setenv("BIOBLOCK_STREAM_TEXT_WINDOW_CHARS", "4096")
    line = "Assessment: hypertension controlled; creatinine 88 umol/L. Repeat renal function in three months.\n"
    payload = (line * 100).encode()
    state, headers = finished(client, submit(client, payload=payload, name="note.txt"))
    assert state["status"] in {"ready", "held"}, state
    assert state["result"]["summary"]["streaming"]["windows"] > 1
    response = client.get(state["download_path"], headers=headers)
    assert response.status_code == 200 and response.content == payload
    assert (
        hashlib.sha256(response.content).hexdigest()
        == response.headers["x-bioblock-artifact-sha256"]
    )
