import io
import os
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse

from services.dicom_anonymization import DicomAnonymizationError, anonymize_dicom_file_bytes
from services.ingestion import (
    HEADER_READ_LIMIT,
    TEXT_READ_LIMIT_BYTES,
    IngestionError,
    detect_modality,
    route_for_ingestion,
)
from services.privacy_profile_demo import build_week5_profile_demo


os.environ.setdefault("BIOBLOCK_STUDY_SALT", "week5-demo-local-salt")
PrivacyProfile = Literal["strict", "research"]

app = FastAPI(
    title="BioBlock Week 5 Upload Privacy Profile Demo",
    description=(
        "Upload a file, select strict or research, and run the real BioBlock "
        "anonymization services in a local demo wrapper."
    ),
    version="0.2.0",
)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "message": "BioBlock Week 5 Upload Privacy Profile Demo",
        "docs": "/docs",
        "upload_endpoint": "/api/v1/ingest",
        "dicom_download_endpoint": "/anonymize_dicom",
        "policy_summary_endpoint": "/api/v1/privacy-profile-demo",
    }


@app.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    return """
    <!doctype html>
    <html>
      <head>
        <title>BioBlock Week 5 Upload Demo</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 0; background: #f6f8fb; color: #172033; }
          main { max-width: 900px; margin: 0 auto; padding: 32px; }
          .box { background: white; border: 1px solid #d8e0ea; border-radius: 8px; padding: 20px; margin-top: 18px; }
          code { display: block; padding: 12px; border-radius: 6px; background: #101827; color: #e6edf7; }
          a { color: #1f5fbf; font-weight: 700; }
        </style>
      </head>
      <body>
        <main>
          <h1>BioBlock Week 5 Upload Privacy Profile Demo</h1>
          <p>This is the upload-based FastAPI demo. Use Swagger to upload a file and choose strict or research.</p>
          <section class="box">
            <h2>Use this for mentor demo</h2>
            <p><a href="/docs">Open Swagger /docs</a></p>
            <code>POST /api/v1/ingest</code>
            <p>Upload text, DICOM, or NIfTI and set profile to strict or research.</p>
            <code>POST /anonymize_dicom</code>
            <p>Upload a DICOM and download the anonymized .dcm output for strict or research.</p>
          </section>
        </main>
      </body>
    </html>
    """


@app.post("/api/v1/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    profile: PrivacyProfile = Form("strict"),
) -> Dict[str, Any]:
    """
    Upload a biomedical file and run anonymization using the selected profile.

    Supported demo inputs: .txt, .dcm/.dicom, .nii/.nii.gz, .csv, .svs/.tif/.tiff.
    The JSON response is safe: it does not include file bytes or raw DICOM pixels.
    """
    try:
        header = await file.read(HEADER_READ_LIMIT)
        if not header:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        text_content: Optional[bytes] = None
        file_content: Optional[bytes] = None

        try:
            modality = detect_modality(file.filename, file.content_type, header)
        except IngestionError:
            modality = None

        if modality == "text":
            await file.seek(0)
            text_content = await file.read(TEXT_READ_LIMIT_BYTES + 1)
            if len(text_content) > TEXT_READ_LIMIT_BYTES:
                raise HTTPException(status_code=413, detail="Text upload is too large")
        elif modality in {"dicom", "nifti", "wsi"}:
            await file.seek(0)
            file_content = await file.read()

        return route_for_ingestion(
            filename=file.filename,
            content_type=file.content_type,
            header=header,
            profile=profile,
            text_content=text_content,
            file_content=file_content,
        )
    except IngestionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@app.post("/anonymize_dicom")
async def anonymize_dicom(
    file: UploadFile = File(...),
    profile: PrivacyProfile = Form("strict"),
):
    """
    Upload a DICOM and download an anonymized .dcm using strict or research.
    """
    try:
        contents = await file.read()
        result = anonymize_dicom_file_bytes(contents, profile=profile)
    except DicomAnonymizationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    safe_name = (file.filename or "dicom.dcm").strip().replace("\\", "/").rsplit("/", 1)[-1]
    stem = safe_name.rsplit(".", 1)[0] if "." in safe_name else safe_name
    metadata_summary = result["metadata_summary"]

    return StreamingResponse(
        io.BytesIO(result["anonymized_dicom_bytes"]),
        media_type="application/dicom",
        headers={
            "Content-Disposition": f'attachment; filename="{profile}_anonymized_{stem}.dcm"',
            "X-BioBlock-Profile": profile,
            "X-BioBlock-Anonymization-Status": result["anonymization_status"],
            "X-BioBlock-Fields-Scrubbed": str(metadata_summary["fields_scrubbed"]),
            "X-BioBlock-Private-Tags-Removed": str(metadata_summary["private_tags_removed"]),
            "X-BioBlock-Pixel-Data-Preserved": str(metadata_summary["pixel_data_preserved"]).lower(),
        },
    )


@app.get("/api/v1/privacy-profile-demo")
def privacy_profile_demo() -> Dict[str, Any]:
    return build_week5_profile_demo()
