import json
import os
import sys

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app  # noqa: E402


client = TestClient(app)


def test_week5_privacy_profile_demo_endpoint_returns_safe_comparison():
    response = client.get("/api/v1/privacy-profile-demo")

    assert response.status_code == 200
    body = response.json()
    response_text = json.dumps(body)

    assert body["status"] == "success"
    assert body["demo"] == "week5_privacy_profiles"
    assert set(body["profiles_loaded"]) >= {"strict", "research"}

    strict = body["profile_comparison"]["strict"]
    research = body["profile_comparison"]["research"]

    assert strict["text_demo"]["safe_output_shape"].endswith("<REDACTED_DATE>.")
    assert "<SHIFTED_DATE>" in research["text_demo"]["safe_output_shape"]
    assert "private tags removed" in strict["dicom_demo"]["changed"]
    assert "Modality" in research["dicom_demo"]["preserved_safe_metadata"]
    assert "shape" in research["nifti_demo"]["preserved_safe_metadata"]

    assert "John Doe" not in response_text
    assert "123456" not in response_text
    assert "2026-06-16" not in response_text
    assert "file_bytes" not in response_text
    assert "PixelData" not in response_text

