from typing import Any, Dict, List

from services.privacy_profiles import get_privacy_profile, load_privacy_profiles


TEXT_SAMPLE = "Patient John Doe, MRN 123456, visited on 2026-06-16."
MENTOR_QUESTIONS = [
    "Should research mode preserve generalized age/sex?",
    "Should research mode shift dates instead of deleting them?",
    (
        "Should DICOM private tags always be removed, or should we allow "
        "allowlists later?"
    ),
    "Should OCR confidence threshold differ between strict and research?",
    "Should NIfTI extensions always be removed in strict mode?",
]


def build_week5_profile_demo() -> Dict[str, Any]:
    profiles = load_privacy_profiles()

    return {
        "status": "success",
        "demo": "week5_privacy_profiles",
        "scope": (
            "Local prototype for mentor review. It shows intended strict and "
            "research behavior without printing raw PHI, full metadata dumps, "
            "or file bytes."
        ),
        "profiles_loaded": sorted(profiles),
        "profile_comparison": {
            profile_name: _build_profile_demo(profile_name)
            for profile_name in ("strict", "research")
        },
        "mentor_questions": MENTOR_QUESTIONS,
    }


def _build_profile_demo(profile_name: str) -> Dict[str, Any]:
    settings = get_privacy_profile(profile_name)

    return {
        "profile": profile_name,
        "settings": settings,
        "text_demo": _text_demo(profile_name, settings),
        "dicom_demo": _dicom_demo(settings),
        "nifti_demo": _nifti_demo(settings),
        "ocr_demo": _ocr_demo(profile_name, settings),
        "approval_notes": _approval_notes(profile_name, settings),
    }


def _text_demo(profile_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    safe_output_shape = TEXT_SAMPLE
    changed: List[str] = []

    safe_output_shape = safe_output_shape.replace("John Doe", "<REDACTED_NAME>")
    changed.append("name removed")

    safe_output_shape = safe_output_shape.replace("MRN 123456", "<REDACTED_MRN>")
    changed.append("MRN removed")

    if settings["date_strategy"] == "shift" and profile_name == "research":
        safe_output_shape = safe_output_shape.replace(
            "2026-06-16",
            "<SHIFTED_DATE>",
        )
        changed.append("date shifted")
        proposed = [
            "Deterministic study-level date shifting is implemented for research demo mode."
        ]
    else:
        safe_output_shape = safe_output_shape.replace("2026-06-16", "<REDACTED_DATE>")
        changed.append("date redacted")
        proposed = []

    return {
        "synthetic_input": "withheld",
        "safe_output_shape": safe_output_shape,
        "changed": changed,
        "implemented_now": [
            "Text service validates strict and research profile names.",
            "Direct identifiers such as MRNs are removed or replaced.",
            "Common dates are redacted by the current text anonymizer.",
        ],
        "proposed_needs_approval": proposed,
    }


def _dicom_demo(settings: Dict[str, Any]) -> Dict[str, Any]:
    changed = [
        "PatientName removed",
        "PatientID removed",
        _date_change_label(settings),
    ]
    if settings["remove_dicom_private_tags"]:
        changed.append("private tags removed")
    else:
        changed.append("private tags require allowlist approval before preservation")

    preserved: List[str] = []
    if settings["preserve_dicom_technical_metadata"]:
        preserved.extend(["Modality", "Rows", "Columns"])

    return {
        "synthetic_metadata": "direct identifiers withheld",
        "changed": changed,
        "preserved_safe_metadata": preserved or ["minimal technical metadata"],
        "implemented_now": [
            "DICOM direct PHI fields are scrubbed.",
            "Strict mode currently removes DICOM private tags.",
            "Anonymized DICOM bytes can be downloaded from /anonymize_dicom.",
        ],
        "proposed_needs_approval": [
            "Research date shifting is implemented for DICOM date fields in demo mode.",
            "Private tag allowlists need explicit mentor approval before use.",
        ],
    }


def _nifti_demo(settings: Dict[str, Any]) -> Dict[str, Any]:
    changed = ["descrip cleared", "aux_file cleared"]
    proposed = []

    if settings["remove_nifti_extensions"]:
        changed.append("extensions removed")
    else:
        changed.append("extension preservation needs safety rules")
        proposed.append("Decide whether research mode can preserve safe extensions.")

    return {
        "synthetic_header_text": "withheld",
        "changed": changed,
        "preserved_safe_metadata": ["shape", "datatype", "affine"],
        "implemented_now": [
            "Text-like NIfTI header fields are cleared.",
            "Shape, affine, datatype, and image data are preserved.",
            "Strict mode currently removes NIfTI extensions.",
        ],
        "proposed_needs_approval": proposed,
    }


def _ocr_demo(profile_name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "profile": profile_name,
        "configured_confidence_threshold": settings["ocr_confidence_threshold"],
        "implemented_now": [
            "OCR redaction supports a confidence threshold.",
            "The upload path reads this profile config for DICOM OCR redaction.",
        ],
        "proposed_needs_approval": [
            "Confirm whether strict should use a lower threshold than research."
        ],
    }


def _approval_notes(profile_name: str, settings: Dict[str, Any]) -> List[str]:
    notes = [
        "Do not preserve direct identifiers in either profile.",
        "Use this endpoint for policy review before wiring production behavior.",
    ]

    if profile_name == "research" and settings["allow_generalized_demographics"]:
        notes.append("Generalized demographics require mentor approval.")

    return notes


def _date_change_label(settings: Dict[str, Any]) -> str:
    if settings["date_strategy"] == "shift":
        return "date shifted"
    if settings["date_strategy"] == "generalize":
        return "date generalized"
    return "date redacted"

