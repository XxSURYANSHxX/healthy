"""The release audit: one post-release check over every released format.

Outputs of the real sanitizers pass. Artifacts with an identifier planted in
one channel - text, metadata, private data, attachments, links, burned-in
pixels, quasi-identifiers - fail with a category and a reason code, and the CI
report names only the fixture, the category and the code. Every value is
invented.
"""

import io
import os
import re
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services import release_audit as audit  # noqa: E402

GIVEN, FAMILY = "Zelquin", "Vantreske"
NAME = f"{GIVEN} {FAMILY}"
RECORD = "48213377"


class NoTextEngine:
    ocr_engine_status = "available"
    #: A scripted reader, not a rendering engine: it cannot read a
    #: picture of a control string, so it states what the gate
    #: would have found.
    ocr_control_probe = "passed"

    def detect_text_boxes(self, image):
        return []


class UnavailableEngine:
    ocr_engine_status = "unavailable"

    def detect_text_boxes(self, image):
        raise AssertionError("an unavailable engine is never called")


def _codes(report):
    return {(finding.category, finding.reason_code) for finding in report.findings}


def _reasons(report):
    return {finding.reason_code for finding in report.findings}


# -- text and tables --------------------------------------------------------------------------


def test_an_anonymized_note_passes_and_a_note_with_identifiers_fails():
    from services.text_anonymization import anonymize_clinical_text_with_replacements

    raw = f"Patient {NAME} reviewed. Call 555-0142 about MRN {RECORD}."
    result, applied = anonymize_clinical_text_with_replacements(raw)

    assert result["residual_phi_categories"] == {}
    assert audit.audit_released_artifact(
        result["anonymized_text"].encode(),
        audit.FORMAT_TEXT,
        text_scanner=audit.artifact_scanner({item[3] for item in applied}),
    ).passed
    failed = audit.audit_released_artifact(raw.encode(), audit.FORMAT_TEXT)
    assert audit.REASON_STRUCTURED in _reasons(failed)


def test_the_audit_and_the_release_gate_agree_on_a_surviving_record_number():
    from services.text_anonymization import (
        anonymize_clinical_text_with_replacements,
        residual_phi_categories,
    )

    # The first pass now replaces an unlabelled record number itself, and the
    # audit of what it wrote finds nothing.
    result, applied = anonymize_clinical_text_with_replacements(
        f"Reviewed record {RECORD} today."
    )
    assert (
        RECORD not in result["anonymized_text"]
        and not result["residual_phi_categories"]
    )
    written = audit.audit_released_artifact(
        result["anonymized_text"].encode(),
        audit.FORMAT_TEXT,
        text_scanner=audit.artifact_scanner({item[3] for item in applied}),
    )
    assert not written.findings

    # Where one does reach the gate, the residual check and the audit flag
    # the same category.
    raw = f"Reviewed record {RECORD} today."
    held = set(residual_phi_categories(raw))
    report = audit.audit_released_artifact(raw.encode(), audit.FORMAT_TEXT)
    assert held and held <= {finding.category for finding in report.findings}


def test_a_model_scanner_can_be_added_for_real_model_runs():
    def scanner(text):
        return ["PERSON"] if FAMILY in text else []

    report = audit.audit_released_artifact(
        f"Seen by {NAME}.".encode(), audit.FORMAT_TEXT, text_scanner=scanner
    )

    assert ("PERSON", audit.REASON_MODEL) in _codes(report)


def test_a_released_csv_passes_and_a_linkable_table_fails(monkeypatch):
    from services import tabular_privacy
    from services.csv_sanitizer import anonymize_csv_bytes
    from services.csv_stream import PROFILE_ENV_VAR

    monkeypatch.setenv(PROFILE_ENV_VAR, "risk_based")

    result = anonymize_csv_bytes(
        f"patient_name,diagnosis,systolic_bp\n{NAME},I10,128\nMarbrint Morrowgale,E11.9,141\n".encode()
    )
    released = result.output_bytes
    linkable = "age,sex,zip,diagnosis\n" + "".join(
        f"{30 + n},F,<LOCATION_{n}>,{('I10', 'E11.9')[n % 2]}\n" for n in range(8)
    )

    assert audit.audit_released_artifact(
        released,
        audit.FORMAT_CSV,
        text_scanner=audit.artifact_scanner(result.generated_tokens),
    ).passed
    report = audit.audit_released_artifact(
        linkable.encode(),
        audit.FORMAT_CSV,
        provenance={
            "tabular_privacy": tabular_privacy.PolicyConfig(
                profile=tabular_privacy.PROFILE_SAFE_HARBOR_PLUS_QI
            ).public_metadata()
        },
    )
    assert ("QUASI_IDENTIFIER", audit.REASON_CSV_DATASET) in _codes(report)


def test_identifier_removal_makes_no_table_shape_claim_in_the_audit(monkeypatch):
    """A generic audit never imports mutable process QI settings."""
    from services.csv_stream import PROFILE_ENV_VAR

    monkeypatch.setenv(PROFILE_ENV_VAR, "risk_based")
    linkable = "age,sex,zip,diagnosis\n" + "".join(
        f"{30 + n},F,<LOCATION_{n}>,{('I10', 'E11.9')[n % 2]}\n" for n in range(8)
    )
    report = audit.audit_released_artifact(linkable.encode(), audit.FORMAT_CSV)
    assert ("QUASI_IDENTIFIER", audit.REASON_CSV_DATASET) not in _codes(report)


def test_csv_cells_are_read_one_at_a_time_under_their_column_rules():
    from services.csv_sanitizer import anonymize_csv_bytes

    rows = ["patient_name,age,admission_date,sex,zip,diagnosis,note"] + [
        f"{NAME},{20 + n},{2015 + n % 5},{'FM'[n % 2]},0213{n % 3},"
        f"{('I10', 'E11.9')[n % 2]},stable on review"
        for n in range(30)
    ]
    result = anonymize_csv_bytes(
        ("\n".join(rows) + "\n").encode(), residual_scanner=lambda text, provenance: []
    )
    released = result.output_bytes
    seen = []

    def dates_everywhere(text):
        seen.append(text)
        return ["DATE_TIME"] if re.search(r"\d", text) else []

    dates_everywhere.emitted_tokens = result.generated_tokens
    report = audit.audit_released_artifact(
        released, audit.FORMAT_CSV, text_scanner=dates_everywhere
    )

    assert report.passed, report.findings
    assert "stable on review" in seen
    assert not any(re.search(r"\d", text) for text in seen)
    # Column names built from the classifier's words are not free text.
    assert not {"age", "admission_date", "zip", "diagnosis"} & set(seen)

    leaky = (
        f"note,mrn,diagnosis\nSeen by {NAME},{RECORD},I10\nstable,<RECORD_ID_1>,E11.9\n"
    )
    leaked = audit.audit_released_artifact(
        leaky.encode(),
        audit.FORMAT_CSV,
        text_scanner=lambda text: ["PERSON"] if FAMILY in text else [],
    )
    assert ("PERSON", audit.REASON_MODEL) in _codes(leaked)
    assert ("IDENTIFYING_TEXT", audit.REASON_CSV_IDENTIFIER_COLUMN) in _codes(leaked)

    named_header = audit.audit_released_artifact(
        f"results for {FAMILY},diagnosis\nstable,I10\nreviewed,E11.9\n".encode(),
        audit.FORMAT_CSV,
        text_scanner=lambda text: ["PERSON"] if FAMILY in text else [],
    )
    assert ("PERSON", audit.REASON_MODEL) in _codes(named_header)


def test_a_released_workbook_passes_and_properties_authors_and_links_fail():
    import openpyxl
    from openpyxl.comments import Comment
    from services.workbook_sanitizer import anonymize_workbook_bytes

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["A2"] = "note", "Stable on inhaler"
    sheet["A2"].comment = Comment("Reviewed", NAME)
    sheet["A3"] = "record"
    sheet["A3"].hyperlink = "https://example.invalid/chart"
    workbook.properties.creator = NAME
    buffer = io.BytesIO()
    workbook.save(buffer)
    raw = buffer.getvalue()

    result = anonymize_workbook_bytes(raw)
    released = result.output_bytes

    assert audit.audit_released_artifact(
        released,
        audit.FORMAT_WORKBOOK,
        text_scanner=audit.artifact_scanner(result.generated_tokens),
    ).passed
    raw_report = audit.audit_released_artifact(raw, audit.FORMAT_WORKBOOK)
    assert {
        audit.REASON_WORKBOOK_PROPERTIES,
        audit.REASON_WORKBOOK_AUTHOR,
        audit.REASON_WORKBOOK_LINK,
    } <= _reasons(raw_report)
    locations = raw_report.public_dict()["finding_locations"]
    assert any(item["location"] == "sheet:0:cell:A3" for item in locations)
    assert all(NAME not in item["location"] for item in locations)


def test_csv_audit_finding_names_every_physical_cell_sharing_the_text():
    content = (
        "id_note,note\n1,Stable on inhaler\n2,Jordan Smithers was seen\n"
        "3,Stable on inhaler\n4,Jordan Smithers was seen\n5,Jordan Smithers was seen\n"
    ).encode()

    def scanner(value):
        return ["PERSON"] if "Smithers" in value else []

    setattr(scanner, audit.READS_EXACT_TEXT, True)
    report = audit.audit_released_artifact(
        content, audit.FORMAT_CSV, text_scanner=scanner
    )

    locations = {
        item["location"]
        for item in report.public_dict()["finding_locations"]
        if item["category"] == "PERSON"
    }
    # Row 0 is the header; every duplicate row is a separate occurrence.
    assert locations == {"row:2:column:1", "row:4:column:1", "row:5:column:1"}
    assert all("Smithers" not in location for location in locations)
    assert len(report.findings) == 1


def _clean_text_scanner(_value):
    return []


setattr(_clean_text_scanner, audit.READS_EXACT_TEXT, True)
#: Tokens the sanitizer emitted for these invented cells; any other token-shaped
#: text would be treated as an original identifier by the audit.
setattr(
    _clean_text_scanner,
    "emitted_tokens",
    frozenset({"<PERSON_485>", "<PHONE_241>", "<PERSON_486>", "<PERSON_1>"}),
)


def _note_workbook(note):
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"], sheet["A2"] = "note", note
    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def test_a_label_already_satisfied_by_a_placeholder_does_not_announce_a_later_result():
    twin = _note_workbook(
        "Asian MRN <PERSON_485>; extension <PHONE_241>; laboratory result 4417; <PERSON_486>."
    )
    report = audit.audit_released_artifact(
        twin, audit.FORMAT_WORKBOOK, text_scanner=_clean_text_scanner
    )
    assert audit.REASON_STRUCTURED not in _reasons(report), report.public_dict()


@pytest.mark.parametrize(
    "note",
    [
        "Asian MRN 4417; extension <PHONE_241>.",
        "Seen; MRN: 4417; extension <PHONE_241>.",
        "Asian MRN <PERSON_485>; MRN 4417.",
        "MRN <PERSON_1> 4417",
    ],
)
def test_a_real_label_value_pair_is_still_found_beside_placeholders(note):
    report = audit.audit_released_artifact(
        _note_workbook(note), audit.FORMAT_WORKBOOK, text_scanner=_clean_text_scanner
    )
    assert audit.REASON_STRUCTURED in _reasons(report)
    assert any(
        item["location"] == "sheet:0:cell:A2"
        for item in report.public_dict()["finding_locations"]
    )


def _band_workbook(note="stable"):
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["age", "sex", "note"])
    for _ in range(12):
        sheet.append(["30–39", "F", note])
    stream = io.BytesIO()
    workbook.save(stream)
    return stream.getvalue()


def _dash_means_ssn(text):
    # A stand-in for the model's misreading of an age band as part of an SSN.
    return ["US_SSN"] if "–" in text else []


def test_workbook_audit_does_not_let_a_model_read_a_quasi_identifier_age_band_as_an_ssn():
    report = audit.audit_released_artifact(
        _band_workbook(), audit.FORMAT_WORKBOOK, text_scanner=_dash_means_ssn
    )

    assert audit.REASON_MODEL not in _reasons(report), report.public_dict()


def test_workbook_audit_still_asks_the_model_about_free_text_beside_the_band():
    report = audit.audit_released_artifact(
        _band_workbook("seen 30–39 days ago"),
        audit.FORMAT_WORKBOOK,
        text_scanner=_dash_means_ssn,
    )

    assert audit.REASON_MODEL in _reasons(report)
    assert {
        item["location"][:14] for item in report.public_dict()["finding_locations"]
    } == {"sheet:0:cell:C"}


def test_workbook_audit_reads_retained_vml_text_independently_of_cells():
    import openpyxl
    from openpyxl.comments import Comment

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet["A1"] = "note"
    sheet["A2"] = "Stable"
    sheet["A2"].comment = Comment("Reviewed", "anonymized")
    stream = io.BytesIO()
    workbook.save(stream)
    changed = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as source:
        with zipfile.ZipFile(changed, "w") as output:
            for info in source.infolist():
                payload = source.read(info.filename)
                if info.filename.endswith(".vml"):
                    payload = payload.replace(
                        b'<div style="text-align:left"/>',
                        b'<div style="text-align:left">Jordan Fictional</div>',
                        1,
                    )
                output.writestr(info, payload)

    def scanner(value):
        return ["PERSON"] if "Jordan Fictional" in value else []

    result = audit.audit_released_artifact(
        changed.getvalue(), audit.FORMAT_WORKBOOK, text_scanner=scanner
    )
    assert ("PERSON", audit.REASON_MODEL) in _codes(result)
    assert any(
        item["location"].startswith("package:xl/drawings/")
        for item in result.public_dict()["finding_locations"]
    )


# -- documents and images ----------------------------------------------------------------------
