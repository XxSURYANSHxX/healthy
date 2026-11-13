"""Phase 13: .xlsx workbook anonymization.

Every workbook is built in memory with openpyxl and invented values. Text
channels must come back anonymized, numbers and formulas unchanged, and
anything that cannot be rewritten safely must hold the workbook.
"""

import io
import logging
import os
import sys
import zipfile
from datetime import datetime
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "workbook-sanitizer-test-salt")

openpyxl = pytest.importorskip("openpyxl")
from openpyxl.comments import Comment  # noqa: E402
from openpyxl.packaging.custom import StringProperty  # noqa: E402
from openpyxl.workbook.defined_name import DefinedName  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402
from services import workbook_sanitizer as wbs  # noqa: E402
from services.multiformat import MultiformatError  # noqa: E402

PHONE = "555-0142"
NAME = "Jordan Fictional"


def test_model_review_finding_keeps_workbook_cell_coordinates(monkeypatch):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Data"
    sheet.append(["note"])
    sheet.append(["Invented narrative"])
    buffer = io.BytesIO()
    book.save(buffer)
    book.close()

    def reviewed(_self, text):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": (
                    ["ambiguous_model_edit"] if text == "Invented narrative" else []
                ),
            },
            [],
        )

    monkeypatch.setattr(wbs.TextChannelSanitizer, "_anonymize", reviewed)
    monkeypatch.setattr(
        wbs,
        "verify_extracted_text",
        lambda *_args, **_kwargs: SimpleNamespace(passed=True, blocking_categories={}),
    )
    result = wbs.anonymize_workbook_bytes(buffer.getvalue())
    findings = [
        item
        for item in result.summary["review_findings"]
        if item["reason_code"] == "ambiguous_model_edit"
    ]
    assert any(
        item["surface"] == "workbook_cell"
        and item["sheet_index"] == 0
        and item["row"] == 2
        and item["column_index"] == 0
        for item in findings
    )
    assert "Invented narrative" not in repr(findings)


@pytest.fixture(autouse=True)
def _legacy_models(monkeypatch):
    monkeypatch.setenv("PHI_MODEL_MODE", "legacy_test")


def _book(
    sheet_title="results",
    formula="=B3*2",
    extra_cells=None,
    defined_formula=None,
    defined_name="contact_value",
    hidden=True,
    date_cell=True,
    image=False,
):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = sheet_title
    sheet["A1"] = "contact"
    sheet["B1"] = f"Call {PHONE} for results"
    sheet["B2"] = "see portal"
    sheet["B2"].hyperlink = f"https://portal.example.invalid/{PHONE}"
    sheet["A3"] = "sodium"
    sheet["B3"] = 138
    sheet["B4"] = formula
    if date_cell:
        sheet["B5"] = datetime(2024, 1, 2)
    sheet["B1"].comment = Comment(f"Follow up {PHONE}", NAME)
    sheet.oddHeader.center.text = f"Record {PHONE}"
    validation = DataValidation(type="list", formula1='"yes,no"', prompt=f"Ask {PHONE}")
    sheet.add_data_validation(validation)
    validation.add("C1")
    for coordinate, value in (extra_cells or {}).items():
        sheet[coordinate] = value
    if hidden:
        notes = workbook.create_sheet("notes")
        notes.sheet_state = "hidden"
        notes["A1"] = f"Backup phone {PHONE}"
    workbook.properties.creator = NAME
    workbook.properties.title = f"Cohort {PHONE}"
    workbook.custom_doc_props.append(StringProperty(name="Owner", value=NAME))
    if defined_formula:
        workbook.defined_names.add(DefinedName(defined_name, attr_text=defined_formula))
    if image:
        from openpyxl.drawing.image import Image as SheetImage
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buffer, format="PNG")
        buffer.seek(0)
        sheet.add_image(SheetImage(buffer), "D1")
    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def _open(output):
    return openpyxl.load_workbook(io.BytesIO(output))


# -- what is rewritten, removed and kept --------------------------------------------------


def test_text_channels_are_rewritten_and_numbers_and_formulas_kept():
    result = wbs.anonymize_workbook_bytes(_book())

    workbook = _open(result.output_bytes)
    sheet = workbook["results"]
    assert PHONE not in sheet["B1"].value and sheet["B1"].value.startswith("Call ")
    assert sheet["B3"].value == 138
    assert sheet["B4"].value == "=B3*2"
    assert PHONE not in sheet["B1"].comment.text
    assert sheet["B1"].comment.author == wbs.NEUTRAL_AUTHOR
    assert PHONE not in sheet.oddHeader.center.text
    assert PHONE not in sheet.data_validations.dataValidation[0].prompt
    assert result.releasable, result.reason_codes


def test_a_hidden_sheet_is_rewritten_and_stays_hidden():
    result = wbs.anonymize_workbook_bytes(_book())

    notes = _open(result.output_bytes)["notes"]
    assert notes.sheet_state == "hidden"
    assert PHONE not in notes["A1"].value
    assert result.summary["hidden_sheets"] == 1


def test_a_very_hidden_sheet_is_read_and_rewritten_like_any_other():
    workbook = openpyxl.load_workbook(io.BytesIO(_book(hidden=False)))
    vault = workbook.create_sheet("vault")
    vault.sheet_state = "veryHidden"
    vault["A1"] = f"Backup phone {PHONE}"
    vault["A2"] = 138
    buffer = io.BytesIO()
    workbook.save(buffer)

    result = wbs.anonymize_workbook_bytes(buffer.getvalue())

    reopened = _open(result.output_bytes)["vault"]
    assert reopened.sheet_state == "veryHidden"
    assert PHONE not in reopened["A1"].value and reopened["A2"].value == 138
    with zipfile.ZipFile(io.BytesIO(result.output_bytes)) as package:
        assert not any(
            PHONE.encode() in package.read(name) for name in package.namelist()
        )


def test_hyperlinks_and_every_property_are_removed():
    result = wbs.anonymize_workbook_bytes(_book())

    workbook = _open(result.output_bytes)
    assert workbook["results"]["B2"].hyperlink is None
    assert workbook.properties.title is None
    assert workbook.properties.creator in (None, "openpyxl")
    assert not list(workbook.custom_doc_props.props)
    assert NAME.encode() not in result.output_bytes
    assert result.summary["hyperlinks_removed"] == 1
    assert result.summary["custom_properties_removed"] == 1


def test_typed_dates_are_removed_under_the_strict_profile():
    result = wbs.anonymize_workbook_bytes(_book())

    assert _open(result.output_bytes)["results"]["B5"].value is None
    assert result.summary["date_cells_removed"] == 1
    assert "workbook_date_cells_removed" in result.warnings


def test_a_repeated_identifier_is_removed_from_every_cell():
    extra = {f"D{row}": f"Repeat {PHONE}" for row in range(1, 6)}

    result = wbs.anonymize_workbook_bytes(_book(extra_cells=extra))

    sheet = _open(result.output_bytes)["results"]
    assert all(PHONE not in sheet[f"D{row}"].value for row in range(1, 6))


# -- what holds the workbook --------------------------------------------------------------


def test_an_identifier_inside_formula_text_holds_the_workbook():
    result = wbs.anonymize_workbook_bytes(_book(formula=f'="Call {PHONE}"&B3'))

    assert wbs.REASON_PHI_IN_FORMULA in result.reason_codes and not result.releasable


def test_an_identifier_in_a_sheet_name_holds_the_workbook():
    result = wbs.anonymize_workbook_bytes(_book(sheet_title=f"Call {PHONE}"))

    assert wbs.REASON_PHI_IN_SHEET_NAME in result.reason_codes


@pytest.mark.parametrize("title", ["renal", "notes", "archive", "clinical results"])
def test_a_structural_sheet_title_is_field_provenance(title):
    assert wbs._structural_sheet_title(title)


@pytest.mark.parametrize("title", ["Ilse Vorhaven", "Call 555-0187", "renal Vorhaven"])
def test_an_unrecognised_sheet_title_still_reaches_identity_detection(title):
    assert not wbs._structural_sheet_title(title)


def test_output_verification_preserves_numeric_results_without_joining_cells(
    monkeypatch,
):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "renal"
    sheet.append(["analyte", "value", "unit"])
    sheet.append(["Urea", 5.4, "mmol/L"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    inspected = []

    def verify(texts, *args, **kwargs):
        inspected.extend(texts)
        return SimpleNamespace(passed=True, blocking_categories={})

    monkeypatch.setattr(wbs, "verify_extracted_text", verify)
    result = wbs.anonymize_workbook_bytes(buffer.getvalue())

    assert result.output_bytes
    assert not any("Urea: 5.4: mmol/L" in text for text in inspected)
    reopened = openpyxl.load_workbook(io.BytesIO(result.output_bytes))
    try:
        assert [reopened.active.cell(2, column).value for column in (1, 2, 3)] == [
            "Urea",
            5.4,
            "mmol/L",
        ]
    finally:
        reopened.close()


def test_an_identifier_in_a_defined_name_constant_holds_the_workbook():
    result = wbs.anonymize_workbook_bytes(
        _book(formula="=contact_value", defined_formula=f'"Call {PHONE}"')
    )

    assert wbs.REASON_PHI_IN_FORMULA in result.reason_codes


def test_an_unused_defined_name_is_a_proven_removed_surface():
    payload = _book(defined_name="OrlaVenn", defined_formula=f'"Call {PHONE}"')

    result = wbs.anonymize_workbook_bytes(payload)
    reopened = _open(result.output_bytes)

    assert "OrlaVenn" not in reopened.defined_names
    assert wbs.REASON_REMOVED_DEFINED_NAME in result.warnings
    assert result.summary["defined_names_removed"] == 1
    assert result.releasable, result.reason_codes


def test_builtin_print_area_is_typed_schema_not_identity():
    workbook = openpyxl.load_workbook(io.BytesIO(_book()))
    workbook["results"].print_area = "A1:B5"
    buffer = io.BytesIO()
    workbook.save(buffer)

    result = wbs.anonymize_workbook_bytes(buffer.getvalue())

    assert result.releasable, result.reason_codes
    assert _open(result.output_bytes)["results"].print_area == "'results'!$A$1:$B$5"


def test_a_formula_into_another_workbook_holds_the_workbook():
    result = wbs.anonymize_workbook_bytes(_book(formula="=[1]Sheet1!$A$1"))

    assert wbs.REASON_EXTERNAL_FORMULA in result.reason_codes


def test_an_identifier_like_integer_holds_the_workbook():
    identifier_like = wbs.IDENTIFIER_LIKE_INTEGER * 4821 + 7

    result = wbs.anonymize_workbook_bytes(_book(extra_cells={"E1": identifier_like}))

    assert wbs.REASON_NUMERIC_IDENTIFIER in result.reason_codes


def test_a_macro_enabled_workbook_is_converted_rather_than_held():
    """Holding one was safe and useless: a clinic exports what its software
    exports, and a workbook is not less releasable for having had a macro
    attached. The project is dropped and the file rebuilt without it; the
    conditions under which it is refused instead are in
    test_xlsm_conversion.
    """
    payload = b"macro payload that must not survive"
    source = zipfile.ZipFile(io.BytesIO(_book()))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("xl/vbaProject.bin", payload)

    result = wbs.anonymize_workbook_bytes(buffer.getvalue())

    assert result.releasable, result.reason_codes
    assert wbs.REASON_MACROS_REMOVED in result.warnings
    assert result.output_extension == ".xlsx"
    assert payload not in result.output_bytes


def test_active_content_still_holds_a_macro_enabled_workbook():
    """The line the conversion draws, asserted here too so that widening it
    is a visible change in this file as well.
    """
    source = zipfile.ZipFile(io.BytesIO(_book()))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for name in source.namelist():
            target.writestr(name, source.read(name))
        target.writestr("xl/vbaProject.bin", b"a macro project")
        target.writestr("xl/activeX/activeX1.xml", b"<x/>")

    result = wbs.anonymize_workbook_bytes(buffer.getvalue())

    assert not result.releasable
    assert (
        result.output_bytes and not result.releasable
    )  # universal review contract: a held copy, never released


def test_an_object_the_writer_would_drop_holds_the_workbook_without_output():
    result = wbs.anonymize_workbook_bytes(_book(image=True))

    assert wbs.REASON_UNSUPPORTED_OBJECT in result.reason_codes
    assert (
        result.output_bytes and not result.releasable
    )  # universal review contract: a held copy, never released


def test_an_unsafe_container_is_held_without_output():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        target.writestr("xl/workbook.xml", "<workbook/>")
        target.writestr("xl/padding.bin", b"\x00" * 5_000_000)

    result = wbs.anonymize_workbook_bytes(buffer.getvalue())

    assert wbs.REASON_CONTAINER in result.reason_codes and result.output_bytes is None


def test_a_rewrite_that_does_not_land_is_caught_on_reopening(monkeypatch):
    monkeypatch.setattr(wbs._Slot, "write", lambda self, text: None)

    result = wbs.anonymize_workbook_bytes(_book())

    assert wbs.REASON_VALUE_CHANGED in result.errors and not result.releasable


# -- input handling ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload,status", [(b"", 400), (b"%PDF-1.7 not a workbook", 415)]
)
def test_invalid_input_is_rejected(payload, status):
    with pytest.raises(MultiformatError) as excinfo:
        wbs.anonymize_workbook_bytes(payload)
    assert excinfo.value.status_code == status


def test_a_zip_that_is_not_a_workbook_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as target:
        target.writestr("word/document.xml", "<document/>")

    with pytest.raises(MultiformatError) as excinfo:
        wbs.anonymize_workbook_bytes(buffer.getvalue())
    assert excinfo.value.status_code == 415


def test_an_oversized_workbook_is_rejected(monkeypatch):
    monkeypatch.setattr(wbs, "MAX_WORKBOOK_BYTES", 64)

    with pytest.raises(MultiformatError) as excinfo:
        wbs.anonymize_workbook_bytes(_book())
    assert excinfo.value.status_code == 413


def test_the_uploaded_bytes_are_never_modified():
    payload = _book()
    original = bytes(payload)

    wbs.anonymize_workbook_bytes(payload)

    assert payload == original


def test_no_value_reaches_the_result_or_the_logs(caplog):
    payload = _book()
    caplog.set_level(logging.DEBUG)

    result = wbs.anonymize_workbook_bytes(payload)

    serialized = repr(result.to_public_dict()) + caplog.text
    assert PHONE not in serialized and NAME not in serialized


# -- through the ingest route --------------------------------------------------------------


@pytest.fixture(scope="module")
def api_client():
    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app)


def test_the_ingest_route_releases_a_verified_workbook(
    api_client, tmp_path, monkeypatch, simulated_production_detectors
):
    from services import sanitized_artifacts

    monkeypatch.setenv(sanitized_artifacts.OUTPUT_DIR_ENV_VAR, str(tmp_path))

    import streaming_support as support

    payload = _book()
    response = api_client.post(
        "/api/v1/ingest",
        files={
            "file": ("cohort.xlsx", io.BytesIO(payload), "application/octet-stream")
        },
        data={"profile": "strict"},
        headers=support.signed_upload_headers_for("owner-a", "/api/v1/ingest", payload),
    )

    body = response.json()
    assert response.status_code == 200, body
    assert body["release_decision"]["releasable"] is True
    download = support.signed(api_client, "owner-a", "GET", body["download_path"])
    assert download.status_code == 200
    assert PHONE not in _open(download.content)["results"]["B1"].value
    assert PHONE not in response.text and NAME not in response.text


def test_the_ingest_route_holds_a_workbook_with_an_image(api_client):
    response = api_client.post(
        "/api/v1/ingest",
        files={
            "file": (
                "cohort.xlsx",
                io.BytesIO(_book(image=True)),
                "application/octet-stream",
            )
        },
        data={"profile": "strict"},
    )

    body = response.json()
    assert body["release_decision"]["disposition"] == "manual_review_required"
    assert wbs.REASON_UNSUPPORTED_OBJECT in body["release_decision"]["reason_codes"]
    assert not body["release_decision"]["releasable"]
