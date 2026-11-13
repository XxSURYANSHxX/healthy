"""Phase 13: cell-preserving CSV anonymization.

Parsing and exact structure, header classification, the policy for each cell,
cell boundaries through batched full_v1, and the independent verifier. Some
tests use a deterministic stand-in for full_v1 so that boundary cases can be
forced; every value is invented.
"""

import csv
import io
import logging
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "csv-sanitizer-test-salt")

from services import csv_sanitizer as cs  # noqa: E402
from services import csv_stream  # noqa: E402
from services.multiformat import TEXT_SEPARATOR, MultiformatError  # noqa: E402

MIXED = (
    "patient_name,mrn,dob,phone,email,sodium_mmol_l,heart_rate,blood_pressure,note\n"
    "Zelquin Vantreske,MRN-4821337,1961-03-14,555-0142,zv@example.invalid,138,72,120/80,"
    "Stable on metformin 500 mg\n"
    "Marbrint Morrowgale,MRN-4821338,1972-11-02,555-0143,mm@example.invalid,141,88,130/85,"
    "Stable on metformin 500 mg\n"
)
IDENTIFIERS = (
    "Zelquin",
    "Vantreske",
    "Marbrint",
    "Morrowgale",
    "4821337",
    "4821338",
    "1961-03-14",
    "555-0142",
    "zv@example.invalid",
)


def _run(text, **kwargs):
    payload = text.encode("utf-8") if isinstance(text, str) else text
    return cs.anonymize_csv_bytes(payload, **kwargs)


def _rows(result):
    body = result.output_bytes.decode("utf-8-sig")
    return list(csv.reader(io.StringIO(body, newline="")))


# -- parsing and structure ----------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        'item,comment\n"a, b","He said ""fine"",\nthen left"\nsodium,"stable, repeat later"\n',
        "﻿diagnosis,weight_kg\r\nasthma,70.5\r\n",
        "diagnosis,weight_kg\nasthma,70.5",
        '"diagnosis","weight_kg"\n"asthma","70.5"\n',
        "diagnosis,weight_kg,extra\nasthma,,\n,70.5,\n",
    ],
)
def test_a_csv_without_phi_is_rewritten_byte_for_byte(payload):
    result = _run(payload)

    assert result.output_bytes == payload.encode("utf-8")
    assert result.summary["round_trip_exact"] is True
    assert result.releasable, result.reason_codes


def test_crlf_bom_and_quote_all_are_reported():
    result = _run('﻿"diagnosis","weight_kg"\r\n"asthma","70.5"\r\n')

    assert result.summary["line_endings"] == "crlf"
    assert result.summary["byte_order_mark"] is True
    assert result.summary["quoting"] == "all"


def test_utf8_names_are_replaced_inside_their_own_cells():
    result = _run("patient_name,diagnosis\nZoë Ångström-Łukasz,asthma\n")

    rows = _rows(result)
    assert rows[1][0].startswith("<PERSON_") and rows[1][1] == "asthma"
    assert "Ångström" not in result.output_bytes.decode("utf-8")


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_other_delimiters_are_refused(delimiter):
    payload = delimiter.join(["name", "age", "weight"]) + "\nZelquin" + delimiter + "45"
    payload += delimiter + "70\nMarbrint" + delimiter + "50" + delimiter + "80\n"

    with pytest.raises(MultiformatError) as excinfo:
        _run(payload)
    assert excinfo.value.status_code == 415


@pytest.mark.parametrize(
    "payload,status",
    [
        (b"", 400),
        (b"\n\n", 400),
        (b'"unterminated\n', 400),
        (b"a,b\n1\n", 400),
        (b'a,b\n"x"y,1\n', 400),
        (b"name\nZo\xeb\n", 400),
        (b"\xff\xfen\x00a\x00m\x00e\x00", 415),
        (b"name\nZel\x00quin\n", 400),
        (b"%PDF-1.7\nname\n", 415),
    ],
)
def test_malformed_ambiguous_or_mismatched_input_is_rejected(payload, status):
    with pytest.raises(MultiformatError) as excinfo:
        cs.anonymize_csv_bytes(payload)
    assert excinfo.value.status_code == status


@pytest.mark.parametrize(
    "limit,value,payload",
    [
        ("MAX_CSV_BYTES", 10, "a,b\n1,2\n3,4\n"),
        ("MAX_CSV_CHARS", 5, "a,b\n1,2\n3,4\n"),
        ("MAX_CSV_ROWS", 1, "a,b\n1,2\n3,4\n"),
        ("MAX_CSV_COLUMNS", 2, "a,b,c\n1,2,3\n"),
        ("MAX_CSV_CELL_CHARS", 3, "a,b\nlonger,2\n"),
        ("MAX_CSV_CELLS", 3, "a,b\n1,2\n"),
    ],
)
def test_every_size_and_shape_limit_is_enforced(monkeypatch, limit, value, payload):
    monkeypatch.setattr(cs, limit, value)

    with pytest.raises(MultiformatError) as excinfo:
        _run(payload)
    assert excinfo.value.status_code == 413


def test_the_uploaded_bytes_are_never_modified():
    payload = bytearray(MIXED.encode("utf-8"))

    cs.anonymize_csv_bytes(payload)

    assert bytes(payload) == MIXED.encode("utf-8")


# -- header classification -----------------------------------------------------------------


@pytest.mark.parametrize(
    "header,entity_type",
    [
        ("mrn", "MEDICAL_RECORD_NUMBER"),
        ("medical_record_number", "MEDICAL_RECORD_NUMBER"),
        ("medical record no", "MEDICAL_RECORD_NUMBER"),
        ("patient_name", "PERSON"),
        ("Patient Name", "PERSON"),
        ("PatientName", "PERSON"),
        ("FIRST", "PERSON"),
        ("dob", "DATE"),
        ("date_of_birth", "DATE"),
        ("chart review date", "DATE"),
        ("phone", "PHONE_NUMBER"),
        ("mobile", "PHONE_NUMBER"),
        ("contact number", "PHONE_NUMBER"),
        ("accession", "ACCESSION_NUMBER"),
        ("specimen id", "ACCESSION_NUMBER"),
        ("device serial", "DEVICE_ID"),
        ("email", "EMAIL_ADDRESS"),
        ("e-mail", "EMAIL_ADDRESS"),
        ("ssn", "US_SSN"),
        ("zip_code", "LOCATION"),
        ("street address", "ADDRESS"),
        ("health_plan_id", "HEALTH_PLAN_ID"),
        ("account_number", "ACCOUNT_NUMBER"),
        ("license_number", "LICENSE_NUMBER"),
        ("vin", "VEHICLE_ID"),
        ("license_plate", "VEHICLE_ID"),
        ("ip_address", "IP_ADDRESS"),
        ("website", "URL"),
        ("photo_url", "IMAGE_REFERENCE"),
        ("fingerprint", "BIOMETRIC_ID"),
        ("encounter_id", "IDENTIFIER"),
        ("age at diagnosis", "AGE_OVER_89"),
    ],
)
def test_identifying_headers_are_recognised_by_their_words(header, entity_type):
    assert cs.classify_header(header).entity_type == entity_type


def test_a_clinician_header_is_a_column_of_names_but_a_note_about_one_is_not():
    assert cs.classify_header("referring physician").entity_type == "PERSON"
    assert cs.classify_header("referring physician").provider is True
    assert cs.classify_header("clinician_note").policy == cs.POLICY_TEXT


@pytest.mark.parametrize(
    "header",
    [
        "sodium_mmol_l",
        "heart_rate",
        "blood_pressure",
        "temperature_c",
        "weight_kg",
        "dose_mg",
        "hba1c_percent",
        "gcs_score",
        "viral_load_copies_ml",
    ],
)
def test_clinical_measurement_headers_are_recognised(header):
    assert cs.classify_header(header).policy == cs.POLICY_CLINICAL


@pytest.mark.parametrize(
    "header",
    [
        "drug_name",
        "test_id",
        "icd10_code",
        "diagnosis",
        "state",
        "diagnosis_year",
    ],
)
def test_clinical_subjects_are_not_identifiers(header):
    assert cs.classify_header(header).entity_type is None


# -- the policy for each cell -------------------------------------------------------------


def test_identifiers_are_replaced_and_clinical_values_kept_cell_for_cell():
    result = _run(MIXED)

    rows = _rows(result)
    assert [len(row) for row in rows] == [9, 9, 9]
    assert rows[0] == MIXED.splitlines()[0].split(",")
    assert [row[5:8] for row in rows[1:]] == [
        ["138", "72", "120/80"],
        ["141", "88", "130/85"],
    ]
    assert rows[1][8] == rows[2][8] == "Stable on metformin 500 mg"
    output = result.output_bytes.decode("utf-8")
    assert not any(identifier in output for identifier in IDENTIFIERS)
    assert result.summary["protected_numeric_cells"] == 6
    # The identifier-removal profile (the default) replaces the direct
    # identifiers and dates cell by cell and does not hold a table for its
    # size: Safe Harbor defines no minimum row count.
    assert result.reason_codes == []
    assert result.summary["csv_privacy_profile"] == "identifier_removal"
    assert result.releasable


def test_one_value_keeps_one_surrogate_and_distinct_values_do_not_share_one():
    body = "patient_name,mrn\n" + "".join(
        f"{name},{mrn}\n"
        for name, mrn in [
            ("Zelquin Vantreske", "MRN-1100001"),
            ("Marbrint Morrowgale", "MRN-1100002"),
        ]
        * 5
    )

    rows = _rows(_run(body))

    assert (
        len({row[0] for row in rows[1::2]}) == 1
        and len({row[0] for row in rows[2::2]}) == 1
    )
    assert rows[1][0] != rows[2][0] and rows[1][1] != rows[2][1]


def test_distinct_texts_go_to_full_v1_once_each_without_crossing_cells():
    body = "diagnosis,note\n" + "asthma,Reviewed in clinic\n" * 40

    result = _run(body)

    # "diagnosis" is a classifier word and stays out of the model; "note" is not.
    assert result.summary["vocabulary_header_cells"] == 1
    assert result.summary["text_cells"] == 81
    assert result.summary["unique_text_cells"] == 3
    assert result.summary["model_calls"] == 3


def test_structured_whole_cell_values_never_reach_the_model():
    seen = []
    body = (
        "contact_detail,diagnosis\nzv@example.invalid,asthma\n555-0142,asthma\n"
        "912-34-5678,asthma\n192.0.2.44,asthma\nhttps://zv.example.invalid/x,asthma\n"
        "2021-03-04,asthma\n"
    )

    result = _run(body, text_anonymizer=_stand_in(seen))

    assert not any(
        re.search(r"@|555-0142|912-34|192\.0\.2|https?://|2021-03-04", text)
        for text in seen
    )
    assert result.summary["deterministic_cells"] == 6
    assert set(result.categories_detected) >= {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "IP_ADDRESS",
        "URL",
        "DATE",
    }


def test_null_markers_booleans_and_numbers_never_reach_the_model():
    seen = []
    body = "allergy,smoker,weight_kg,count\nN/A,yes,70.5,12\nnone,no,,3\n"

    _run(body, text_anonymizer=_stand_in(seen))

    assert not any(
        value in text for text in seen for value in ("N/A", "yes", "70.5", "12")
    )


def test_ages_over_89_and_birth_years_that_reveal_them():
    result = _run("age,birth_year,diagnosis\n94,1920,asthma\n45,1980,asthma\n")

    rows = _rows(result)
    assert rows[1][:2] == ["90+", "<REDACTED_DATE>"]
    assert rows[2][:2] == ["45", "1980"]


def test_clinical_numbers_of_every_shape_are_kept():
    values = [
        "45%",
        "3:1",
        "10-20",
        "0.75",
        "-2.5",
        "<0.01",
        "1,250",
        "12500000",
        "37.2 C",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows([[f"level_{index}" for index in range(len(values))], values])
    body = buffer.getvalue()

    result = _run(body)

    assert _rows(result)[1] == values
    assert result.summary["protected_numeric_cells"] == len(values)
    assert result.releasable, result.reason_codes


@pytest.mark.parametrize(
    "body",
    [
        "ref,value\n4821337,3\n",
        "heart_rate,value\n555-0142,3\n",
        "platelet_count,value\n" + "1" * 9 + ",3\n",
    ],
)
def test_a_number_that_could_be_an_identifier_holds_the_file(body):
    result = _run(body)

    assert cs.REASON_AMBIGUOUS_NUMBER in result.reason_codes
    assert not result.releasable


def test_a_header_of_unknown_labels_is_still_a_header():
    result = _run("diagnosis,note\nasthma,Reviewed in clinic\ncopd,Seen today\n")

    assert result.summary["has_header"] is True
    assert result.summary["rows"] == 2


def test_a_first_row_of_data_is_not_mistaken_for_a_header():
    result = _run("asthma,Reviewed in clinic\nasthma,Reviewed in clinic\n")

    assert result.summary["has_header"] is False
    assert result.summary["rows"] == 2


def test_an_identifier_used_as_a_header_is_replaced():
    rows = _rows(_run("diagnosis,zv.header@example.invalid\nasthma,yes\n"))

    assert rows[0] == ["diagnosis", "<EMAIL_1>"]


def test_a_headerless_csv_is_anonymized_by_content():
    result = _run(
        "Zelquin Vantreske called from 555-0142,138\nFollow up at zv@example.invalid,140\n"
    )

    output = result.output_bytes.decode("utf-8")
    assert result.summary["has_header"] is False
    assert "555-0142" not in output and "zv@example.invalid" not in output
    assert _rows(result)[0][1] == "138" and _rows(result)[1][1] == "140"


# -- cell boundaries through batched full_v1 --------------------------------------------------

_NAME = re.compile(r"Zelquin Vantreske|Marbrint Morrowgale")


def _stand_in(seen=None, crossing=False, rebuild_wrong=False, names=_NAME):
    """A deterministic full_v1: replaces the two invented names by surrogate."""

    def anonymize(text, profile, allocator):
        if seen is not None:
            seen.append(text)
        replacements = [
            (
                match.start(),
                match.end(),
                "PERSON",
                allocator.surrogate_for("PERSON", match.group()),
            )
            for match in names.finditer(text)
        ]
        if crossing and TEXT_SEPARATOR in text:
            cut = text.index(TEXT_SEPARATOR)
            replacements = [
                r for r in replacements if r[1] <= cut - 2 or r[0] >= cut + 5
            ]
            replacements.append(
                (cut - 2, cut + len(TEXT_SEPARATOR) + 2, "PERSON", "PATIENT_900")
            )
        replacements.sort()
        output, cursor = [], 0
        for start, end, _type, replacement in replacements:
            output.append(text[cursor:start])
            output.append(replacement)
            cursor = end
        output.append(text[cursor:])
        anonymized = "".join(output) + ("?" if rebuild_wrong else "")
        return {
            "anonymized_text": anonymized,
            "detected_entities": {"PERSON": len(replacements)} if replacements else {},
            "residual_phi_categories": {},
            "review_required_reasons": [],
        }, replacements

    return anonymize


def test_replacements_land_in_their_own_cells_even_across_boundaries_inside_values():
    body = (
        'note,diagnosis\n"Seen by Zelquin Vantreske.\n\nNext visit",asthma\n'
        "Marbrint Morrowgale reviewed,asthma\n"
    )

    result = _run(body, text_anonymizer=_stand_in())

    rows = _rows(result)
    assert rows[1][0] == "Seen by <PERSON_1>.\n\nNext visit"
    assert rows[2][0] == "<PERSON_2> reviewed"
    assert rows[1][1] == rows[2][1] == "asthma"


def test_each_identifier_cell_is_already_isolated():
    seen = []
    body = "Zelquin Vantreske\nMarbrint Morrowgale\n"

    result = _run(body, text_anonymizer=_stand_in(seen, crossing=True))

    assert result.summary["batches_redone_per_cell"] == 0
    assert len(seen) == 2
    assert all(TEXT_SEPARATOR not in text for text in seen)
    assert cs.REASON_CROSSING not in result.reason_codes
    assert sorted(row[0] for row in _rows(result)) == ["<PERSON_1>", "<PERSON_2>"]


def test_a_replacement_that_requires_crossing_cells_cannot_be_created():
    seen = []
    body = "note\nfirst part\nsecond part\n"

    result = _run(body, text_anonymizer=_stand_in(seen, crossing=True))

    assert [text.count(TEXT_SEPARATOR) for text in seen] == [0, 0, 0]
    assert cs.REASON_CROSSING not in result.reason_codes
    assert [row[0] for row in _rows(result)] == ["note", "first part", "second part"]
    assert result.releasable


def test_an_output_that_does_not_rebuild_from_its_replacements_is_an_error():
    result = _run(
        "note\nZelquin Vantreske\n", text_anonymizer=_stand_in(rebuild_wrong=True)
    )

    assert cs.REASON_MAPPING in result.errors
    assert not result.releasable


def test_tabular_calls_are_one_distinct_value_each(monkeypatch):
    seen = []
    monkeypatch.setattr(cs, "MAX_BATCH_FIELDS", 2)
    body = "note\n" + "".join(
        f"Visit {word}\n" for word in ("one", "two", "three", "four")
    )

    _run(body, text_anonymizer=_stand_in(seen))

    assert len(seen) == 5
    assert all(TEXT_SEPARATOR not in text for text in seen)


@pytest.mark.parametrize(
    "limit,reason",
    [
        ("MAX_MODEL_FIELDS", cs.REASON_MODEL_FIELD_LIMIT),
        ("MAX_MODEL_CHARS", cs.REASON_MODEL_TEXT_LIMIT),
    ],
)
def test_too_much_distinct_text_holds_before_any_model_call(monkeypatch, limit, reason):
    seen = []
    monkeypatch.setattr(cs, limit, 1)

    result = _run("note\nVisit one\nVisit two\n", text_anonymizer=_stand_in(seen))

    assert reason in result.reason_codes
    assert seen == [] and result.output_bytes is None


# -- the independent verifier ---------------------------------------------------------------


def _render_then(change):
    """Tamper with each row the writer emits; the verifier must still catch it."""
    return change


def test_a_written_cell_that_differs_from_its_plan_is_caught(monkeypatch):
    monkeypatch.setattr(
        csv_stream,
        "_render_output",
        _render_then(lambda text: text.replace("<PERSON_1>", "Zelquin Vantreske", 1)),
    )

    result = _run(MIXED)

    assert cs.REASON_VALUE_CHANGED in result.errors
    assert cs.REASON_PHI_SURVIVED in result.reason_codes
    assert not result.releasable


def test_a_lost_row_is_caught(monkeypatch):
    rows = []

    def drop_last(text):
        rows.append(text)
        return "" if len(rows) == 3 else text

    monkeypatch.setattr(csv_stream, "_render_output", _render_then(drop_last))

    result = _run(MIXED)

    assert cs.REASON_STRUCTURE_CHANGED in result.errors


def test_a_changed_protected_value_is_caught(monkeypatch):
    monkeypatch.setattr(
        csv_stream,
        "_render_output",
        _render_then(lambda text: text.replace(",138,", ",139,")),
    )

    result = _run(MIXED)

    assert cs.REASON_PROTECTED_CHANGED in result.errors


def test_a_batch_boundary_written_into_the_output_is_caught(monkeypatch):
    monkeypatch.setattr(
        csv_stream,
        "_render_output",
        _render_then(
            lambda text: text.replace(
                "Stable on metformin 500 mg", '"Stable.\n\non metformin"', 1
            )
        ),
    )

    result = _run(MIXED)

    assert cs.REASON_MARKER_SURVIVED in result.errors


def test_part_of_an_identifier_left_behind_holds_the_file(monkeypatch):
    from services import text_anonymization

    monkeypatch.setattr(
        text_anonymization,
        "replacement_for_value",
        lambda entity_type, value, **kwargs: value[-7:],
    )

    result = _run("mrn,diagnosis\nMRN-4821337,asthma\n")

    assert cs.REASON_PARTIAL_SURVIVED in result.reason_codes
    assert cs.REASON_HEADER_IDENTIFIER in result.reason_codes
    assert not result.releasable


def test_equal_digits_in_a_typed_clinical_cell_do_not_become_a_global_survivor():
    payload = "pager,lab_result,unit,diagnosis_code\n" "763184,763184,U/L,J63.184\n"
    result = _run(payload)
    output = result.output_bytes.decode("utf-8")

    assert "<PHONE_" in output
    assert ",763184,U/L," in output
    assert "J63.184" in output
    assert cs.REASON_PHI_SURVIVED not in result.reason_codes
    assert cs.REASON_PARTIAL_SURVIVED not in result.reason_codes


def test_a_residual_finding_in_the_rescan_holds_the_file():
    def scanner(text, provenance):
        return [{"blocking": True, "category": "PERSON"}]

    result = _run(MIXED, residual_scanner=scanner)

    assert cs.REASON_RESIDUAL in result.reason_codes
    assert "residual_person" in result.reason_codes
    assert result.residual_verification_passed is False
    assert not result.releasable


def test_no_value_reaches_the_result_or_the_logs(caplog):
    caplog.set_level(logging.DEBUG)

    result = _run(MIXED)

    surface = repr(result.to_public_dict()) + caplog.text
    assert not any(identifier in surface for identifier in IDENTIFIERS)
    assert result.output_media_type == "text/csv; charset=utf-8"
    assert result.output_extension == ".csv"
