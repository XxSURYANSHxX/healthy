"""Focused tabular-privacy-v1 policy and corpus qualification tests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tabular_corpus as build_corpus  # noqa: E402
from services import (  # noqa: E402
    csv_sanitizer,
    dataset_privacy,
    tabular_privacy,
    workbook_sanitizer,
)
from services.multiformat import TEXT_SEPARATOR  # noqa: E402
from tabular_doubles import _clean_scanner, _mock_anonymizer  # noqa: E402

CORPUS = Path(__file__).resolve().parents[1] / "evaluations" / "tabular_hipaa18_v1"


@pytest.fixture(autouse=True)
def temporary_corpus(tmp_path, monkeypatch):
    import tabular_doubles

    monkeypatch.setattr(sys.modules[__name__], "CORPUS", tmp_path)
    monkeypatch.setattr(build_corpus, "ROOT", tmp_path)
    monkeypatch.setattr(tabular_doubles, "ROOT", tmp_path)
    build_corpus.build()


def test_policy_declares_all_six_roles_and_all_18_safe_harbor_categories():
    assert tabular_privacy.POLICY_VERSION == "tabular-privacy-v1"
    assert tabular_privacy.COLUMN_ROLES == frozenset(
        {
            "direct_identifier",
            "quasi_identifier",
            "sensitive_clinical",
            "clinical_utility",
            "non_sensitive",
            "unknown",
        }
    )
    coverage = tabular_privacy.safe_harbor_coverage()
    assert [item["number"] for item in coverage] == list(range(1, 19))
    assert all(item["entity_types"] for item in coverage)


def test_occurrence_context_distinguishes_equal_digits_and_protects_clinical_terms():
    unknown = tabular_privacy.ColumnPolicy(0, tabular_privacy.ROLE_UNKNOWN)
    clinical = tabular_privacy.ColumnPolicy(0, tabular_privacy.ROLE_CLINICAL_UTILITY)
    assert (
        tabular_privacy.occurrence_decision(
            "4417", header="contact", column=unknown, row_context=("Bleep",)
        ).action
        == tabular_privacy.ACTION_REPLACE
    )
    result = tabular_privacy.occurrence_decision(
        "4417", header="lab_value", column=clinical, row_context=("Bleep",)
    )
    assert result.action == tabular_privacy.ACTION_KEEP and result.clinical_invariant
    for value in ("PCV20", "DESS", "PAS-D stain", "HbA1c", "mU/L", "3 months"):
        assert tabular_privacy.stable_identifier_type(value) is None


def test_hormone_analyte_needs_clinical_role_and_matching_header():
    clinical = tabular_privacy.ColumnPolicy(0, tabular_privacy.ROLE_CLINICAL_UTILITY)
    unknown = tabular_privacy.ColumnPolicy(0, tabular_privacy.ROLE_UNKNOWN)
    protected = tabular_privacy.occurrence_decision(
        "Testosterone", header="testosterone", column=clinical
    )
    assert (
        protected.clinical_invariant and protected.action == tabular_privacy.ACTION_KEEP
    )
    assert not tabular_privacy.occurrence_decision(
        "Testosterone", header="note", column=clinical
    ).clinical_invariant
    assert not tabular_privacy.occurrence_decision(
        "Testosterone", header="testosterone", column=unknown
    ).clinical_invariant
    assert not tabular_privacy.occurrence_decision(
        "Testosterone Smith", header="testosterone", column=clinical
    ).clinical_invariant


def test_synthetic_disclaimer_is_exact_not_blanket_synthetic_exemption():
    non_sensitive = tabular_privacy.ColumnPolicy(0, tabular_privacy.ROLE_NON_SENSITIVE)
    protected = tabular_privacy.occurrence_decision(
        "SYNTHETIC - NOT A REAL PATIENT", header="fixture_label", column=non_sensitive
    )
    assert (
        protected.content_invariant and protected.action == tabular_privacy.ACTION_KEEP
    )
    assert not tabular_privacy.occurrence_decision(
        "SYNTHETIC - PATIENT Jane Doe", header="fixture_label", column=non_sensitive
    ).content_invariant
    assert not tabular_privacy.occurrence_decision(
        "SYNTHETIC - NOT A REAL PATIENT", header="note", column=non_sensitive
    ).content_invariant


def test_a_declared_non_sensitive_column_keeps_the_exact_statement_under_any_header():
    """Found by the first fresh confirmation set: the carve-out was keyed to two
    header names, so the same declared statement under another name was rewritten."""
    statement = "SYNTHETIC - NOT A REAL PATIENT"
    declared = tabular_privacy.ColumnPolicy(
        15, tabular_privacy.ROLE_NON_SENSITIVE, explicit=True
    )
    for header in ("document_label", "provenance_statement", "dataset_notice"):
        decision = tabular_privacy.occurrence_decision(
            statement, header=header, column=declared
        )
        assert (
            decision.content_invariant
            and decision.action == tabular_privacy.ACTION_KEEP
        )
    # Only the exact statement: added text, a name or a different sentence is not kept.
    for value in (
        f"{statement} Jordan Fictional",
        "SYNTHETIC - PATIENT Jane Doe",
        "NOT A REAL PATIENT",
        "SYNTHETIC",
    ):
        assert not tabular_privacy.occurrence_decision(
            value, header="document_label", column=declared
        ).content_invariant, value
    # Only a column the signed schema declares non-sensitive: an undeclared
    # column, or a clinical, quasi-identifier or direct-identifier one, is not exempt.
    for column in (
        None,
        tabular_privacy.ColumnPolicy(15, tabular_privacy.ROLE_NON_SENSITIVE),
        tabular_privacy.ColumnPolicy(
            15, tabular_privacy.ROLE_CLINICAL_UTILITY, explicit=True
        ),
        tabular_privacy.ColumnPolicy(
            15,
            tabular_privacy.ROLE_QUASI_IDENTIFIER,
            qi_kind="occupation",
            explicit=True,
        ),
        tabular_privacy.ColumnPolicy(
            15,
            tabular_privacy.ROLE_DIRECT_IDENTIFIER,
            entity_type="PERSON",
            explicit=True,
        ),
    ):
        assert not tabular_privacy.occurrence_decision(
            statement, header="document_label", column=column
        ).content_invariant


def test_generic_facility_type_is_a_category_but_a_named_clinic_is_not():
    for kind in ("institution", "facility"):
        facility = tabular_privacy.ColumnPolicy(
            0, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind=kind
        )
        for value in ("Community Clinic", "Outpatient Center", "Urgent Care Unit"):
            decision = tabular_privacy.occurrence_decision(
                value, header="facility", column=facility
            )
            assert decision.content_invariant and decision.deterministic
            assert csv_sanitizer.is_category_label(
                dataset_privacy.QI_INSTITUTION, value
            )
        for value in ("Smith Clinic", "Royal Free Hospital", "Community Clinic Alice"):
            assert not tabular_privacy.occurrence_decision(
                value, header="facility", column=facility
            ).content_invariant
            assert not csv_sanitizer.is_category_label(
                dataset_privacy.QI_INSTITUTION, value
            )


def test_signed_postal_area_is_replaced_without_short_number_ambiguity():
    declared = tabular_privacy.ColumnPolicy(
        28, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind="zip", explicit=True
    )
    decision = tabular_privacy.occurrence_decision(
        "90000", header="postal_area", column=declared
    )
    assert decision.action == tabular_privacy.ACTION_REPLACE
    assert decision.entity_type == "LOCATION" and decision.deterministic
    for value, header, column in (
        ("90000 Alice", "postal_area", declared),
        ("90000", "account", declared),
        (
            "90000",
            "postal_area",
            tabular_privacy.ColumnPolicy(
                28, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind="zip"
            ),
        ),
    ):
        assert (
            tabular_privacy.occurrence_decision(
                value, header=header, column=column
            ).action
            != tabular_privacy.ACTION_REPLACE
            or tabular_privacy.occurrence_decision(
                value, header=header, column=column
            ).entity_type
            != "LOCATION"
        )


def test_declared_postal_area_accepts_zip_plus_four_and_rejects_mixed_values():
    declared = tabular_privacy.ColumnPolicy(
        28, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind="zip", explicit=True
    )
    for value in ("00501", "02134", "90000-1234", "00501-0001"):
        decision = tabular_privacy.occurrence_decision(
            value, header="postal_area", column=declared
        )
        assert decision.action == tabular_privacy.ACTION_REPLACE, value
        assert decision.entity_type == "LOCATION" and decision.deterministic
    for value in ("9000", "900001", "90000-12", "ZIP 90000", "90000 98"):
        decision = tabular_privacy.occurrence_decision(
            value, header="postal_area", column=declared
        )
        assert not (
            decision.action == tabular_privacy.ACTION_REPLACE
            and decision.entity_type == "LOCATION"
            and decision.deterministic
        ), value


@pytest.mark.parametrize("profile", ["identifier_removal", "safe_harbor_plus_qi"])
def test_postal_replacement_is_column_scoped_in_written_csv(profile):
    """The same digits are replaced only where a signed ZIP column holds them."""
    zips = ["02134", "90000", "00501", "02134", "90000", "00501", "02134", "90000"]
    lines = ["postal_area,spo2_percent,lab_value,comment,note"]
    for index, zip_code in enumerate(zips):
        lines.append(
            f"{zip_code},98,90000,x{zip_code},result 90000 beside {zip_code}"
            if index == 0
            else f"{zip_code},98,00501,x{zip_code},stable"
        )
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": profile,
        "k": 5,
        "l": None,
        "columns": [
            {
                "index": 0,
                "header": "postal_area",
                "role": "quasi_identifier",
                "qi_kind": "zip",
            },
            {"index": 1, "header": "spo2_percent", "role": "clinical_utility"},
            {"index": 2, "header": "lab_value", "role": "clinical_utility"},
            {"index": 3, "header": "comment", "role": "non_sensitive"},
            {"index": 4, "header": "note", "role": "clinical_utility"},
        ],
    }
    result = csv_sanitizer.anonymize_csv_bytes(
        ("\n".join(lines) + "\n").encode(),
        tabular_manifest=manifest,
        text_anonymizer=_mock_anonymizer,
        residual_scanner=_clean_scanner,
    )
    written = list(csv.reader(io.StringIO(result.output_bytes.decode())))
    assert written[0] == lines[0].split(",")
    for source, record in zip(zips, written[1:]):
        # Every ZIP, including leading-zero values, is gone from its column.
        assert source not in record[0]
        if profile == "identifier_removal":
            assert record[0].startswith("<LOCATION_")
        else:
            assert record[0] in {"*"} or record[0].startswith("<LOCATION_")
        # Clinical measurements, and identical digits elsewhere, are exact.
        assert record[1] == "98"
    assert written[1][2] == "90000" and written[2][2] == "00501"
    assert written[1][3] == "x02134" and written[2][3] == "x90000"
    assert written[1][4] == "result 90000 beside 02134"


def _extension_anonymizer(leak: bool = False):
    """A detector double that replaces only the digits after the word 'extension'."""
    import re

    def anonymize(text, profile, allocator):
        replacements = []
        for match in re.finditer(r"(?<=extension )\d+", text):
            token = (
                f"x{match.group()}y"
                if leak
                else allocator.surrogate_for("PHONE_NUMBER", match.group())
            )
            replacements.append((match.start(), match.end(), "PHONE_NUMBER", token))
        output = text
        for start, end, _entity, token in reversed(replacements):
            output = output[:start] + token + output[end:]
        return (
            {
                "anonymized_text": output,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            replacements,
        )

    return anonymize


_TWIN_MANIFEST = {
    "policy_version": "tabular-privacy-v1",
    "profile": "identifier_removal",
    "columns": [
        {"index": 0, "header": "note", "role": "clinical_utility"},
        {"index": 1, "header": "lab_value", "role": "clinical_utility"},
    ],
}


def _twin_csv(note):
    return (f'note,lab_value\n"{note}",4417\nStable,98\nStable,98\n').encode()


def test_same_digits_kept_as_a_clinical_result_are_not_a_partial_survivor():
    note = "Bleep: extension 4417; laboratory result 4417 U/L"
    result = csv_sanitizer.anonymize_csv_bytes(
        _twin_csv(note),
        tabular_manifest=_TWIN_MANIFEST,
        text_anonymizer=_extension_anonymizer(),
        residual_scanner=_clean_scanner,
    )
    written = list(csv.reader(io.StringIO(result.output_bytes.decode())))
    assert "csv_partial_identifier_survived" not in result.reason_codes
    assert "extension 4417" not in written[1][0]
    assert (
        written[1][0].count("4417") == 1
        and "laboratory result 4417 U/L" in written[1][0]
    )
    assert written[1][1] == "4417"


def test_a_replacement_that_leaks_the_replaced_digits_is_still_a_survivor():
    note = "Bleep: extension 4417; laboratory result 4417 U/L"
    result = csv_sanitizer.anonymize_csv_bytes(
        _twin_csv(note),
        tabular_manifest=_TWIN_MANIFEST,
        text_anonymizer=_extension_anonymizer(leak=True),
        residual_scanner=_clean_scanner,
    )
    assert "csv_partial_identifier_survived" in result.reason_codes


def test_a_whole_cell_replacement_that_keeps_source_digits_is_a_survivor():
    """A deterministic whole-cell replacement has no preserved occurrence."""
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "identifier_removal",
        "columns": [
            {
                "index": 0,
                "header": "bleep",
                "role": "direct_identifier",
                "entity_type": "PHONE_NUMBER",
            }
        ],
    }
    result = csv_sanitizer.anonymize_csv_bytes(
        b"bleep\n4417\n4418\n",
        tabular_manifest=manifest,
        text_anonymizer=_mock_anonymizer,
        residual_scanner=_clean_scanner,
    )
    assert "csv_partial_identifier_survived" not in result.reason_codes
    written = list(csv.reader(io.StringIO(result.output_bytes.decode())))
    assert "4417" not in written[1][0] and "4418" not in written[2][0]


def test_declared_age_keeps_only_valid_under_90_whole_values():
    age = tabular_privacy.ColumnPolicy(
        0, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind="age", explicit=True
    )
    for value in ("0", "24", "89"):
        decision = tabular_privacy.occurrence_decision(value, header="age", column=age)
        assert decision.action == tabular_privacy.ACTION_KEEP
        assert decision.content_invariant and decision.deterministic
    for value in ("90", "120"):
        decision = tabular_privacy.occurrence_decision(value, header="age", column=age)
        assert decision.action == tabular_privacy.ACTION_REPLACE
        assert decision.entity_type == "AGE_OVER_89"
    assert (
        tabular_privacy.occurrence_decision("121", header="age", column=age).action
        == tabular_privacy.ACTION_HOLD
    )
    assert not tabular_privacy.occurrence_decision(
        "24 Jane Doe", header="age", column=age
    ).content_invariant


def test_manifest_profiles_and_duplicate_headers_fail_closed():
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "safe_harbor_plus_qi",
        "k": 7,
        "l": 3,
        "columns": [
            {"index": 0, "role": "quasi_identifier", "qi_kind": "age"},
            {"header": "diagnosis", "role": "sensitive_clinical"},
        ],
    }
    config = tabular_privacy.policy_config(manifest, allow_environment_manifest=False)
    assert (config.k, config.l, config.t) == (7, 3, None)
    columns = tabular_privacy.classify_columns(
        ["age", "diagnosis"], [["42", "I10"]], config
    )
    assert [item.role for item in columns] == ["quasi_identifier", "sensitive_clinical"]
    with pytest.raises(tabular_privacy.ManifestError):
        tabular_privacy.classify_columns(["diagnosis", "diagnosis"], [], config)


def test_equivalence_classes_generalize_and_verify_actual_written_rows():
    roles = dataset_privacy.ColumnRoles(
        {0: dataset_privacy.QI_AGE, 1: dataset_privacy.QI_ZIP}, (2,), ()
    )
    rows = [
        [str(20 + index), f"00{index % 2}00", ("I10", "E11.9")[index % 2]]
        for index in range(10)
    ]
    policy = dataset_privacy.PrivacyPolicy(
        k_anonymity=5, l_diversity=2, t_closeness=None
    )
    outcome = dataset_privacy.anonymize_dataset(rows, roles, policy=policy)
    assert outcome.releasable
    check = dataset_privacy.verify_dataset(outcome.rows, roles, policy=policy)
    assert check.satisfied and check.k >= 5 and check.l >= 2
    assert outcome.generalized


def test_synthetic_pack_small_occupation_group_is_transformed_without_models(tmp_path):
    pack = _occupation_pack(tmp_path)
    policy = json.loads((pack / "qi_policy.json").read_text(encoding="utf-8"))

    def identity(text, _profile, _allocator):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    source = (pack / "synthetic_hipaa18_clinical.csv").read_bytes()
    result = csv_sanitizer.anonymize_csv_bytes(
        source,
        tabular_manifest=policy,
        tabular_profile="safe_harbor_plus_qi",
        text_anonymizer=identity,
        residual_scanner=lambda _text, _provenance: [],
    )
    assert result.output_bytes is not None
    original = list(csv.reader(io.StringIO(source.decode("utf-8"), newline="")))
    rows = list(
        csv.reader(io.StringIO(result.output_bytes.decode("utf-8"), newline=""))
    )
    occupation = rows[0].index("occupation")
    assert any(
        rows[line - 1][occupation] != original[line - 1][occupation] for line in (5, 6)
    )
    assert result.summary["cells_generalized"] > 0


def test_workbook_qi_accepts_its_own_aggregate_age_placeholder(monkeypatch, tmp_path):
    from services.multiformat import MultiformatResult
    from services.surrogates import SurrogateAllocator

    monkeypatch.setenv("BIOBLOCK_STUDY_SALT", "workbook-aggregate-age-test")
    pack = _occupation_pack(tmp_path)
    config = tabular_privacy.policy_config(
        json.loads((pack / "qi_policy.json").read_text(encoding="utf-8")),
        profile="safe_harbor_plus_qi",
        allow_environment_manifest=False,
    )
    workbook = openpyxl.load_workbook(pack / "synthetic_hipaa18_clinical.xlsx")
    result = MultiformatResult(
        "workbook",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "zip",
    )
    state = workbook_sanitizer._apply_tabular_policy(
        workbook, result, "strict", config, SurrogateAllocator()
    )
    assert "csv_quasi_identifier_ambiguous" not in result.reason_codes
    assert state.transformations["cells_generalized"] > 0
    assert state.plans
    assert "Nurse" not in state.replaced_originals


def test_corpus_is_structurally_honest_synthetic_and_digest_pinned():
    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))
    gold = json.loads((CORPUS / "gold.json").read_text(encoding="utf-8"))
    digests = json.loads((CORPUS / "digests.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == 60 and manifest["synthetic_only"] is True
    assert manifest["default_profile"] == "safe_harbor_plus_qi"
    assert len(set(manifest["safe_harbor_categories"])) == 18
    assert manifest["surfaces"]["csv_embedded_image"] == "not_applicable_for_surface"
    assert len({item["category"] for item in gold["direct_identifier_columns"]}) == 18
    for filename, expected in digests.items():
        assert hashlib.sha256((CORPUS / filename).read_bytes()).hexdigest() == expected
    with (CORPUS / "tabular_hipaa18_v1.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.reader(handle))) == 61
    workbook = openpyxl.load_workbook(
        CORPUS / "tabular_hipaa18_v1.xlsx", data_only=False
    )
    try:
        assert [sheet.sheet_state for sheet in workbook.worksheets] == [
            "visible",
            "hidden",
            "veryHidden",
        ]
        assert workbook["Clinical Data"]._images and workbook["Clinical Data"]._charts
        assert workbook["Clinical Data"]["A64"].value.startswith("=[1]")
    finally:
        workbook.close()


def test_builder_produces_identical_artifact_bytes_twice():
    first = build_corpus.build()
    first_bytes = {name: (CORPUS / name).read_bytes() for name in first}
    second = build_corpus.build()
    assert first == second
    assert all(first_bytes[name] == (CORPUS / name).read_bytes() for name in first)


def test_workbook_model_calls_are_isolated_by_surface_and_sheet(monkeypatch):
    seen = []

    def identity(self, text):
        seen.append(text)
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    monkeypatch.setattr(workbook_sanitizer.TextChannelSanitizer, "_anonymize", identity)
    workbook = openpyxl.Workbook()
    workbook.active.title = "Clinical Data"
    workbook.active.append(["note"])
    workbook.active.append(["first isolated narrative"])
    second = workbook.create_sheet("Clinical Notes")
    second.append(["note"])
    second.append(["second isolated narrative"])
    buffer = io.BytesIO()
    workbook.save(buffer)

    workbook_sanitizer.anonymize_workbook_bytes(buffer.getvalue())

    assert seen
    assert all(TEXT_SEPARATOR not in text for text in seen)
    assert not any(
        "first isolated" in text and "second isolated" in text for text in seen
    )


def test_privacy_report_is_value_free_and_records_profile_and_outcome():
    manifest = json.loads(
        (CORPUS / "quasi_identifier_policy.json").read_text(encoding="utf-8")
    )
    gold = json.loads((CORPUS / "gold.json").read_text(encoding="utf-8"))
    result = csv_sanitizer.anonymize_csv_bytes(
        (CORPUS / "tabular_hipaa18_v1.csv").read_bytes(),
        tabular_manifest=manifest,
        tabular_profile="safe_harbor_plus_qi",
        text_anonymizer=_mock_anonymizer,
        residual_scanner=_clean_scanner,
    )
    report = result.summary["tabular_privacy"]
    serialized = json.dumps(report, sort_keys=True)
    assert report["policy_version"] == "tabular-privacy-v1"
    assert report["profile"] == "safe_harbor_plus_qi"
    assert report["candidate_state"] == "held"
    assert len(report["direct_identifier_categories"]) == 18
    assert all(item["value"] not in serialized for item in gold["planted_values"][:100])


def test_confirmed_row_identifier_in_mixed_note_has_exact_span_not_clinical_text():
    columns = tabular_privacy.classify_columns(
        ["patient_name", "note", "lab_value"],
        [["SyntheticPatient-000", "PATIENT_NAME: SyntheticPatient-000; PCV20", "7.2%"]],
    )
    row = ["SyntheticPatient-000", "PATIENT_NAME: SyntheticPatient-000; PCV20", "7.2%"]
    spans = tabular_privacy.confirmed_row_identifier_spans(row[1], row, columns, 1)
    assert [(row[1][start:end], entity) for start, end, entity in spans] == [
        ("SyntheticPatient-000", "PERSON")
    ]


def test_csv_mixed_note_replaces_confirmed_identifier_when_model_misses_it():
    body = (
        "patient_name,note,lab_value\n"
        "SyntheticPatient-000,PATIENT_NAME: SyntheticPatient-000; PCV20,7.2%\n"
        "SyntheticPatient-001,PATIENT_NAME: SyntheticPatient-001; PCV20,7.2%\n"
    ).encode()

    def identity(text, _profile, _allocator):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    result = csv_sanitizer.anonymize_csv_bytes(
        body, text_anonymizer=identity, residual_scanner=_clean_scanner
    )
    assert result.output_bytes is not None
    output = result.output_bytes.decode()
    assert "SyntheticPatient-000" not in output
    assert "SyntheticPatient-001" not in output
    assert output.count("PCV20") == 2


def test_xlsx_mixed_note_replaces_confirmed_identifier_when_model_misses_it(
    monkeypatch,
):
    def identity(self, text):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    monkeypatch.setattr(workbook_sanitizer.TextChannelSanitizer, "_anonymize", identity)
    monkeypatch.setattr(
        workbook_sanitizer,
        "verify_extracted_text",
        lambda _texts, _tokens, **_kwargs: type(
            "Clean", (), {"passed": True, "blocking_categories": ()}
        )(),
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Clinical Data"
    sheet.append(["patient_name", "note", "lab_value"])
    sheet.append(
        ["SyntheticPatient-000", "PATIENT_NAME: SyntheticPatient-000; PCV20", "7.2%"]
    )
    sheet.append(
        ["SyntheticPatient-001", "PATIENT_NAME: SyntheticPatient-001; PCV20", "7.2%"]
    )
    stream = io.BytesIO()
    workbook.save(stream)
    result = workbook_sanitizer.anonymize_workbook_bytes(stream.getvalue())
    assert result.output_bytes is not None
    reopened = openpyxl.load_workbook(io.BytesIO(result.output_bytes), data_only=False)
    try:
        assert "SyntheticPatient-000" not in reopened.active["B2"].value
        assert "SyntheticPatient-001" not in reopened.active["B3"].value
        assert reopened.active["B2"].value.endswith("PCV20")
    finally:
        reopened.close()


def test_xlsx_table_headers_are_schema_not_model_text(monkeypatch):
    def hostile_header_model(self, text):
        rewritten = "<SSN_99>" if text in {"ssn", "icd10", "lab_value"} else text
        spans = [(0, len(text), "US_SSN")] if rewritten != text else []
        return (
            {
                "anonymized_text": rewritten,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            spans,
        )

    monkeypatch.setattr(
        workbook_sanitizer.TextChannelSanitizer, "_anonymize", hostile_header_model
    )
    monkeypatch.setattr(
        workbook_sanitizer,
        "verify_extracted_text",
        lambda _texts, _tokens, **_kwargs: type(
            "Clean", (), {"passed": True, "blocking_categories": ()}
        )(),
    )
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["ssn", "icd10", "lab_value"])
    sheet.append(["910-10-1000", "E11.9", "7.2%"])
    sheet.append(["911-11-1001", "I10", "7.2%"])
    stream = io.BytesIO()
    workbook.save(stream)
    result = workbook_sanitizer.anonymize_workbook_bytes(stream.getvalue())
    assert result.output_bytes is not None
    reopened = openpyxl.load_workbook(io.BytesIO(result.output_bytes))
    try:
        assert [cell.value for cell in reopened.active[1]] == [
            "ssn",
            "icd10",
            "lab_value",
        ]
    finally:
        reopened.close()


def test_embedded_forged_token_is_an_exact_identifier_span():
    text = "PCV20; <PATIENT_999>; 7.2%"
    spans = tabular_privacy.embedded_fixed_identifier_spans(text)
    assert [(text[start:end], entity) for start, end, entity in spans] == [
        ("<PATIENT_999>", "OTHER_UNIQUE_ID")
    ]


def test_declared_schema_header_is_preserved_without_model_edit():
    column = tabular_privacy.ColumnPolicy(
        0, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind="service_line", explicit=True
    )
    assert tabular_privacy.is_declared_schema_header("service_line", column)
    assert not tabular_privacy.is_declared_schema_header("Jordan Fictional", column)


def test_declared_closed_qi_categories_preserve_meaning_but_not_names():
    for kind, safe in (
        ("race_ethnicity", "Black or African American"),
        ("race_ethnicity", "Hispanic or Latino"),
        ("location", "California"),
        ("sex", "F"),
    ):
        column = tabular_privacy.ColumnPolicy(
            0, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind=kind, explicit=True
        )
        decision = tabular_privacy.occurrence_decision(safe, column=column)
        assert decision.content_invariant and decision.deterministic
        assert not tabular_privacy.occurrence_decision(
            "Jordan Fictional", column=column
        ).content_invariant
    postal = tabular_privacy.ColumnPolicy(
        0, tabular_privacy.ROLE_QUASI_IDENTIFIER, qi_kind="zip", explicit=True
    )
    assert not tabular_privacy.occurrence_decision(
        "90000", column=postal
    ).content_invariant


def test_declared_postal_schema_header_is_not_reinterpreted_as_phi_on_verification():
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "identifier_removal",
        "columns": [
            {
                "index": 0,
                "header": "postal_area",
                "role": "quasi_identifier",
                "qi_kind": "zip",
            },
            {"index": 1, "header": "note", "role": "clinical_utility"},
        ],
    }
    scanned = []

    def scanner(text, _provenance):
        scanned.append(text)
        return (
            [{"blocking": True, "category": "POSTAL_CODE", "detector": "stand_in"}]
            if text == "postal_area"
            else []
        )

    result = csv_sanitizer.anonymize_csv_bytes(
        b"postal_area,note\n90000,Stable follow-up\n",
        tabular_manifest=manifest,
        text_anonymizer=_mock_anonymizer,
        residual_scanner=scanner,
    )
    assert "postal_area" not in scanned
    assert "csv_residual_phi_detected" not in result.reason_codes
    assert result.summary.get("review_finding_count", 0) == 0


def test_fixed_identifiers_win_without_consuming_clinical_neighbours():
    direct = tabular_privacy.ColumnPolicy(
        0, tabular_privacy.ROLE_DIRECT_IDENTIFIER, entity_type="DEVICE_ID"
    )
    clinical = tabular_privacy.ColumnPolicy(1, tabular_privacy.ROLE_CLINICAL_UTILITY)
    assert (
        tabular_privacy.occurrence_decision(
            "SYN-DEV-910000", header="device_serial", column=direct
        ).action
        == tabular_privacy.ACTION_REPLACE
    )
    assert tabular_privacy.occurrence_decision(
        "nasogastric tube", header="device_type", column=clinical
    ).clinical_invariant
    assert (
        tabular_privacy.occurrence_decision("2026-01-01", header="patient_date").action
        == tabular_privacy.ACTION_REPLACE
    )
    assert tabular_privacy.occurrence_decision(
        "3 months", header="follow_up", column=clinical
    ).clinical_invariant
    assert tabular_privacy.occurrence_decision(
        "PCV20", header="vaccine", column=clinical
    ).clinical_invariant
    assert not tabular_privacy.occurrence_decision(
        "SYN-UNIQUE-880001",
        header="CODE",
        column=tabular_privacy.ColumnPolicy(2, tabular_privacy.ROLE_UNKNOWN),
    ).clinical_invariant


def test_clinical_phrase_does_not_exempt_an_entire_mixed_identifier_note():
    assert tabular_privacy.is_clinical_invariant("Serum or Plasma", "specimen_type")
    assert not tabular_privacy.is_clinical_invariant(
        "Serum or Plasma Jordan Fictional", "specimen_type"
    )
    assert not tabular_privacy.is_clinical_invariant("500 mg Jordan Fictional", "dose")
    assert not tabular_privacy.is_clinical_invariant(
        "Return in 3 months to Jordan Fictional", "follow_up"
    )


def test_validated_pathology_stain_is_exact_and_header_scoped():
    clinical = tabular_privacy.ColumnPolicy(0, tabular_privacy.ROLE_CLINICAL_UTILITY)
    direct = tabular_privacy.ColumnPolicy(
        0, tabular_privacy.ROLE_DIRECT_IDENTIFIER, entity_type="PERSON"
    )
    assert tabular_privacy.occurrence_decision(
        "H&E stain", header="pathology_stain", column=clinical
    ).clinical_invariant
    assert not tabular_privacy.occurrence_decision(
        "H&E stain Jordan Fictional", header="pathology_stain", column=clinical
    ).clinical_invariant
    assert (
        tabular_privacy.occurrence_decision(
            "H&E stain", header="patient_name", column=direct
        ).action
        == tabular_privacy.ACTION_REPLACE
    )


def test_declared_document_version_date_is_not_a_patient_date():
    nonpatient = tabular_privacy.ColumnPolicy(
        0, tabular_privacy.ROLE_NON_SENSITIVE, explicit=True
    )
    patient = tabular_privacy.ColumnPolicy(
        1, tabular_privacy.ROLE_DIRECT_IDENTIFIER, entity_type="DATE", explicit=True
    )
    source = "2026-09-30"
    assert tabular_privacy.occurrence_decision(
        source, header="document_version_date", column=nonpatient
    ).content_invariant
    assert not tabular_privacy.occurrence_decision(
        source, header="patient_date", column=nonpatient
    ).content_invariant
    assert (
        tabular_privacy.occurrence_decision(
            source, header="patient_date", column=patient
        ).action
        == tabular_privacy.ACTION_REPLACE
    )


def test_audit_provenance_binds_declared_headers_without_echoing_them():
    header = "document_version_date"
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "identifier_removal",
        "columns": [{"index": 0, "header": header, "role": "non_sensitive"}],
    }
    config = tabular_privacy.policy_config(manifest)
    columns = tabular_privacy.classify_columns([header], [["2026-09-30"]], config)
    summary = tabular_privacy.policy_summary(config, columns)
    assert header not in json.dumps(summary)
    assert (
        summary["manifest_columns"][0]["header_sha256"]
        == hashlib.sha256(header.encode()).hexdigest()
    )
    audited = tabular_privacy.config_from_metadata(summary)
    assert (
        len(tabular_privacy.config_for_sheet(audited, [header], primary=True).columns)
        == 1
    )
    with pytest.raises(tabular_privacy.ManifestError):
        tabular_privacy.config_for_sheet(audited, ["patient_date"], primary=True)


def test_ambient_profile_variables_cannot_select_tabular_policy(monkeypatch):
    monkeypatch.setenv("BIOBLOCK_TABULAR_PRIVACY_PROFILE", "safe_harbor_plus_qi")
    monkeypatch.setenv("BIOBLOCK_CSV_PRIVACY_PROFILE", "risk_based")
    monkeypatch.setenv("BIOBLOCK_TABULAR_MANIFEST_PATH", "missing-policy.json")
    assert (
        tabular_privacy.policy_config(allow_environment_manifest=False).profile
        == "identifier_removal"
    )
    assert (
        tabular_privacy.policy_config(
            profile="safe_harbor_plus_qi", allow_environment_manifest=False
        ).profile
        == "safe_harbor_plus_qi"
    )


def test_misleading_artifact_header_cannot_inherit_document_date_declaration():
    config = tabular_privacy.policy_config(
        {
            "policy_version": "tabular-privacy-v1",
            "profile": "identifier_removal",
            "columns": [
                {"index": 0, "header": "document_version_date", "role": "non_sensitive"}
            ],
        }
    )
    with pytest.raises(tabular_privacy.ManifestError):
        tabular_privacy.classify_columns(["patient_date"], [["2026-09-30"]], config)
    with pytest.raises(tabular_privacy.ManifestError):
        tabular_privacy.config_for_sheet(config, ["patient_date"], primary=True)


def test_csv_keeps_declared_document_date_but_removes_patient_date():
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "identifier_removal",
        "columns": [
            {"index": 0, "header": "document_version_date", "role": "non_sensitive"},
            {
                "index": 1,
                "header": "patient_date",
                "role": "direct_identifier",
                "entity_type": "DATE",
            },
        ],
    }
    body = (
        "document_version_date,patient_date\n"
        "2026-09-30,1984-03-07\n"
        "2026-09-30,1991-04-08\n"
    ).encode()
    result = csv_sanitizer.anonymize_csv_bytes(
        body,
        tabular_manifest=manifest,
        residual_scanner=_clean_scanner,
        text_anonymizer=_mock_anonymizer,
    )
    rows = list(csv.reader(io.StringIO(result.output_bytes.decode()), strict=True))
    assert [row[0] for row in rows[1:]] == ["2026-09-30"] * 2
    assert all(row[1] not in {"1984-03-07", "1991-04-08"} for row in rows[1:])


def test_workbook_and_audit_keep_only_the_declared_document_date(monkeypatch):
    from services import release_audit

    def identity(self, text):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    monkeypatch.setattr(workbook_sanitizer.TextChannelSanitizer, "_anonymize", identity)
    monkeypatch.setattr(
        workbook_sanitizer,
        "verify_extracted_text",
        lambda _texts, _tokens, **_kwargs: type(
            "Clean", (), {"passed": True, "blocking_categories": ()}
        )(),
    )
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "identifier_removal",
        "columns": [
            {"index": 0, "header": "document_version_date", "role": "non_sensitive"},
            {
                "index": 1,
                "header": "patient_date",
                "role": "direct_identifier",
                "entity_type": "DATE",
            },
        ],
    }
    workbook = openpyxl.Workbook()
    workbook.active.append(["document_version_date", "patient_date"])
    workbook.active.append(["2026-09-30", "1984-03-07"])
    workbook.active.append(["2026-09-30", "1991-04-08"])
    buffer = io.BytesIO()
    workbook.save(buffer)
    result = workbook_sanitizer.anonymize_workbook_bytes(
        buffer.getvalue(), tabular_manifest=manifest
    )
    reopened = openpyxl.load_workbook(io.BytesIO(result.output_bytes))
    try:
        assert [reopened.active.cell(row, 1).value for row in (2, 3)] == [
            "2026-09-30",
            "2026-09-30",
        ]
        assert all(
            reopened.active.cell(row, 2).value not in {"1984-03-07", "1991-04-08"}
            for row in (2, 3)
        )
        emitted = {reopened.active.cell(row, 2).value for row in (2, 3)}
    finally:
        reopened.close()

    def clean_scanner(_text):
        return ()

    clean_scanner.emitted_tokens = emitted
    audited = release_audit.audit_released_artifact(
        result.output_bytes,
        "workbook",
        text_scanner=clean_scanner,
        provenance={"tabular_privacy": result.summary["tabular_privacy"]},
    )
    assert audited.passed, audited.findings


def test_csv_emitted_token_ledger_distinguishes_source_fake_token_from_new_one():
    from services import release_audit

    body = (
        "patient_name,note,lab_value\n"
        "SyntheticPatient-000,Call <PATIENT_999>; PCV20,7.2%\n"
        "SyntheticPatient-001,Call <PATIENT_999>; PCV20,7.2%\n"
    ).encode()

    def identity(text, _profile, _allocator):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    result = csv_sanitizer.anonymize_csv_bytes(
        body, text_anonymizer=identity, residual_scanner=_clean_scanner
    )
    assert result.output_bytes is not None
    assert b"<PATIENT_999>" not in result.output_bytes
    assert result.generated_tokens
    assert all(
        token.encode() in result.output_bytes for token in result.generated_tokens
    )

    def clean(_text):
        return {}

    setattr(clean, release_audit.READS_EXACT_TEXT, True)
    setattr(clean, "emitted_tokens", result.generated_tokens)
    audit = release_audit.audit_released_artifact(
        result.output_bytes,
        "csv",
        text_scanner=clean,
        provenance={
            "tabular_privacy": result.summary.get("tabular_privacy"),
            "terminology_provenance": result.summary.get("terminology_provenance"),
        },
    )
    assert not any(
        item.reason_code == release_audit.REASON_CSV_ORIGINAL_IDENTIFIER
        for item in audit.findings
    )


def test_workbook_manifest_indices_do_not_retype_other_sheets(monkeypatch):
    def identity(self, text):
        return (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        )

    monkeypatch.setattr(workbook_sanitizer.TextChannelSanitizer, "_anonymize", identity)
    monkeypatch.setattr(
        workbook_sanitizer,
        "verify_extracted_text",
        lambda _texts, _tokens, **_kwargs: type(
            "Clean", (), {"passed": True, "blocking_categories": ()}
        )(),
    )
    manifest = {
        "policy_version": "tabular-privacy-v1",
        "profile": "identifier_removal",
        "columns": [
            {
                "index": 0,
                "header": "patient_name",
                "role": "direct_identifier",
                "entity_type": "PERSON",
            },
            {"index": 1, "header": "lab_value", "role": "clinical_utility"},
        ],
    }
    workbook = openpyxl.Workbook()
    visible = workbook.active
    visible.title = "Clinical Data"
    visible.append(["patient_name", "lab_value"])
    visible.append(["SyntheticPatient-000", "4417"])
    visible.append(["SyntheticPatient-001", "4417"])
    hidden = workbook.create_sheet("Hidden Notes")
    hidden.sheet_state = "hidden"
    hidden.append(["note", "lab_value"])
    hidden.append(["Clinical follow up", "4417"])
    vault = workbook.create_sheet("Very Hidden")
    vault.sheet_state = "veryHidden"
    vault.append(["provider_registration", "lab_value"])
    vault.append(["SYN-REG-310003", "4417"])
    stream = io.BytesIO()
    workbook.save(stream)
    result = workbook_sanitizer.anonymize_workbook_bytes(
        stream.getvalue(), tabular_manifest=manifest
    )
    assert result.output_bytes is not None
    reopened = openpyxl.load_workbook(io.BytesIO(result.output_bytes))
    try:
        assert reopened["Hidden Notes"]["A1"].value == "note"
        assert reopened["Hidden Notes"]["B1"].value == "lab_value"
        assert reopened["Hidden Notes"]["B2"].value == "4417"
        assert reopened["Very Hidden"]["A1"].value == "provider_registration"
        assert reopened["Very Hidden"]["B2"].value == "4417"
        assert reopened["Very Hidden"]["A2"].value != "SYN-REG-310003"
    finally:
        reopened.close()


def test_independent_surface_evaluator_follows_anonymized_hidden_sheet_name():
    from independent_xlsx import inspect
    from tabular_surface_evidence import _surface_evidence

    source = openpyxl.Workbook()
    source.active.title = "Clinical Data"
    hidden = source.create_sheet("Hidden Notes")
    hidden.sheet_state = "hidden"
    hidden["A2"] = "SyntheticPatient-002"
    original = io.BytesIO()
    source.save(original)
    hidden.title = "Sheet1"
    hidden["A2"] = "<PERSON_1>"
    written = io.BytesIO()
    source.save(written)
    protocol = {
        "occurrences": [
            {
                "surface": "hidden_cell",
                "sheet": "Hidden Notes",
                "cell": "A2",
                "value": "SyntheticPatient-002",
                "expected": "transformed",
            }
        ],
        "non_text_surfaces": [],
    }
    counts, exceptions, survivors, package_survivors = _surface_evidence(
        inspect(original.getvalue()), inspect(written.getvalue()), protocol
    )
    assert counts["transformed"] == 1
    assert not exceptions and not survivors and not package_survivors


def test_workbook_model_prefetch_keeps_separate_surface_inputs(monkeypatch):
    from services import model_client, request_deadline
    from services.multiformat import TextChannelSanitizer

    requests = []

    class Client:
        def detect_many(self, detector, items):
            requests.append(
                (detector, tuple((item["surface"], item["text"]) for item in items))
            )
            return [[] for _ in items]

    monkeypatch.setattr(model_client, "worker_enabled", lambda: True)
    monkeypatch.setattr(model_client, "get_client", lambda: Client())
    monkeypatch.setattr(
        TextChannelSanitizer,
        "_anonymize",
        lambda self, text: (
            {
                "anonymized_text": text,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            [],
        ),
    )
    request_deadline.start(30.0)
    try:
        sanitizer = TextChannelSanitizer()
        inputs = ["Jane in first note", "Jane in second note"]
        output = sanitizer.sanitize_isolated(inputs, ["sheet:A2", "sheet:B2"])
        assert [item.text for item in output] == inputs
        assert requests == [
            ("stanford", (("sheet:A2", inputs[0]), ("sheet:B2", inputs[1]))),
            ("gliner", (("sheet:A2", inputs[0]), ("sheet:B2", inputs[1]))),
        ]
    finally:
        request_deadline.clear()


def _occupation_pack(directory):
    """Minimal temporary counterpart of the source's generated occupation fixture."""
    headers = ["age", "occupation", "diagnosis"]
    rows = [
        [
            str(22 + i),
            "Nurse" if i < 3 or i > 4 else "Rare role",
            ("I10", "E11.9")[i % 2],
        ]
        for i in range(10)
    ]
    policy = {
        "policy_version": "tabular-privacy-v1",
        "profile": "safe_harbor_plus_qi",
        "k": 5,
        "l": 2,
        "t": None,
        "columns": [
            {"index": 0, "header": "age", "role": "quasi_identifier", "qi_kind": "age"},
            {
                "index": 1,
                "header": "occupation",
                "role": "quasi_identifier",
                "qi_kind": "occupation",
            },
            {"index": 2, "header": "diagnosis", "role": "sensitive_clinical"},
        ],
    }
    (directory / "qi_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    with (directory / "synthetic_hipaa18_clinical.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        csv.writer(stream).writerows([headers, *rows])
    book = openpyxl.Workbook()
    for row in [headers, *rows]:
        book.active.append(row)
    book.save(directory / "synthetic_hipaa18_clinical.xlsx")
    book.close()
    return directory
