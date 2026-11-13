"""CSV findings belong to isolated cells, and clinical values survive intact."""

import csv
import hashlib
import io
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIOBLOCK_STUDY_SALT", "csv-attribution-test-salt")

from services import csv_sanitizer as cs  # noqa: E402
from services.multiformat import TEXT_SEPARATOR  # noqa: E402


def _rows(result):
    return list(
        csv.reader(io.StringIO(result.output_bytes.decode("utf-8-sig"), newline=""))
    )


def _identity(flag=None):
    """A full_v1 stand-in that replaces nothing and raises flags on request.

    ``flag(text)`` returns (review reasons, residual categories) for a text.
    """
    seen = []

    def anonymize(text, profile, allocator):
        seen.append(text)
        review, residual = flag(text) if flag else ([], {})
        return {
            "anonymized_text": text,
            "detected_entities": {},
            "residual_phi_categories": residual,
            "review_required_reasons": review,
        }, []

    anonymize.seen = seen
    return anonymize


def _clean_scanner(text, provenance):
    return []


def _scanner(predicate, category="PERSON"):
    calls = []

    def scan(text, provenance):
        calls.append(text)
        return (
            [{"blocking": True, "category": category, "detector": "stand_in"}]
            if predicate(text)
            else []
        )

    scan.calls = calls
    return scan


NOTES = "note,diagnosis\n" + "".join(f"Visit note {index},I10\n" for index in range(64))


# -- verification rescans ----------------------------------------------------------------------


def test_a_finding_only_the_join_produces_is_not_a_file_finding():
    scanner = _scanner(lambda text: TEXT_SEPARATOR in text)

    result = cs.anonymize_csv_bytes(
        NOTES.encode(), text_anonymizer=_identity(), residual_scanner=scanner
    )

    assert cs.REASON_RESIDUAL not in result.reason_codes
    assert "review_findings" not in result.summary
    assert result.releasable, result.reason_codes


def test_a_finding_made_of_two_adjacent_cells_is_not_a_file_finding():
    # "Visit note 1" followed by "Visit note 2" reads as one span only joined.
    scanner = _scanner(
        lambda text: "Visit note 1" + TEXT_SEPARATOR + "Visit note 2" in text
    )

    result = cs.anonymize_csv_bytes(
        NOTES.encode(), text_anonymizer=_identity(), residual_scanner=scanner
    )

    assert cs.REASON_RESIDUAL not in result.reason_codes


def test_a_cell_that_holds_on_its_own_holds_the_file_with_its_coordinates():
    body = (
        "note,diagnosis\n"
        + "".join(f"Visit note {index},I10\n" for index in range(40))
        + "Seen by Quorvath,I10\n" * 3
    )
    scanner = _scanner(lambda text: "Quorvath" in text)

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=_identity(), residual_scanner=scanner
    )

    assert (
        cs.REASON_RESIDUAL in result.reason_codes
        and "residual_person" in result.reason_codes
    )
    findings = result.summary["review_findings"]
    assert result.summary["review_finding_count"] == 3
    assert sorted(item["row_index"] for item in findings) == [41, 42, 43]
    assert {item["column_id"] for item in findings} == {"column_0"}
    assert all(
        item["reason_code"] == cs.REASON_RESIDUAL
        and item["reviewable"]
        and item["category"] == "PERSON"
        and item["detector_source"] == "stand_in"
        for item in findings
    )
    # Value-safe: coordinates and codes, never the value.
    assert "Quorvath" not in repr(result.to_public_dict())
    assert not result.releasable


def test_attribution_scans_each_distinct_cell_without_cross_cell_context():
    scanner = _scanner(lambda text: "Visit note 37" in text)

    cs.anonymize_csv_bytes(
        NOTES.encode(), text_anonymizer=_identity(), residual_scanner=scanner
    )

    assert len(scanner.calls) == 65
    assert "Visit note 37" in scanner.calls
    assert all(TEXT_SEPARATOR not in text for text in scanner.calls)


def test_findings_at_a_batch_boundary_are_attributed(monkeypatch):
    monkeypatch.setattr(cs, "MAX_BATCH_FIELDS", 8)
    scanner = _scanner(
        lambda text: text.startswith("Visit note 7") or text.endswith("Visit note 8")
    )

    result = cs.anonymize_csv_bytes(
        NOTES.encode(), text_anonymizer=_identity(), residual_scanner=scanner
    )

    rows = {item["row_index"] for item in result.summary["review_findings"]}
    assert rows == {8, 9}


def test_a_structured_identifier_in_a_numeric_cell_is_attributed():
    body = "note,count\nstable,3\nstable,4\nstable,212-555-0199\n"

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=_identity(), residual_scanner=_clean_scanner
    )

    assert cs.REASON_RESIDUAL in result.reason_codes
    (finding,) = result.summary["review_findings"]
    assert (finding["row_index"], finding["column_id"]) == (3, "column_1")
    assert finding["detector_source"] == "structured_identifier_scan"


# -- first-pass flags --------------------------------------------------------------------------


def test_a_review_flag_only_the_join_raises_is_dropped_and_replacements_stand():
    anonymizer = _identity(
        lambda text: (
            (["ambiguous_person_requires_review"], {})
            if TEXT_SEPARATOR in text
            else ([], {})
        )
    )

    result = cs.anonymize_csv_bytes(
        NOTES.encode(), text_anonymizer=anonymizer, residual_scanner=_clean_scanner
    )

    assert cs.REASON_TEXT_REVIEW not in result.reason_codes
    assert "ambiguous_person_requires_review" not in result.reason_codes
    assert result.releasable


def test_a_review_flag_one_cell_raises_holds_with_that_cells_coordinates():
    body = NOTES + "Reviewed by Ostravel,I10\n" * 2
    anonymizer = _identity(
        lambda text: (
            (["ambiguous_person_requires_review"], {})
            if "Ostravel" in text
            else ([], {})
        )
    )

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=anonymizer, residual_scanner=_clean_scanner
    )

    assert {cs.REASON_TEXT_REVIEW, "ambiguous_person_requires_review"} <= set(
        result.reason_codes
    )
    findings = [
        item
        for item in result.summary["review_findings"]
        if item["reason_code"] == cs.REASON_TEXT_REVIEW
    ]
    assert sorted(item["row_index"] for item in findings) == [65, 66]
    assert {item["category"] for item in findings} == {"PERSON"}
    assert {item["detector_source"] for item in findings} == {
        "first_pass_evidence_model"
    }


def test_a_residual_flag_one_cell_raises_holds_with_that_cells_coordinates():
    body = NOTES + "Callback 212 555 0199,I10\n"
    anonymizer = _identity(
        lambda text: ([], {"PHONE_NUMBER": 1}) if "Callback" in text else ([], {})
    )

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=anonymizer, residual_scanner=_clean_scanner
    )

    assert {cs.REASON_ANONYMIZER_RESIDUAL, "residual_phone_number"} <= set(
        result.reason_codes
    )
    rows = {item["row_index"] for item in result.summary["review_findings"]}
    assert rows == {65}


def test_a_replacement_can_never_cross_a_cell_boundary():
    seen = []

    def crossing(text, profile, allocator):
        seen.append(text)
        replacements = []
        if TEXT_SEPARATOR in text:
            cut = text.index(TEXT_SEPARATOR)
            replacements = [
                (cut - 1, cut + len(TEXT_SEPARATOR) + 1, "PERSON", "<PERSON_9>")
            ]
        output = text
        for start, end, _kind, token in replacements:
            output = text[:start] + token + text[end:]
        return {
            "anonymized_text": output,
            "detected_entities": {},
            "residual_phi_categories": {},
            "review_required_reasons": [],
        }, replacements

    result = cs.anonymize_csv_bytes(
        NOTES.encode(), text_anonymizer=crossing, residual_scanner=_clean_scanner
    )

    assert "<PERSON_9>" not in result.output_bytes.decode()
    assert result.summary["batches_redone_per_cell"] == 0
    assert result.releasable
    assert len(seen) == 65
    assert all(TEXT_SEPARATOR not in text for text in seen)


# -- clinical and research utility -------------------------------------------------------------


def test_decimal_precision_is_not_an_identifier_but_a_long_integer_part_is():
    body = "healthcare_expenses,healthcare_coverage\n" + "".join(
        f"{1000 + index}.123456789012,{50 + index}.98765432109876\n"
        for index in range(12)
    )

    result = cs.anonymize_csv_bytes(body.encode(), residual_scanner=_clean_scanner)

    assert cs.REASON_AMBIGUOUS_NUMBER not in result.reason_codes
    assert result.output_bytes == body.encode()
    assert result.releasable, result.reason_codes
    for value in ("123456789.5", "123456789012", "1234567890.25"):
        held = cs.anonymize_csv_bytes(
            f"healthcare_expenses\n{value}\n".encode(), residual_scanner=_clean_scanner
        )
        assert cs.REASON_AMBIGUOUS_NUMBER in held.reason_codes, value


def test_titles_and_suffixes_are_kept_and_anything_else_there_is_replaced():
    body = "PREFIX,SUFFIX,diagnosis\nMrs.,PhD,I10\nMr.,Jr.,E11.9\nQuorvath,Ostrav,I10\n"
    anonymizer = _identity()

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=anonymizer, residual_scanner=_clean_scanner
    )

    rows = _rows(result)
    assert rows[0] == ["PREFIX", "SUFFIX", "diagnosis"]
    assert rows[1][:2] == ["Mrs.", "PhD"] and rows[2][:2] == ["Mr.", "Jr."]
    assert rows[3][0].startswith("<PERSON_") and rows[3][1].startswith("<PERSON_")
    assert "Quorvath" not in result.output_bytes.decode()
    # Neither a title nor a suffix is ever read by the text model.
    assert not {"Mrs.", "PhD", "Mr.", "Jr.", "PREFIX", "SUFFIX"} & set(anonymizer.seen)
    assert result.summary["coded_name_affix_cells"] == 4


def test_a_free_text_header_is_still_read_and_a_vocabulary_one_is_kept():
    anonymizer = _identity()
    body = "FIRST,notes by Quorvath,ZIP\nAlvenor,stable,02139\n"

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=anonymizer, residual_scanner=_clean_scanner
    )

    read = TEXT_SEPARATOR.join(anonymizer.seen)
    assert "notes by Quorvath" in read
    assert "FIRST" not in read and "ZIP" not in read
    assert result.summary["vocabulary_header_cells"] == 2


def test_race_ethnicity_and_state_labels_are_kept_for_the_dataset_policy():
    anonymizer = _identity()
    body = "RACE,ETHNICITY,STATE,note\n" + "".join(
        f"{race},{ethnicity},{state},visit {index}\n"
        for index, (race, ethnicity, state) in enumerate(
            [
                ("white", "nonhispanic", "Massachusetts"),
                ("black", "hispanic", "MA"),
                ("asian", "nonhispanic", "New York"),
            ]
            * 4
        )
    )

    result = cs.anonymize_csv_bytes(
        body.encode(),
        text_anonymizer=anonymizer,
        residual_scanner=_scanner(lambda text: "Massachusetts" in text),
    )

    read = TEXT_SEPARATOR.join(anonymizer.seen)
    for label in ("white", "nonhispanic", "Massachusetts", "New York"):
        assert label not in read
    rows = _rows(result)
    # Kept, or generalized by the dataset policy - never rewritten by the model -
    # and never rescanned by it either: "Massachusetts" alone is not a finding.
    assert {row[2] for row in rows[1:]} <= {"Massachusetts", "MA", "New York", "*"}
    assert cs.REASON_RESIDUAL not in result.reason_codes
    assert result.summary["categorical_quasi_identifier_cells"] == 36


def test_a_name_in_a_race_column_is_still_read_as_text():
    anonymizer = _identity()
    body = "RACE,note\nwhite,a\nAlvenor Quorvath,b\n"

    result = cs.anonymize_csv_bytes(
        body.encode(),
        text_anonymizer=anonymizer,
        residual_scanner=_scanner(lambda text: "Quorvath" in text),
    )

    assert "Alvenor Quorvath" in TEXT_SEPARATOR.join(anonymizer.seen)
    assert cs.REASON_RESIDUAL in result.reason_codes
    assert {item["row_index"] for item in result.summary["review_findings"]} == {2}


def test_a_death_date_header_is_a_date_column():
    for header in ("DEATHDATE", "date_of_death", "DateOfBirth"):
        assert cs.classify_header(header).policy == cs.POLICY_DATE, header


def _synthea_like(rows=40, seed=7):
    """The shape of a Synthea patients table, with invented values."""
    generator = random.Random(seed)
    header = [
        "Id",
        "BIRTHDATE",
        "DEATHDATE",
        "SSN",
        "DRIVERS",
        "PASSPORT",
        "PREFIX",
        "FIRST",
        "LAST",
        "SUFFIX",
        "MAIDEN",
        "MARITAL",
        "RACE",
        "ETHNICITY",
        "GENDER",
        "BIRTHPLACE",
        "ADDRESS",
        "CITY",
        "STATE",
        "COUNTY",
        "ZIP",
        "LAT",
        "LON",
        "HEALTHCARE_EXPENSES",
        "HEALTHCARE_COVERAGE",
    ]
    firsts = ["Alvenor", "Brisk", "Corvane", "Dellith", "Esmor"]
    lasts = ["Quorvath", "Ostravel", "Pennrick", "Tallowin"]
    lines = [",".join(header)]
    for index in range(rows):
        expenses = f"{generator.randint(1000, 1999999)}.{generator.randint(10 ** 11, 10 ** 12)}"
        coverage = (
            f"{generator.randint(0, 99999)}.{generator.randint(10 ** 12, 10 ** 13)}"
        )
        death = "" if index % 5 else f"20{10 + index % 9}-0{1 + index % 8}-1{index % 9}"
        lines.append(
            ",".join(
                [
                    f"{index:08x}-aaaa-4bbb-8ccc-{index:012x}",
                    f"19{50 + index % 40}-0{1 + index % 9}-2{index % 8}",
                    death,
                    f"999-{10 + index % 80}-{1000 + index}",
                    f"S999{index:05d}",
                    "" if index % 3 else f"X{index:08d}X",
                    "Mrs." if index % 2 else "Mr.",
                    f"{firsts[index % 5]}{index}",
                    f"{lasts[index % 4]}{index}",
                    "PhD" if index == 3 else "",
                    "" if index % 4 else f"{lasts[(index + 1) % 4]}",
                    "M" if index % 2 else "S",
                    "white" if index % 3 else "black",
                    "nonhispanic" if index % 4 else "hispanic",
                    "F" if index % 2 else "M",
                    f"Zorvale Town{index % 3}  Massachusetts  US",
                    f"{100 + index} Fenwick Row",
                    f"Zorvale{index % 3}",
                    "Massachusetts",
                    f"Hollin County{index % 2}",
                    "" if index % 2 else f"0{2100 + index % 7}",
                    f"42.{1000000 + index}",
                    f"-71.{2000000 + index}",
                    expenses,
                    coverage,
                ]
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_a_synthea_shaped_table_keeps_its_structure_and_its_clinical_values():
    payload = _synthea_like()
    source = list(csv.reader(io.StringIO(payload.decode(), newline="")))

    result = cs.anonymize_csv_bytes(payload, residual_scanner=_clean_scanner)

    output = _rows(result)
    header = source[0]
    column = {name: index for index, name in enumerate(header)}
    # Structure: same rows, columns, order, header, line endings and quoting.
    assert len(output) == len(source) and all(len(row) == 25 for row in output)
    assert output[0] == header
    assert (
        b"\r\n" not in result.output_bytes
        and result.summary["round_trip_exact"] is True
    )
    # Clinical and research values survive exactly, digit for digit.
    for name in ("HEALTHCARE_EXPENSES", "HEALTHCARE_COVERAGE"):
        assert [row[column[name]] for row in output] == [
            row[column[name]] for row in source
        ]
    assert cs.REASON_AMBIGUOUS_NUMBER not in result.reason_codes
    # Direct identifiers never survive.
    text = result.output_bytes.decode()
    for name in (
        "Id",
        "SSN",
        "DRIVERS",
        "PASSPORT",
        "FIRST",
        "LAST",
        "MAIDEN",
        "ADDRESS",
        "LAT",
        "LON",
    ):
        for original, released in zip(source[1:], output[1:]):
            if original[column[name]]:
                assert released[column[name]] != original[column[name]], name
                assert original[column[name]] not in text, name
    # Nulls stay where they were for the columns kept as they are.
    for name in ("SUFFIX", "HEALTHCARE_EXPENSES"):
        assert [bool(row[column[name]]) for row in output] == [
            bool(row[column[name]]) for row in source
        ]
    # Titles and suffixes are vocabulary and kept; dates never keep day and month.
    assert {row[column["PREFIX"]] for row in output[1:]} <= {"Mr.", "Mrs.", "", "*"}
    for original, released in zip(source[1:], output[1:]):
        for name in ("BIRTHDATE", "DEATHDATE"):
            if original[column[name]]:
                assert released[column[name]] != original[column[name]], name


def test_the_same_table_twice_gives_the_same_bytes():
    payload = _synthea_like(rows=25, seed=11)

    first = cs.anonymize_csv_bytes(payload, residual_scanner=_clean_scanner)
    second = cs.anonymize_csv_bytes(payload, residual_scanner=_clean_scanner)

    assert first.output_bytes == second.output_bytes
    assert (
        hashlib.sha256(first.output_bytes).hexdigest()
        == hashlib.sha256(second.output_bytes).hexdigest()
    )


@pytest.mark.parametrize(
    "body, check",
    [
        # An identifier beside a clinical measurement: replaced, measurement kept.
        (
            "mrn,systolic_bp\nMRN-4401928,128\nMRN-4401929,131\n",
            lambda rows: rows[1][1] == "128" and "4401928" not in rows[1][0],
        ),
        # A postal code beside decimals.
        (
            "zip,creatinine\n02139,0.91\n02140,1.07\n",
            lambda rows: [row[1] for row in rows[1:]] == ["0.91", "1.07"]
            and rows[1][0] != "02139",
        ),
        # A name beside a coded value.
        (
            "patient_name,diagnosis\nAlvenor Quorvath,J45.909\n",
            lambda rows: rows[1][1] == "J45.909" and "Quorvath" not in rows[1][0],
        ),
        # A numeric record id beside laboratory results.
        (
            "record id,hba1c_percent,sodium\n77441029,6.8,139\n",
            lambda rows: rows[1][1:] == ["6.8", "139"] and rows[1][0] != "77441029",
        ),
        # Quoted commas and newlines survive as one cell.
        (
            'note,dose_mg\n"stable, repeat\nin 3 months",500\n',
            lambda rows: rows[1] == ["stable, repeat\nin 3 months", "500"],
        ),
        # Unicode clinical text.
        (
            "note,weight_kg\nPatientin stabil, Blutdruck gut eingestellt,70.5\n".replace(
                "stabil, Blutdruck", "stabil; Blutdruck"
            ),
            lambda rows: rows[1][1] == "70.5",
        ),
        # Empty cells and duplicate rows keep their places.
        (
            "diagnosis,weight_kg,note\nI10,,\nI10,,\n,72.0,\n",
            lambda rows: rows[1:]
            == [["I10", "", ""], ["I10", "", ""], ["", "72.0", ""]],
        ),
        # A mixed-type column: codes and measurements side by side.
        (
            "result,units\n5.4,mmol/L\n<0.5,mg/dL\nnegative,\n",
            lambda rows: [row[0] for row in rows[1:]] == ["5.4", "<0.5", "negative"],
        ),
    ],
)
def test_adversarial_tables_keep_their_clinical_content(body, check):
    # The text model is a stand-in that changes nothing: what is checked is
    # the table itself - cells, quoting, nulls and header-driven replacement.
    result = cs.anonymize_csv_bytes(
        body.encode("utf-8"),
        text_anonymizer=_identity(),
        residual_scanner=_clean_scanner,
    )

    rows = _rows(result)
    assert len(rows) == len(list(csv.reader(io.StringIO(body, newline=""))))
    assert check(rows), rows


def test_a_long_cell_is_attributed_as_one_cell():
    long_note = "stable " * 3000 + "Quorvath"
    body = f'note,weight_kg\n"{long_note}",70.5\nfine,71.0\n'
    scanner = _scanner(lambda text: "Quorvath" in text)

    result = cs.anonymize_csv_bytes(
        body.encode(), text_anonymizer=_identity(), residual_scanner=scanner
    )

    assert {item["row_index"] for item in result.summary["review_findings"]} == {1}
    assert _rows(result)[2] == ["fine", "71.0"]


def test_unicode_names_are_still_found_in_their_own_cells():
    result = cs.anonymize_csv_bytes(
        "patient_name,diagnosis\nŽofia Ąlvenör,I10\n".encode(),
        residual_scanner=_clean_scanner,
    )

    rows = _rows(result)
    assert rows[1][0].startswith("<PERSON_") and rows[1][1] == "I10"


# -- the final audit of a streamed CSV names the cells it objects to ------------------------------


def test_streamed_csv_audit_finding_is_attributed_to_its_cells(tmp_path):

    source = tmp_path / "source.csv"
    source.write_text(
        "note,diagnosis\nStable visit,I10\nUnclear review wording,I10\n"
        "Unclear review wording,I10\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()

    def audit_scanner(text):
        return ["PERSON"] if "Unclear" in text else []

    setattr(audit_scanner, "reads_exact_text", True)
    result = _stream_csv(
        str(source),
        "strict",
        work,
        {
            "text_anonymizer": _identity(),
            "residual_scanner": _clean_scanner,
            "audit_scanner": audit_scanner,
        },
        owner=None,
    )
    audit = result["summary"]["release_audit"]
    assert audit["passed"] is False
    assert audit["findings"]
    located = {
        item["location"]
        for item in audit["finding_locations"]
        if item["category"] == "PERSON"
    }
    # Both physical occurrences are named; no cell value is serialized.
    assert located == {"row:2:column:0", "row:3:column:0"}
    assert "Unclear" not in repr(audit)


def test_streamed_csv_audit_trusts_only_the_tokens_this_run_emitted(tmp_path):
    """A clean, releasable stream must not call its own surrogates original identifiers."""
    import re

    source = tmp_path / "source.csv"
    rows = "".join(f"Seen with Zelquin Vantreske{n};,I10" + chr(10) for n in range(6))
    source.write_text("note,diagnosis" + chr(10) + rows, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    def replace_name(text, profile, allocator):
        found = [
            (m.start(), m.end(), "PERSON", allocator.surrogate_for("PERSON", m.group()))
            for m in re.finditer(r"Zelquin Vantreske\d", text)
        ]
        output = text
        for start, end, _entity, token in reversed(found):
            output = output[:start] + token + output[end:]
        return (
            {
                "anonymized_text": output,
                "detected_entities": {},
                "residual_phi_categories": {},
                "review_required_reasons": [],
            },
            found,
        )

    result = _stream_csv(
        str(source),
        "strict",
        work,
        {
            "text_anonymizer": replace_name,
            "residual_scanner": _clean_scanner,
            "finding_scanner": _clean_scanner,
        },
        owner=None,
    )
    reasons = {
        item["reason_code"] for item in result["summary"]["release_audit"]["findings"]
    }
    assert "audit_csv_original_identifier_present" not in reasons, result["summary"][
        "release_audit"
    ]


def _stream_csv(source, profile, work, options, owner):
    import secrets
    import shutil
    from types import SimpleNamespace

    from services.stream_ingestion import Pipeline

    shutil.copyfile(source, work / "input")
    pipeline = Pipeline()
    try:
        outcome = pipeline.work(
            SimpleNamespace(job_id=secrets.token_urlsafe(24), owner=owner),
            work,
            "csv",
            profile,
            options=options,
        )
        return outcome.result
    finally:
        pipeline.shutdown()
