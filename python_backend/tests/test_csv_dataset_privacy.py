"""The CSV sanitizer applies the dataset-level privacy policy after direct identifiers.

A table with quasi-identifiers is released only when the written file itself,
reopened, meets k = 5, l = 2 and t = 0.2; otherwise it is held. Measurements
and diagnoses are never changed. Every value is invented.
"""

import csv
import io
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import csv_sanitizer  # noqa: E402
from services import dataset_privacy as privacy  # noqa: E402
from services.csv_sanitizer import (
    anonymize_csv_bytes as _anonymize_csv_bytes,
)  # noqa: E402

HEADER = [
    "patient_name",
    "age",
    "admission_date",
    "sex",
    "zip",
    "diagnosis",
    "systolic_bp",
]
FIRST = ("Zelquin", "Marbrint", "Ilsaveth", "Orvenne", "Thessaly", "Quorvan")
LAST = (
    "Vantreske",
    "Morrowgale",
    "Quennetholm",
    "Harrowdale",
    "Ostrivane",
    "Pellamere",
)
DIAGNOSES = ("J45.909", "E11.9", "I10", "N18.3")
ZIPS = ("02139", "02141", "10027", "60637", "94305", "98105")


def anonymize_csv_bytes(*args, **kwargs):
    """Bind the legacy alias explicitly; ambient process state cannot choose it."""
    kwargs.setdefault("tabular_profile", "risk_based")
    return _anonymize_csv_bytes(*args, **kwargs)


def _quiet_scanner(text, provenance):
    # The local test NER reads short header labels such as "zip" as
    # organizations depending on the cells batched beside them. The residual
    # rescan has its own tests; these release tests are about k, l and t.
    return []


def _table(rows, header=HEADER):
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows([header, *rows])
    return buffer.getvalue().encode("utf-8")


def _people(size, seed=7):
    rng = random.Random(seed)
    return [
        [
            f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            str(rng.randint(18, 89)),
            str(rng.randint(2016, 2023)),
            rng.choice("FM"),
            rng.choice(ZIPS),
            DIAGNOSES[index % len(DIAGNOSES)],
            str(rng.randint(95, 180)),
        ]
        for index in range(size)
    ]


def _reopened(result):
    return list(
        csv.reader(io.StringIO(result.output_bytes.decode("utf-8"), newline=""))
    )


def _roles(header=HEADER):
    return privacy.classify_columns(
        [csv_sanitizer.header_tokens(cell) for cell in header],
        [csv_sanitizer.classify_header(cell) for cell in header],
    )


def test_a_table_is_released_only_when_its_written_file_meets_k_l_and_t():
    people = _people(40)

    result = anonymize_csv_bytes(_table(people), residual_scanner=_quiet_scanner)

    assert result.releasable, result.reason_codes
    rows = _reopened(result)
    assert rows[0] == HEADER and len(rows) == 41
    check = privacy.verify_dataset(rows[1:], _roles(), is_null=csv_sanitizer._is_null)
    assert check.satisfied and check.k >= 5 and check.l >= 2 and check.t <= 0.2
    assert [row[6] for row in rows[1:]] == [person[6] for person in people]
    output = result.output_bytes.decode("utf-8")
    assert not any(last in output for last in LAST)
    assert not any(zip_code in output for zip_code in ZIPS)
    assert result.summary["dataset_privacy_verified_on_output"] is True
    assert result.summary["k_achieved"] >= 5


def test_the_dataset_policy_changes_only_quasi_identifier_columns(monkeypatch):
    body = _table(_people(40))

    with_policy = _reopened(anonymize_csv_bytes(body, residual_scanner=_quiet_scanner))
    from services import csv_stream

    monkeypatch.setattr(csv_stream._Engine, "dataset_privacy", lambda engine: None)
    without_policy = _reopened(
        anonymize_csv_bytes(body, residual_scanner=_quiet_scanner)
    )

    # Diagnoses feed l and t, and measurements are protected: neither changes.
    changed = {
        column
        for old, new in zip(without_policy, with_policy)
        for column, (before, after) in enumerate(zip(old, new))
        if before != after
    }
    assert changed and changed <= {1, 2, 3, 4}


def test_fewer_than_five_rows_with_quasi_identifiers_are_held():
    result = anonymize_csv_bytes(_table(_people(4)))

    assert not result.releasable
    assert privacy.REASON_TOO_FEW_ROWS in result.reason_codes


def test_a_small_table_without_quasi_identifiers_is_still_released():
    result = anonymize_csv_bytes(
        _table([["I10", "128"], ["E11.9", "141"]], ["diagnosis", "systolic_bp"])
    )

    assert result.releasable, result.reason_codes
    assert "k_achieved" not in result.summary


def test_a_table_whose_diagnoses_can_never_be_diverse_is_held():
    people = _people(30)
    for person in people:
        person[5] = "I10"

    result = anonymize_csv_bytes(_table(people))

    assert not result.releasable
    assert privacy.REASON_IMPOSSIBLE in result.reason_codes


def test_an_unreadable_quasi_identifier_is_held():
    people = _people(30)
    people[3][1] = "about forty"

    result = anonymize_csv_bytes(_table(people))

    assert not result.releasable
    assert privacy.REASON_AMBIGUOUS in result.reason_codes


def test_generalized_years_are_released_as_ranges_not_mistaken_for_identifiers():
    people = [
        [
            "Zelquin Vantreske",
            "50",
            str(2010 + index),
            "F",
            "02139",
            DIAGNOSES[index % 2],
            "120",
        ]
        for index in range(20)
    ]

    result = anonymize_csv_bytes(_table(people), residual_scanner=_quiet_scanner)

    assert result.releasable, result.reason_codes
    years = [row[2] for row in _reopened(result)[1:]]
    assert "2010-2014" in years and len(set(years)) >= 2
    assert result.summary["cells_generalized"] >= 15


def test_a_written_file_that_fails_the_second_check_is_never_releasable(monkeypatch):
    real = privacy.verify_dataset
    calls = []

    def second_check_fails(*args, **kwargs):
        calls.append(1)
        check = real(*args, **kwargs)
        if len(calls) == 1:
            return check
        return privacy.DatasetCheck(
            False, check.rows, check.equivalence_classes, 4, check.l, check.t
        )

    monkeypatch.setattr(privacy, "verify_dataset", second_check_fails)

    result = anonymize_csv_bytes(_table(_people(40)), residual_scanner=_quiet_scanner)

    assert len(calls) == 2
    assert not result.releasable
    assert privacy.REASON_NOT_VERIFIED in result.errors
