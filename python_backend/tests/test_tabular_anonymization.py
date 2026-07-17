import csv
import io
import json
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tabular_anonymization import (  # noqa: E402
    TabularAnonymizationError,
    anonymize_tabular_csv,
)


SYNTHETIC_CSV = """name,email,phone,mrn,age,gender,admission_date,diagnosis
Alice Adams,alice@example.com,555-111-2222,MRN-001,31,F,2024-01-02,diabetes
Bob Baker,bob@example.com,555-111-3333,MRN-002,32,F,2024-01-15,asthma
Carol Chen,carol@example.com,555-111-4444,MRN-003,33,M,2024-02-02,diabetes
Dan Diaz,dan@example.com,555-111-5555,MRN-004,34,M,2024-02-20,asthma
Eve Evans,eve@example.com,555-111-6666,MRN-005,61,F,2025-03-01,cancer
Frank Fox,frank@example.com,555-111-7777,MRN-006,62,F,2025-03-15,flu
Gina Gray,gina@example.com,555-111-8888,MRN-007,63,M,2025-04-01,cancer
Hank Hill,hank@example.com,555-111-9999,MRN-008,64,M,2025-04-15,flu
""".encode("utf-8")

RAW_VALUES = (
    "Alice Adams",
    "alice@example.com",
    "555-111-2222",
    "MRN-001",
    "diabetes",
    "asthma",
    "cancer",
    "flu",
)


def internal_rows(result):
    return list(csv.DictReader(io.StringIO(result["_internal_anonymized_csv"])))


def test_direct_identifier_columns_are_removed_and_summary_is_safe():
    result = anonymize_tabular_csv(SYNTHETIC_CSV, k=2, l=2)

    assert result["anonymization_status"] == "completed"
    assert result["direct_identifiers_removed"] == ["name", "email", "phone", "mrn"]
    response_text = json.dumps(result)
    assert "_internal_anonymized_csv" not in result
    for raw_value in RAW_VALUES:
        assert raw_value not in response_text


def test_numeric_date_and_categorical_quasi_identifiers_are_generalized():
    result = anonymize_tabular_csv(
        SYNTHETIC_CSV,
        k=2,
        l=2,
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)

    assert rows[0]["age"] == "31-32"
    assert rows[1]["age"] == "31-32"
    assert rows[0]["admission_date"] == "2024-01"
    assert rows[2]["admission_date"] == "2024-02"
    assert all(row["gender"] == "*" for row in rows)
    assert result["generalized_cells_count"] > 0
    assert result["suppressed_cells_count"] >= len(rows)


def test_direct_identifier_columns_are_absent_from_internal_anonymized_csv():
    result = anonymize_tabular_csv(
        SYNTHETIC_CSV,
        k=2,
        l=2,
        include_anonymized_csv=True,
    )
    rows = internal_rows(result)

    assert rows
    assert "name" not in rows[0]
    assert "email" not in rows[0]
    assert "phone" not in rows[0]
    assert "mrn" not in rows[0]


def test_every_equivalence_class_satisfies_k_when_possible():
    result = anonymize_tabular_csv(SYNTHETIC_CSV, k=2, l=2)

    assert result["k_anonymity_satisfied"] is True
    assert result["min_group_size"] >= 2
    assert result["equivalence_classes"] == 4


def test_l_diversity_passes_when_possible():
    result = anonymize_tabular_csv(SYNTHETIC_CSV, k=2, l=2)

    assert result["sensitive_column"] == "diagnosis"
    assert result["l_diversity_satisfied"] is True


def test_l_diversity_failure_is_reported_clearly():
    csv_bytes = b"name,age,gender,diagnosis\nA,40,F,flu\nB,41,F,flu\nC,42,M,flu\nD,43,M,flu\n"

    result = anonymize_tabular_csv(csv_bytes, k=2, l=2)

    assert result["anonymization_status"] == "completed"
    assert result["k_anonymity_satisfied"] is True
    assert result["l_diversity_satisfied"] is False
    assert any("l-diversity" in warning for warning in result["warnings"])


def test_empty_csv_is_rejected():
    with pytest.raises(TabularAnonymizationError) as exc:
        anonymize_tabular_csv(b"  \n")

    assert exc.value.status_code == 400
    assert "empty" in exc.value.detail.lower()


def test_invalid_csv_is_rejected():
    with pytest.raises(TabularAnonymizationError) as exc:
        anonymize_tabular_csv(b'age,diagnosis\n"42,flu\n')

    assert exc.value.status_code == 400
    assert "invalid csv" in exc.value.detail.lower()


def test_missing_quasi_identifier_column_is_rejected_clearly():
    with pytest.raises(TabularAnonymizationError) as exc:
        anonymize_tabular_csv(
            b"diagnosis\nflu\n",
            quasi_identifiers=["age"],
        )

    assert exc.value.status_code == 400
    assert "Missing quasi-identifier column" in exc.value.detail


def test_missing_sensitive_column_without_explicit_request_is_not_applicable():
    csv_bytes = b"name,age,gender\nA,30,F\nB,31,F\nC,32,M\nD,33,M\nE,34,F\n"

    result = anonymize_tabular_csv(csv_bytes, k=5, l=2)

    assert result["sensitive_column"] is None
    assert result["l_diversity_satisfied"] == "not_applicable"
    assert any("not evaluated" in warning for warning in result["warnings"])


def test_explicit_missing_sensitive_column_is_rejected_clearly():
    with pytest.raises(TabularAnonymizationError) as exc:
        anonymize_tabular_csv(
            b"age,diagnosis\n40,flu\n41,cold\n",
            sensitive_column="outcome",
        )

    assert exc.value.status_code == 400
    assert exc.value.detail == "Missing sensitive column: outcome"


def test_default_response_does_not_expose_raw_row_values():
    result = anonymize_tabular_csv(SYNTHETIC_CSV, k=2, l=2)
    response_text = json.dumps(result)

    assert "_internal_anonymized_csv" not in result
    for raw_value in RAW_VALUES:
        assert raw_value not in response_text
