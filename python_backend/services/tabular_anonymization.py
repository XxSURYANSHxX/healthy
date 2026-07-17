from __future__ import annotations

import csv
import io
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


UNKNOWN_VALUE = "UNKNOWN"
SUPPRESSED_VALUE = "*"

DEFAULT_DIRECT_IDENTIFIER_COLUMNS = (
    "name",
    "full_name",
    "email",
    "phone",
    "mobile",
    "mrn",
    "medical_record_number",
    "patient_id",
    "ssn",
    "address",
)

DEFAULT_QUASI_IDENTIFIER_COLUMNS = (
    "age",
    "gender",
    "sex",
    "zip",
    "zip_code",
    "postal_code",
    "city",
    "state",
    "date",
    "admission_date",
    "discharge_date",
    "diagnosis_date",
)

DEFAULT_SENSITIVE_COLUMNS = (
    "diagnosis",
    "disease",
    "condition",
    "outcome",
)

DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%Y%m%d",
    "%Y-%m",
)


class TabularAnonymizationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


def anonymize_tabular_csv(
    file_bytes: bytes,
    k: int = 5,
    l: int = 2,
    direct_identifiers: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
    sensitive_column: Optional[str] = None,
    include_anonymized_csv: bool = False,
) -> Dict[str, Any]:
    _validate_thresholds(k, l)
    header, records = _read_csv(file_bytes)
    normalized_columns = _build_normalized_column_map(header)

    direct_identifier_columns = _resolve_column_list(
        available_columns=normalized_columns,
        requested=direct_identifiers,
        defaults=DEFAULT_DIRECT_IDENTIFIER_COLUMNS,
        column_kind="direct identifier",
    )
    sensitive_column_name = _resolve_sensitive_column(
        available_columns=normalized_columns,
        requested=sensitive_column,
    )
    quasi_identifier_columns = _resolve_column_list(
        available_columns=normalized_columns,
        requested=quasi_identifiers,
        defaults=DEFAULT_QUASI_IDENTIFIER_COLUMNS,
        column_kind="quasi-identifier",
    )

    overlap = set(direct_identifier_columns).intersection(quasi_identifier_columns)
    if overlap:
        raise TabularAnonymizationError(
            "Columns cannot be both direct identifiers and quasi-identifiers: "
            + ", ".join(sorted(overlap))
        )
    if sensitive_column_name and sensitive_column_name in direct_identifier_columns:
        raise TabularAnonymizationError(
            "Sensitive column cannot also be removed as a direct identifier"
        )

    retained_columns = [
        column for column in header if column not in set(direct_identifier_columns)
    ]
    retained_column_set = set(retained_columns)
    missing_quasi_after_removal = [
        column for column in quasi_identifier_columns if column not in retained_column_set
    ]
    if missing_quasi_after_removal:
        raise TabularAnonymizationError(
            "Missing quasi-identifier column(s): "
            + ", ".join(missing_quasi_after_removal)
        )

    working_rows = _remove_direct_identifiers(
        records=records,
        retained_columns=retained_columns,
    )
    quasi_identifier_types = {
        column: _infer_column_type(working_rows, column)
        for column in quasi_identifier_columns
    }

    partitions = _mondrian_partitions(
        rows=working_rows,
        quasi_identifier_types=quasi_identifier_types,
        k=k,
        l=l,
        sensitive_column=sensitive_column_name,
    )
    anonymized_rows, generalized_cells, suppressed_cells = _generalize_partitions(
        rows=working_rows,
        partitions=partitions,
        quasi_identifier_types=quasi_identifier_types,
    )

    equivalence_groups = _equivalence_groups(
        rows=anonymized_rows,
        quasi_identifiers=quasi_identifier_columns,
    )
    group_sizes = [len(indices) for indices in equivalence_groups.values()]
    min_group_size = min(group_sizes) if group_sizes else 0
    k_anonymity_satisfied = bool(group_sizes) and min_group_size >= k

    warnings: List[str] = []
    if not k_anonymity_satisfied:
        warnings.append(
            "k-anonymity could not be satisfied; minimum equivalence class "
            f"size is {min_group_size}."
        )

    if sensitive_column_name is None:
        l_diversity_satisfied: Any = "not_applicable"
        warnings.append(
            "l-diversity was not evaluated because no sensitive column was "
            "provided or detected."
        )
    else:
        l_diversity_satisfied = _l_diversity_satisfied(
            rows=anonymized_rows,
            equivalence_groups=equivalence_groups,
            sensitive_column=sensitive_column_name,
            l=l,
        )
        if not l_diversity_satisfied:
            warnings.append(
                "l-diversity could not be satisfied for at least one "
                "equivalence class."
            )

    result: Dict[str, Any] = {
        "anonymization_status": "completed",
        "rows_in": len(records),
        "rows_out": len(anonymized_rows),
        "k": k,
        "l": l,
        "direct_identifiers_removed": direct_identifier_columns,
        "quasi_identifiers_used": quasi_identifier_columns,
        "sensitive_column": sensitive_column_name,
        "equivalence_classes": len(equivalence_groups),
        "min_group_size": min_group_size,
        "k_anonymity_satisfied": k_anonymity_satisfied,
        "l_diversity_satisfied": l_diversity_satisfied,
        "generalized_cells_count": generalized_cells,
        "suppressed_cells_count": suppressed_cells,
        "warnings": warnings,
    }

    if include_anonymized_csv:
        result["_internal_anonymized_csv"] = _write_csv(retained_columns, anonymized_rows)

    return result


def _validate_thresholds(k: int, l: int) -> None:
    if not isinstance(k, int) or k < 1:
        raise TabularAnonymizationError("k must be a positive integer")
    if not isinstance(l, int) or l < 1:
        raise TabularAnonymizationError("l must be a positive integer")


def _read_csv(file_bytes: bytes) -> Tuple[List[str], List[Dict[str, str]]]:
    if not file_bytes:
        raise TabularAnonymizationError("CSV input is empty")

    try:
        csv_text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise TabularAnonymizationError("CSV uploads must be UTF-8 encoded") from exc

    if not csv_text.strip():
        raise TabularAnonymizationError("CSV input is empty")

    try:
        reader = csv.reader(io.StringIO(csv_text), strict=True)
        raw_rows = [row for row in reader if any(cell.strip() for cell in row)]
    except csv.Error as exc:
        raise TabularAnonymizationError("Invalid CSV format") from exc

    if not raw_rows:
        raise TabularAnonymizationError("CSV input is empty")

    header = [cell.strip() for cell in raw_rows[0]]
    if not header or not any(header):
        raise TabularAnonymizationError("CSV header row is missing")
    if any(not column for column in header):
        raise TabularAnonymizationError("CSV header row contains empty column names")

    normalized_header = [_normalize_column_name(column) for column in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise TabularAnonymizationError("CSV column names must be unique")

    if len(raw_rows) == 1:
        raise TabularAnonymizationError("CSV input must include at least one data row")

    records: List[Dict[str, str]] = []
    expected_columns = len(header)
    for row_number, row in enumerate(raw_rows[1:], start=2):
        if len(row) != expected_columns:
            raise TabularAnonymizationError(
                f"CSV row {row_number} has {len(row)} fields; "
                f"expected {expected_columns}"
            )
        records.append(
            {
                column: _clean_cell(value)
                for column, value in zip(header, row)
            }
        )

    if not records:
        raise TabularAnonymizationError("CSV input must include at least one data row")

    return header, records


def _clean_cell(value: Any) -> str:
    cleaned = str(value).strip()
    return cleaned if cleaned else UNKNOWN_VALUE


def _normalize_column_name(column_name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", column_name.strip().lower())
    return normalized.strip("_")


def _build_normalized_column_map(header: Sequence[str]) -> Dict[str, str]:
    return {_normalize_column_name(column): column for column in header}


def _resolve_column_list(
    available_columns: Dict[str, str],
    requested: Optional[List[str]],
    defaults: Sequence[str],
    column_kind: str,
) -> List[str]:
    if requested is not None:
        resolved: List[str] = []
        missing: List[str] = []
        seen = set()
        for column in requested:
            normalized = _normalize_column_name(column)
            actual_column = available_columns.get(normalized)
            if actual_column is None:
                missing.append(column)
                continue
            if actual_column not in seen:
                resolved.append(actual_column)
                seen.add(actual_column)

        if missing:
            raise TabularAnonymizationError(
                f"Missing {column_kind} column(s): " + ", ".join(missing)
            )
        if column_kind == "quasi-identifier" and not resolved:
            raise TabularAnonymizationError(
                "At least one quasi-identifier column is required"
            )
        return resolved

    resolved_defaults = [
        available_columns[normalized_default]
        for normalized_default in defaults
        if normalized_default in available_columns
    ]
    if column_kind == "quasi-identifier" and not resolved_defaults:
        raise TabularAnonymizationError(
            "No quasi-identifier columns were detected; provide quasi_identifiers"
        )
    return resolved_defaults


def _resolve_sensitive_column(
    available_columns: Dict[str, str],
    requested: Optional[str],
) -> Optional[str]:
    if requested:
        actual_column = available_columns.get(_normalize_column_name(requested))
        if actual_column is None:
            raise TabularAnonymizationError(
                f"Missing sensitive column: {requested}"
            )
        return actual_column

    for candidate in DEFAULT_SENSITIVE_COLUMNS:
        actual_column = available_columns.get(candidate)
        if actual_column is not None:
            return actual_column
    return None


def _remove_direct_identifiers(
    records: Sequence[Dict[str, str]],
    retained_columns: Sequence[str],
) -> List[Dict[str, str]]:
    return [
        {
            column: _clean_cell(record.get(column, UNKNOWN_VALUE))
            for column in retained_columns
        }
        for record in records
    ]


def _infer_column_type(rows: Sequence[Dict[str, str]], column: str) -> str:
    values = [
        row[column]
        for row in rows
        if row.get(column, UNKNOWN_VALUE) != UNKNOWN_VALUE
    ]
    if not values:
        return "categorical"

    if all(_parse_number(value) is not None for value in values):
        return "numeric"
    if all(_parse_date(value) is not None for value in values):
        return "date"
    return "categorical"


def _parse_number(value: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_date(value: str):
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(value, date_format).date()
        except (TypeError, ValueError):
            continue
    return None


def _mondrian_partitions(
    rows: Sequence[Dict[str, str]],
    quasi_identifier_types: Dict[str, str],
    k: int,
    l: int,
    sensitive_column: Optional[str],
) -> List[List[int]]:
    final_partitions: List[List[int]] = []
    pending_partitions: List[List[int]] = [list(range(len(rows)))]

    while pending_partitions:
        indices = pending_partitions.pop()
        if len(indices) < 2 * k:
            final_partitions.append(indices)
            continue

        split = _best_split(
            rows=rows,
            indices=indices,
            quasi_identifier_types=quasi_identifier_types,
            k=k,
            l=l,
            sensitive_column=sensitive_column,
        )
        if split is None:
            final_partitions.append(indices)
            continue

        left_indices, right_indices = split
        pending_partitions.append(left_indices)
        pending_partitions.append(right_indices)

    return final_partitions


def _best_split(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    quasi_identifier_types: Dict[str, str],
    k: int,
    l: int,
    sensitive_column: Optional[str],
) -> Optional[Tuple[List[int], List[int]]]:
    best: Optional[Tuple[Tuple[int, float], List[int], List[int]]] = None

    for column, column_type in quasi_identifier_types.items():
        split = _split_partition(rows, indices, column, column_type)
        if split is None:
            continue

        left_indices, right_indices = split
        if len(left_indices) < k or len(right_indices) < k:
            continue

        if sensitive_column is not None:
            if not _partition_l_diverse(rows, left_indices, sensitive_column, l):
                continue
            if not _partition_l_diverse(rows, right_indices, sensitive_column, l):
                continue

        score = _split_score(rows, indices, column, column_type)
        if best is None or score > best[0]:
            best = (score, left_indices, right_indices)

    if best is None:
        return None
    return best[1], best[2]


def _split_partition(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
    column_type: str,
) -> Optional[Tuple[List[int], List[int]]]:
    if column_type in {"numeric", "date"}:
        return _split_ordered(rows, indices, column, column_type)
    return _split_categorical(rows, indices, column)


def _split_ordered(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
    column_type: str,
) -> Optional[Tuple[List[int], List[int]]]:
    sortable: List[Tuple[bool, float, int]] = []
    distinct_values = set()
    for row_index in indices:
        raw_value = rows[row_index][column]
        parsed_value: Optional[float]
        if column_type == "numeric":
            parsed_value = _parse_number(raw_value)
        else:
            parsed_date = _parse_date(raw_value)
            parsed_value = float(parsed_date.toordinal()) if parsed_date else None

        if parsed_value is None:
            sortable.append((True, 0.0, row_index))
        else:
            sortable.append((False, parsed_value, row_index))
            distinct_values.add(parsed_value)

    if len(distinct_values) < 2:
        return None

    sortable.sort(key=lambda item: (item[0], item[1], item[2]))
    midpoint = len(sortable) // 2
    left_indices = [item[2] for item in sortable[:midpoint]]
    right_indices = [item[2] for item in sortable[midpoint:]]
    return left_indices, right_indices


def _split_categorical(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
) -> Optional[Tuple[List[int], List[int]]]:
    counts = Counter(rows[row_index][column] for row_index in indices)
    if len(counts) < 2:
        return None

    left_categories = set()
    right_categories = set()
    left_size = 0
    right_size = 0
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if left_size <= right_size:
            left_categories.add(category)
            left_size += count
        else:
            right_categories.add(category)
            right_size += count

    left_indices = [
        row_index
        for row_index in indices
        if rows[row_index][column] in left_categories
    ]
    right_indices = [
        row_index
        for row_index in indices
        if rows[row_index][column] in right_categories
    ]
    if not left_indices or not right_indices:
        return None
    return left_indices, right_indices


def _split_score(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    column: str,
    column_type: str,
) -> Tuple[int, float]:
    if column_type == "numeric":
        numbers = [
            _parse_number(rows[row_index][column])
            for row_index in indices
        ]
        known_numbers = [number for number in numbers if number is not None]
        if len(known_numbers) < 2:
            return (0, 0.0)
        return (2, max(known_numbers) - min(known_numbers))

    if column_type == "date":
        dates = [
            _parse_date(rows[row_index][column])
            for row_index in indices
        ]
        known_ordinals = [
            float(parsed_date.toordinal())
            for parsed_date in dates
            if parsed_date is not None
        ]
        if len(known_ordinals) < 2:
            return (0, 0.0)
        return (2, max(known_ordinals) - min(known_ordinals))

    unique_values = {rows[row_index][column] for row_index in indices}
    return (1, float(len(unique_values)))


def _partition_l_diverse(
    rows: Sequence[Dict[str, str]],
    indices: Sequence[int],
    sensitive_column: str,
    l: int,
) -> bool:
    sensitive_values = {
        rows[row_index].get(sensitive_column, UNKNOWN_VALUE)
        for row_index in indices
    }
    return len(sensitive_values) >= l


def _generalize_partitions(
    rows: Sequence[Dict[str, str]],
    partitions: Sequence[Sequence[int]],
    quasi_identifier_types: Dict[str, str],
) -> Tuple[List[Dict[str, str]], int, int]:
    anonymized_rows = [dict(row) for row in rows]
    generalized_cells = 0
    suppressed_cells = 0

    for partition in partitions:
        for column, column_type in quasi_identifier_types.items():
            generalized_value, is_suppressed = _generalized_value(
                rows=rows,
                partition=partition,
                column=column,
                column_type=column_type,
            )
            for row_index in partition:
                original_value = anonymized_rows[row_index][column]
                anonymized_rows[row_index][column] = generalized_value
                if original_value != generalized_value:
                    generalized_cells += 1
                if is_suppressed:
                    suppressed_cells += 1

    return anonymized_rows, generalized_cells, suppressed_cells


def _generalized_value(
    rows: Sequence[Dict[str, str]],
    partition: Sequence[int],
    column: str,
    column_type: str,
) -> Tuple[str, bool]:
    values = [rows[row_index][column] for row_index in partition]
    if any(value == UNKNOWN_VALUE for value in values):
        return UNKNOWN_VALUE, True

    if column_type == "numeric":
        numbers = [_parse_number(value) for value in values]
        known_numbers = [number for number in numbers if number is not None]
        if not known_numbers:
            return UNKNOWN_VALUE, True
        return (
            f"{_format_number(min(known_numbers))}-{_format_number(max(known_numbers))}",
            False,
        )

    if column_type == "date":
        dates = [_parse_date(value) for value in values]
        known_dates = [parsed_date for parsed_date in dates if parsed_date is not None]
        if not known_dates:
            return UNKNOWN_VALUE, True
        min_date = min(known_dates)
        max_date = max(known_dates)
        if min_date.year == max_date.year and min_date.month == max_date.month:
            return f"{min_date.year:04d}-{min_date.month:02d}", False
        if min_date.year == max_date.year:
            return f"{min_date.year:04d}", False
        return f"{min_date.year:04d}-{max_date.year:04d}", False

    return SUPPRESSED_VALUE, True


def _format_number(number: float) -> str:
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}".rstrip("0").rstrip(".")


def _equivalence_groups(
    rows: Sequence[Dict[str, str]],
    quasi_identifiers: Sequence[str],
) -> Dict[Tuple[str, ...], List[int]]:
    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for row_index, row in enumerate(rows):
        key = tuple(row[column] for column in quasi_identifiers)
        groups[key].append(row_index)
    return dict(groups)


def _l_diversity_satisfied(
    rows: Sequence[Dict[str, str]],
    equivalence_groups: Dict[Tuple[str, ...], List[int]],
    sensitive_column: str,
    l: int,
) -> bool:
    if not equivalence_groups:
        return False

    for row_indices in equivalence_groups.values():
        sensitive_values = {
            rows[row_index].get(sensitive_column, UNKNOWN_VALUE)
            for row_index in row_indices
        }
        if len(sensitive_values) < l:
            return False
    return True


def _write_csv(
    retained_columns: Sequence[str],
    anonymized_rows: Sequence[Dict[str, str]],
) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(retained_columns),
        lineterminator="\n",
    )
    writer.writeheader()
    for row in anonymized_rows:
        writer.writerow(row)
    return buffer.getvalue()
