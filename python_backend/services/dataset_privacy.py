"""Dataset-level privacy for released tables: k-anonymity, l-diversity, t-closeness.

Removing direct identifiers is not enough for a table. A row that is the only
one with a given age, sex and area can still be linked to a person, and rows
that share those values and one diagnosis disclose that diagnosis for everyone
among them. After direct identifiers are removed, a table with
quasi-identifiers is generalized until every group of rows sharing them - an
equivalence class - satisfies one fixed policy:

* k-anonymity, k = 5: every class has at least five rows;
* distinct l-diversity, l = 2: every class has at least two distinct non-empty
  values in every sensitive column;
* t-closeness, t = 0.2: in every sensitive column, the distribution of values
  in a class is within 0.2 of the whole table's - total variation distance for
  categories, the ordered earth mover's distance for numbers.

Quasi-identifiers are age, date or year, sex, location, ZIP or postal area,
race or ethnicity, occupation and institution columns, named by their headers.
Location values have already been replaced by pseudonyms, but equal pseudonyms
still group rows, so they are generalized like any other category. Clinical
measurements and every other column are never changed.

Generalization is a deterministic Mondrian partitioning: a partition is split
on the quasi-identifier with the widest spread, at the boundary between
distinct values nearest the median, and only when both halves still satisfy
the policy. Each final partition then gets exactly one output value per
quasi-identifier column: the value all its rows share, the numeric range its
rows span ("34-41"), or "*". The result is verified by recomputing the
classes from the output cells alone.

A table is held rather than released when it has fewer than five rows, when a
column is both a quasi-identifier and sensitive or names two kinds, when a
quasi-identifier value cannot be read as its kind, or when a sensitive column
has fewer than two distinct values in the whole table, so no generalization
could meet l.

Nothing here logs or reports a value: counts and the achieved k, l and t only.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

POLICY_VERSION = "dataset-privacy-v1"

QI_AGE = "age"
QI_YEAR = "year"
QI_SEX = "sex"
QI_ZIP = "zip"
QI_LOCATION = "location"
QI_RACE_ETHNICITY = "race_ethnicity"
QI_OCCUPATION = "occupation"
QI_INSTITUTION = "institution"
#: Generalized by widening a numeric range. An age is not one of these: it is
#: written as its fixed band (see ``age_band``) and only ever suppressed beyond it.
ORDERED_KINDS = frozenset({QI_YEAR})

SUPPRESSED = "*"

REASON_TOO_FEW_ROWS = "csv_dataset_too_few_rows"
REASON_AMBIGUOUS = "csv_quasi_identifier_ambiguous"
REASON_IMPOSSIBLE = "csv_dataset_privacy_impossible"
REASON_NOT_VERIFIED = "csv_dataset_privacy_not_verified"

#: Floating-point slack when comparing a distance with t.
EPSILON = 1e-9
MAX_AGE = 130


@dataclass(frozen=True)
class PrivacyPolicy:
    k_anonymity: int = 5
    # l is enforced only when sensitive columns exist.  ``None`` lets a
    # manifest explicitly decline an l claim; t is opt-in because it is not
    # meaningful for every sensitive attribute or sample size.
    l_diversity: Optional[int] = 2
    t_closeness: Optional[float] = 0.2
    min_rows: int = 5


POLICY = PrivacyPolicy()

_POSTAL = frozenset({"zip", "zipcode", "postcode", "postal"})
#: Header vocabulary for free-text columns: (kind, tokens).
_VOCABULARY: Tuple[Tuple[str, FrozenSet[str]], ...] = (
    (QI_SEX, frozenset({"sex", "gender"})),
    (QI_RACE_ETHNICITY, frozenset({"race", "ethnicity", "ethnic", "ancestry"})),
    (QI_OCCUPATION, frozenset({"occupation", "job", "profession", "employment"})),
    (
        QI_INSTITUTION,
        frozenset(
            {
                "hospital",
                "clinic",
                "facility",
                "institution",
                "site",
                "centre",
                "center",
                "practice",
                "ward",
                "department",
            }
        ),
    ),
    (
        QI_LOCATION,
        frozenset(
            {
                "state",
                "province",
                "region",
                "country",
                "district",
                "territory",
            }
        ),
    ),
)
SENSITIVE_TOKENS = frozenset(
    {
        "diagnosis",
        "diagnoses",
        "dx",
        "condition",
        "conditions",
        "disease",
        "disorder",
        "icd",
        "icd9",
        "icd10",
        "snomed",
        "problem",
        "comorbidity",
        "comorbidities",
    }
)

_AGE = re.compile(r"\d{1,3}(?:\.\d+)?")
_YEAR = re.compile(r"\d{4}")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")
_GENERATED = re.compile(r"<[A-Z][A-Z0-9_]*>")


def _blank(value: str) -> bool:
    return not value.strip()


def _fold(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _format_number(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:g}"


# -- column roles ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnRoles:
    #: column -> quasi-identifier kind
    quasi_identifiers: Dict[int, str]
    sensitive: Tuple[int, ...]
    ambiguous: Tuple[int, ...]

    @property
    def applicable(self) -> bool:
        return bool(self.quasi_identifiers)


def classify_columns(
    tokens: Sequence[Sequence[str]], classes: Sequence[Any]
) -> ColumnRoles:
    """The quasi-identifier kind and sensitivity of each column.

    ``tokens`` are each header's words and ``classes`` its CSV column class. A
    date, birth-year or age column is an ordered quasi-identifier, and a
    location column is a categorical one. Other direct identifiers are already
    replaced one value per person and are not generalized; clinical
    measurement columns are neither.
    """
    quasi: Dict[int, str] = {}
    sensitive: List[int] = []
    ambiguous: List[int] = []
    for column, (words, column_class) in enumerate(zip(tokens, classes)):
        token_set = frozenset(words)
        policy = getattr(column_class, "policy", None)
        kinds = set()
        is_sensitive = False
        if policy == "age":
            kinds.add(QI_AGE)
        elif policy in ("date", "birth_year"):
            kinds.add(QI_YEAR)
        elif policy == "phi":
            if getattr(column_class, "entity_type", None) == "LOCATION":
                kinds.add(QI_ZIP if token_set & _POSTAL else QI_LOCATION)
        elif policy == "text":
            kinds.update(
                kind for kind, vocabulary in _VOCABULARY if token_set & vocabulary
            )
            is_sensitive = bool(token_set & SENSITIVE_TOKENS)
        if len(kinds) > 1 or (kinds and is_sensitive):
            ambiguous.append(column)
        elif kinds:
            quasi[column] = kinds.pop()
        elif is_sensitive:
            sensitive.append(column)
    return ColumnRoles(quasi, tuple(sensitive), tuple(ambiguous))


# -- distributions ------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Distribution:
    shares: Dict[str, float]
    #: Values in numeric order, when every value is a number.
    order: Optional[Tuple[str, ...]]

    @classmethod
    def of(cls, values: Sequence[str]) -> "_Distribution":
        counts = Counter(values)
        size = len(values)
        shares = {value: count / size for value, count in counts.items()}
        order = None
        if counts and all(_NUMBER.fullmatch(value) for value in counts):
            order = tuple(sorted(counts, key=lambda value: (float(value), value)))
        return cls(shares, order)

    def distance(self, counts: Counter, size: int) -> float:
        if self.order is not None:
            if len(self.order) < 2:
                return 0.0
            cumulative = total = 0.0
            for value in self.order[:-1]:
                cumulative += counts.get(value, 0) / size - self.shares[value]
                total += abs(cumulative)
            return total / (len(self.order) - 1)
        return 0.5 * sum(
            abs(counts.get(value, 0) / size - share)
            for value, share in self.shares.items()
        )


def distribution_distance(population: Sequence[str], sample: Sequence[str]) -> float:
    """The t-closeness distance of a sample of values from the population it is drawn from."""
    return _Distribution.of(list(population)).distance(Counter(sample), len(sample))


def _sensitive_key(value: str, is_null: Callable[[str], bool]) -> str:
    return "" if is_null(value) else _fold(value)


# -- verification ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetCheck:
    satisfied: bool
    rows: int
    equivalence_classes: int
    k: int
    l: Optional[int]  # noqa: E741 - the policy's own name
    t: Optional[float]

    def summary(self) -> Dict[str, Any]:
        return {
            "equivalence_classes": self.equivalence_classes,
            "k_achieved": self.k,
            "l_achieved": self.l,
            "t_achieved": None if self.t is None else round(self.t, 4),
        }


def verify_dataset(
    rows: Sequence[Sequence[str]],
    roles: ColumnRoles,
    is_null: Callable[[str], bool] = _blank,
    policy: PrivacyPolicy = POLICY,
) -> DatasetCheck:
    """Recompute the equivalence classes from the cells and check k, l and t."""
    columns = sorted(roles.quasi_identifiers)
    widest = max(columns + list(roles.sensitive), default=-1)
    if roles.ambiguous or any(len(row) <= widest for row in rows):
        return DatasetCheck(False, len(rows), 0, 0, None, None)
    classes: Dict[Tuple[str, ...], List[int]] = {}
    for index, row in enumerate(rows):
        classes.setdefault(tuple(row[column] for column in columns), []).append(index)
    smallest = min((len(members) for members in classes.values()), default=0)
    l_value: Optional[int] = None
    t_value: Optional[float] = None
    for column in roles.sensitive:
        values = [_sensitive_key(row[column], is_null) for row in rows]
        population = _Distribution.of(values)
        for members in classes.values():
            counts = Counter(values[index] for index in members)
            distinct = sum(1 for value in counts if value)
            if policy.l_diversity is not None:
                l_value = distinct if l_value is None else min(l_value, distinct)
            if policy.t_closeness is not None:
                distance = population.distance(counts, len(members))
                t_value = distance if t_value is None else max(t_value, distance)
    satisfied = (
        len(rows) >= policy.min_rows
        and smallest >= policy.k_anonymity
        and (
            policy.l_diversity is None
            or (l_value is not None and l_value >= policy.l_diversity)
        )
        and (
            policy.t_closeness is None
            or (t_value is not None and t_value <= policy.t_closeness + EPSILON)
        )
    )
    return DatasetCheck(satisfied, len(rows), len(classes), smallest, l_value, t_value)


# -- generalization -------------------------------------------------------------------------------


@dataclass
class DatasetPrivacyOutcome:
    applicable: bool
    rows: List[List[str]]
    #: (row, column) -> original value, for every generalized cell.
    generalized: Dict[Tuple[int, int], str] = field(default_factory=dict)
    reason_codes: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def releasable(self) -> bool:
        return not self.reason_codes


@dataclass
class _Table:
    size: int
    kinds: Dict[int, str]
    #: ordered column -> number per row, None where the cell is empty or generated
    numbers: Dict[int, List[Optional[float]]]
    #: quasi-identifier column -> comparison key per row
    keys: Dict[int, List[str]]
    spans: Dict[int, float]
    distinct: Dict[int, int]
    #: sensitive column -> key per row
    sensitive: Dict[int, List[str]]
    populations: Dict[int, _Distribution]


#: Ages at or above this are one band, as Safe Harbor requires.
AGE_TOP = 90
AGE_TOP_BAND = f"{AGE_TOP}+"
#: A band's bounds are joined by an en dash: a hyphen between two numbers is read
#: as a date by spreadsheet programs ("10-19" becomes 19 October).
AGE_BAND_SEPARATOR = "–"
_AGE_BAND = re.compile(rf"(\d{{1,2}}){AGE_BAND_SEPARATOR}(\d{{1,2}})")


def age_band(text: str) -> Optional[str]:
    """The one canonical band of a valid age: 0-9, 10-19, ... 80-89 and 90+.

    A value that is already a canonical band is returned unchanged. Anything
    else - mixed text, a sign, an age above ``MAX_AGE`` - is None, never cast.
    """
    stripped = text.strip()
    if stripped == AGE_TOP_BAND:
        return stripped
    match = _AGE_BAND.fullmatch(stripped)
    if match:
        low, high = int(match.group(1)), int(match.group(2))
        return stripped if low % 10 == 0 and high == low + 9 and low < AGE_TOP else None
    if not _AGE.fullmatch(stripped):
        return None
    value = float(stripped)
    if value > MAX_AGE:
        return None
    if value >= AGE_TOP:
        return AGE_TOP_BAND
    low = int(value) // 10 * 10
    return f"{low}{AGE_BAND_SEPARATOR}{low + 9}"


def _ordered_value(kind: str, text: str) -> Optional[float]:
    stripped = text.strip()
    if kind == QI_YEAR and _YEAR.fullmatch(stripped):
        return float(stripped)
    return None


def _read_table(
    rows: Sequence[Sequence[str]],
    roles: ColumnRoles,
    is_null: Callable[[str], bool],
    placeholders: FrozenSet[str],
) -> Optional[_Table]:
    widest = max(list(roles.quasi_identifiers) + list(roles.sensitive))
    if any(len(row) <= widest for row in rows):
        return None
    numbers: Dict[int, List[Optional[float]]] = {}
    keys: Dict[int, List[str]] = {}
    spans: Dict[int, float] = {}
    for column, kind in roles.quasi_identifiers.items():
        column_keys: List[str] = []
        if kind == QI_AGE:
            for row in rows:
                text = row[column]
                band = age_band(text)
                if band is None and not (
                    is_null(text)
                    or _GENERATED.fullmatch(text.strip())
                    or text.strip() in placeholders
                ):
                    return None
                column_keys.append(band if band is not None else _fold(text))
        elif kind in ORDERED_KINDS:
            column_numbers: List[Optional[float]] = []
            for row in rows:
                text = row[column]
                number = _ordered_value(kind, text)
                if number is None and not (
                    is_null(text)
                    or _GENERATED.fullmatch(text.strip())
                    or text.strip() in placeholders
                ):
                    return None
                column_numbers.append(number)
                column_keys.append(
                    _format_number(number) if number is not None else _fold(text)
                )
            present = [number for number in column_numbers if number is not None]
            spans[column] = (max(present) - min(present)) if present else 0.0
            numbers[column] = column_numbers
        else:
            column_keys = [_fold(row[column]) for row in rows]
        keys[column] = column_keys
    sensitive = {
        column: [_sensitive_key(row[column], is_null) for row in rows]
        for column in roles.sensitive
    }
    return _Table(
        size=len(rows),
        kinds=dict(roles.quasi_identifiers),
        numbers=numbers,
        keys=keys,
        spans=spans,
        distinct={column: len(set(values)) for column, values in keys.items()},
        sensitive=sensitive,
        populations={
            column: _Distribution.of(values) for column, values in sensitive.items()
        },
    )


def _acceptable(table: _Table, indices: Sequence[int], policy: PrivacyPolicy) -> bool:
    if len(indices) < policy.k_anonymity:
        return False
    for column, values in table.sensitive.items():
        counts = Counter(values[index] for index in indices)
        if policy.l_diversity is not None and (
            sum(1 for value in counts if value) < policy.l_diversity
        ):
            return False
        if policy.t_closeness is not None and (
            table.populations[column].distance(counts, len(indices))
            > policy.t_closeness + EPSILON
        ):
            return False
    return True


def _width(table: _Table, column: int, indices: Sequence[int]) -> float:
    keys = table.keys[column]
    if column in table.numbers:
        numbers = table.numbers[column]
        present = [numbers[index] for index in indices if numbers[index] is not None]
        missing = {keys[index] for index in indices if numbers[index] is None}
        width = 0.0
        if len(present) > 1 and table.spans[column] > 0:
            width += (max(present) - min(present)) / table.spans[column]
        if present and missing:
            width += 1.0
        if len(missing) > 1:
            width += (len(missing) - 1) / table.distinct[column]
        return width
    distinct = len({keys[index] for index in indices})
    return (distinct - 1) / max(1, table.distinct[column] - 1)


def _split_categorical(keys: Sequence[str], indices: Sequence[int]):
    counts = Counter(keys[index] for index in indices)
    if len(counts) < 2:
        return None
    left_keys = set()
    left = right = 0
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if left <= right:
            left_keys.add(key)
            left += count
        else:
            right += count
    left_rows = [index for index in indices if keys[index] in left_keys]
    right_rows = [index for index in indices if keys[index] not in left_keys]
    return (left_rows, right_rows) if left_rows and right_rows else None


def _split(table: _Table, column: int, indices: Sequence[int]):
    keys = table.keys[column]
    if column not in table.numbers:
        return _split_categorical(keys, indices)
    numbers = table.numbers[column]
    present = [index for index in indices if numbers[index] is not None]
    missing = [index for index in indices if numbers[index] is None]
    if present and missing:
        return present, missing
    if missing:
        return _split_categorical(keys, missing)
    ordered = sorted(present, key=lambda index: (numbers[index], index))
    boundaries = [
        position
        for position in range(1, len(ordered))
        if numbers[ordered[position - 1]] != numbers[ordered[position]]
    ]
    if not boundaries:
        return None
    middle = len(ordered) / 2
    cut = min(boundaries, key=lambda position: (abs(position - middle), position))
    return ordered[:cut], ordered[cut:]


def _partitions(table: _Table, policy: PrivacyPolicy) -> List[List[int]]:
    final: List[List[int]] = []
    pending: List[List[int]] = [list(range(table.size))]
    while pending:
        indices = pending.pop()
        chosen = None
        if len(indices) >= 2 * policy.k_anonymity:
            widths = sorted(
                ((_width(table, column, indices), column) for column in table.kinds),
                key=lambda item: (-item[0], item[1]),
            )
            for width, column in widths:
                if width <= 0:
                    break
                split = _split(table, column, indices)
                if split and all(_acceptable(table, side, policy) for side in split):
                    chosen = split
                    break
        if chosen is None:
            final.append(sorted(indices))
        else:
            pending.extend(list(side) for side in chosen)
    return sorted(final, key=lambda part: part[0])


def _shared_value(
    table: _Table, column: int, partition: Sequence[int], texts: Sequence[str]
) -> str:
    if table.kinds[column] == QI_AGE:
        bands = {table.keys[column][index] for index in partition}
        if len(bands) == 1 and age_band(next(iter(bands))) == next(iter(bands)):
            return next(iter(bands))
    if len(set(texts)) == 1:
        return texts[0]
    if column in table.numbers:
        numbers = [table.numbers[column][index] for index in partition]
        if all(number is not None for number in numbers):
            low, high = min(numbers), max(numbers)
            if low == high:
                return _format_number(low)
            return f"{_format_number(low)}-{_format_number(high)}"
        return SUPPRESSED
    if len({table.keys[column][index] for index in partition}) == 1:
        # One value written more than one way: one spelling for all of them.
        return min(texts)
    return SUPPRESSED


def _hold(outcome: DatasetPrivacyOutcome, reason: str) -> DatasetPrivacyOutcome:
    outcome.reason_codes.append(reason)
    return outcome


def _band_valid_ages(
    rows: Sequence[Sequence[str]], roles: ColumnRoles, outcome: DatasetPrivacyOutcome
) -> None:
    """Write every valid age as its band even when the table is held for another reason.

    A band needs no equivalence-class decision, so a review copy of a table held
    for an unresolved column or too few rows still carries no exact age in a
    declared age column. An age that is not a valid number is left as it is.
    """
    for column, kind in roles.quasi_identifiers.items():
        if kind != QI_AGE:
            continue
        for index, row in enumerate(rows):
            if column >= len(row):
                continue
            band = age_band(row[column])
            if band is not None and band != row[column]:
                outcome.rows[index][column] = band
                outcome.generalized[(index, column)] = row[column]


def anonymize_dataset(
    rows: Sequence[Sequence[str]],
    roles: ColumnRoles,
    is_null: Callable[[str], bool] = _blank,
    placeholders: Iterable[str] = (),
    policy: PrivacyPolicy = POLICY,
) -> DatasetPrivacyOutcome:
    """Generalize the quasi-identifiers of data rows until the policy holds, then verify."""
    output = [list(row) for row in rows]
    outcome = DatasetPrivacyOutcome(roles.applicable, output)
    if not roles.applicable and not roles.ambiguous:
        return outcome
    outcome.summary.update(
        {
            "dataset_privacy_policy": POLICY_VERSION,
            "quasi_identifier_columns": len(roles.quasi_identifiers),
            "sensitive_columns": len(roles.sensitive),
            "k_required": policy.k_anonymity,
            "l_required": policy.l_diversity,
            "t_required": policy.t_closeness,
        }
    )
    if roles.ambiguous:
        # The table is held: some column's role is unresolved, so no claim can be made about it
        # as a whole. Its review copy is still generalized over the quasi-identifiers that are
        # known, so a person inspecting it sees no more than the policy would have released.
        # Only group size (k) is enforced there: l and t depend on which columns are sensitive,
        # and that is part of what is unresolved.
        known = ColumnRoles(roles.quasi_identifiers, (), ())
        if known.applicable and len(rows) >= policy.min_rows:
            inner = anonymize_dataset(
                rows,
                known,
                is_null,
                placeholders,
                PrivacyPolicy(policy.k_anonymity, None, None, policy.min_rows),
            )
            outcome.rows, outcome.generalized = inner.rows, inner.generalized
            outcome.summary.update(inner.summary)
        else:
            _band_valid_ages(rows, roles, outcome)
        return _hold(outcome, REASON_AMBIGUOUS)
    if len(rows) < policy.min_rows:
        _band_valid_ages(rows, roles, outcome)
        return _hold(outcome, REASON_TOO_FEW_ROWS)
    table = _read_table(rows, roles, is_null, frozenset(placeholders))
    if table is None:
        _band_valid_ages(rows, roles, outcome)
        return _hold(outcome, REASON_AMBIGUOUS)
    root = list(range(table.size))
    if not _acceptable(table, root, policy):
        # The whole table is one class at most, and even it fails l.
        _band_valid_ages(rows, roles, outcome)
        return _hold(outcome, REASON_IMPOSSIBLE)

    partitions = _partitions(table, policy)
    for partition in partitions:
        for column in table.kinds:
            texts = [rows[index][column] for index in partition]
            value = _shared_value(table, column, partition, texts)
            for index, text in zip(partition, texts):
                if text != value:
                    output[index][column] = value
                    outcome.generalized[(index, column)] = text
    check = verify_dataset(output, roles, is_null, policy)
    outcome.summary.update(check.summary())
    outcome.summary.update(
        {
            "partitions": len(partitions),
            "generalized_cells": sum(
                1
                for (index, column) in outcome.generalized
                if output[index][column] != SUPPRESSED
            ),
            "suppressed_cells": sum(
                1
                for (index, column) in outcome.generalized
                if output[index][column] == SUPPRESSED
            ),
        }
    )
    if not check.satisfied:
        _hold(outcome, REASON_NOT_VERIFIED)
    return outcome
