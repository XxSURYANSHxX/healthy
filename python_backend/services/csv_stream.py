"""Bounded-memory CSV anonymization: the one CSV engine, for small files and large.

The whole-file sanitizer planned every cell of a table held in memory. This
engine reaches the same decisions reading the file as a stream, in passes, and
keeps state per column and per distinct value - bounded in memory and spilled,
keyed by digest and sealed, beyond that (``services.spill_store``):

1. **Plan.** Every record is read and validated: encoding, BOM, dialect,
   header, widths, record and field sizes, the layout to write back, the
   evidence for clinical terminology pairs (``PairEvidence``) and the tokens
   the input already carries. Nothing is decided from an early sample: a pair
   a later row contradicts is refused.
2. **Classify.** Every cell is decided with the final plan, exactly as
   ``csv_sanitizer`` always decided it. Structured replacements are allocated
   in row order; distinct texts that need the models are queued in first-seen
   order; nothing is kept per cell.
3. **Screen and model.** Terminology displays are screened, then the queued
   texts go to full_v1 a bounded unit at a time. Answers are kept by value.
4. **Dataset privacy.** When the header names quasi-identifiers, the exact
   k-anonymity, l-diversity and t-closeness policy runs on a bounded
   projection of the transformed quasi-identifier and sensitive columns. A
   table whose exact evaluation cannot fit its bounds is held, never sampled.
5. **Write.** Every row is transformed and written as it is read; the output
   digest is computed as it is written.
6. **Verify.** Source and output are reread in lockstep. Every cell must be
   exactly its plan; protected, coded and terminology cells keep their rules;
   no replaced value, and no long digit run of one, survives anywhere; every
   text is rescanned by the residual detectors in bounded batches; the
   dataset policy is recomputed from the output cells.

Every pass reads the source afresh and re-hashes it; a source that changes
between passes is refused. The result is releasable only when every unit of
every pass completed and verified.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import heapq
import io
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from services import csv_sanitizer as cs
from services import dataset_privacy, tabular_privacy
from services.csv_terminology import (
    OutputPairs,
    PairEvidence,
    is_analyte_cell,
    is_analyte_name,
    is_measurement_unit,
    pair_provenance,
    terminology_display_accepted,
    terminology_words_only,
)
from services.multiformat import (
    FORMAT_CSV,
    TEXT_BATCH_BYTES,
    TEXT_SEPARATOR,
    MultiformatResult,
    StageClock,
    generated_token_provenance,
    sniff_signature,
)
from services.privacy_profiles import PrivacyProfileError, get_privacy_profile
from services.request_deadline import check as _deadline_check
from services.spill_store import (
    EmittedLedger,
    JobSpill,
    SpillMap,
    SpillQueue,
    SpillSet,
    SpillUnavailable,
    StreamingAllocator,
)
from services.stream_units import UnitBudget, UnitLedger
from services.surrogates import SURROGATE_PATTERN

REASON_RECORD_TOO_LARGE = "csv_record_too_large"
REASON_DATASET_BOUNDS = "csv_dataset_privacy_bounds_exceeded"

#: The CSV privacy profile, explicit. The default removes identifiers the way
#: Safe Harbor enumerates them (with the configured date, age and geography
#: handling) and never holds a table for its row count: Safe Harbor defines no
#: minimum. The risk-based profile is an engineering policy, not Safe Harbor:
#: it evaluates equivalence groups of the sanitized quasi-identifier values
#: against documented k, l and t thresholds.
PROFILE_ENV_VAR = "BIOBLOCK_CSV_PRIVACY_PROFILE"
TABULAR_PROFILE_ENV_VAR = tabular_privacy.PROFILE_ENV_VAR
PROFILE_IDENTIFIER_REMOVAL = tabular_privacy.PROFILE_IDENTIFIER_REMOVAL
PROFILE_SAFE_HARBOR_PLUS_QI = tabular_privacy.PROFILE_SAFE_HARBOR_PLUS_QI
# Compatibility name for callers and historical tests.  It resolves to the
# explicitly named production profile and is never reported as a new result.
PROFILE_RISK_BASED = tabular_privacy.PROFILE_RISK_BASED_LEGACY


def csv_privacy_profile(requested: Optional[str] = None) -> str:
    raw = (requested or PROFILE_IDENTIFIER_REMOVAL).strip().lower()
    if raw == PROFILE_RISK_BASED:
        return PROFILE_RISK_BASED
    try:
        return tabular_privacy.normalize_profile(raw)
    except tabular_privacy.ManifestError:
        # Historical behavior was fail-closed to identifier removal for an
        # unknown environment value.  Keep that deployment-safe fallback;
        # explicit function arguments are validated by the entry point.
        return PROFILE_IDENTIFIER_REMOVAL


REASON_UNITS_INCOMPLETE = "stream_units_incomplete"
REASON_SPILL_UNAVAILABLE = "stream_spill_key_unavailable"
REASON_TOKENS_OVERFLOW = "csv_existing_tokens_exceed_bound"
REASON_REPLACED_UNVERIFIABLE = "csv_replaced_value_unverifiable"

#: The largest data table the exact dataset policy is evaluated on in memory.
#: The same number of rows the whole-file engine ever accepted.
DATASET_EXACT_MAX_ROWS = cs.MAX_CSV_ROWS
#: Distinct replaced-value lengths the survivor search enumerates, and the
#: longest value it looks for. Beyond either, survivors cannot be proven absent.
MAX_SURVIVOR_LENGTHS = 4_096
MAX_SURVIVOR_LENGTH = cs.MAX_CSV_CELL_CHARS
#: Coordinates remembered per distinct output text for its findings.
FINDING_COORDINATES = cs.MAX_REVIEW_FINDINGS


def whole_word_survivor(
    cell: str, contains: Callable[[str], bool], lengths: Sequence[int]
) -> bool:
    r"""Whether any replaced value occurs in a cell with no word character either side.

    The whole-file verifier searched the joined output with ``(?<!\w)value(?!\w)``.
    Here every span of the cell that starts after a non-word character (or at
    the start), ends before one (or at the end), and has the length of some
    replaced value is looked up by digest - the same answer, with no value set
    held in memory.
    """
    size = len(cell)
    if size < 4:
        return False
    boundaries = [match.start() for match in _NON_WORD.finditer(cell)]
    ends = set(boundaries)
    ends.add(size)
    for start in [0] + [index + 1 for index in boundaries]:
        room = size - start
        for length in lengths:
            if length > room:
                break
            if start + length in ends and contains(cell[start : start + length]):
                return True
    return False


_NON_WORD = re.compile(r"\W")


def _render_output(text: str) -> str:
    """Each written row's text on its way to the sink. Identity; tests tamper here."""
    return text


PHASE_WEIGHTS = {
    "plan": 0.15,
    "classify": 0.15,
    "model": 0.2,
    "write": 0.2,
    "verify": 0.3,
}


@dataclass(frozen=True)
class Limits:
    """What one CSV may be. The bytes entry point keeps the whole-file limits."""

    max_bytes: int
    max_rows: int
    max_cells: int
    max_chars: int
    max_record_chars: int
    max_model_fields: int = cs.MAX_MODEL_FIELDS
    max_model_chars: int = cs.MAX_MODEL_CHARS
    max_columns: int = cs.MAX_CSV_COLUMNS
    max_cell_chars: int = cs.MAX_CSV_CELL_CHARS


def whole_file_limits() -> Limits:
    """The bytes entry point's limits, read when called so a test may lower them."""
    return Limits(
        cs.MAX_CSV_BYTES,
        cs.MAX_CSV_ROWS,
        cs.MAX_CSV_CELLS,
        cs.MAX_CSV_CHARS,
        cs.MAX_CSV_CHARS,
        cs.MAX_MODEL_FIELDS,
        cs.MAX_MODEL_CHARS,
        cs.MAX_CSV_COLUMNS,
        cs.MAX_CSV_CELL_CHARS,
    )


def streaming_limits(max_bytes: int, budget: Optional[UnitBudget] = None) -> Limits:
    """Limits for the streaming path: size from the capacity policy, rows unbounded."""
    budget = budget or UnitBudget.from_environment()
    unbounded = 1 << 62
    return Limits(max_bytes, unbounded, unbounded, unbounded, budget.max_record_chars)


# -- sources ------------------------------------------------------------------------------------


class Source:
    """Something that can be read from the start more than once."""

    def __init__(self, opener: Callable[[], BinaryIO], size: int):
        self._opener = opener
        self.size = size

    def open(self) -> BinaryIO:
        return self._opener()

    @classmethod
    def of_bytes(cls, content: bytes) -> "Source":
        return cls(lambda: io.BytesIO(content), len(content))

    @classmethod
    def of_path(cls, path) -> "Source":
        return cls(lambda: open(path, "rb"), os.path.getsize(path))


class _Digesting(io.RawIOBase):
    """A binary stream that hashes and counts what is read through it."""

    def __init__(self, raw: BinaryIO):
        self._raw = raw
        self.digest = hashlib.sha256()
        self.count = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        data = self._raw.read(len(buffer))
        if not data:
            return 0
        self.digest.update(data)
        self.count += len(data)
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        try:
            self._raw.close()
        finally:
            super().close()


@dataclass
class _Record:
    row: List[str]
    raw: str
    index: int
    line: int
    byte_end: int


class RecordReader:
    """The records of a CSV, validated as the whole-file parser validated them."""

    def __init__(self, source: Source, limits: Limits):
        self.source = source
        self.limits = limits
        self.bom = False
        self.sha256 = ""
        self.chars = 0
        self.rows = 0
        self.width: Optional[int] = None
        self.prefix = ""

    def __iter__(self) -> Iterator[_Record]:
        raw = self.source.open()
        head = raw.read(4)
        if head.startswith(cs._OTHER_BOMS):
            raw.close()
            raise cs._reject("CSV uploads must be UTF-8 encoded", 415)
        raw.seek(0)
        self.bom = head.startswith(codecs.BOM_UTF8)
        stream = _Digesting(raw)
        text = io.TextIOWrapper(
            io.BufferedReader(stream, 1 << 20),
            encoding="utf-8-sig" if self.bom else "utf-8",
            errors="strict",
            newline="",
        )
        consumed: List[str] = []
        limit = self.limits.max_record_chars
        meaningful = False

        def lines() -> Iterator[str]:
            nonlocal meaningful
            while True:
                try:
                    line = text.readline(limit + 1)
                except UnicodeDecodeError as exc:
                    raise cs._reject("CSV uploads must be UTF-8 encoded") from exc
                if not line:
                    return
                if "\x00" in line:
                    raise cs._reject("CSV uploads must not contain NUL bytes")
                self.chars += len(line)
                if self.chars > self.limits.max_chars:
                    raise cs._reject(
                        f"CSV uploads must be {self.limits.max_chars} characters or fewer",
                        413,
                    )
                if len(self.prefix) < cs.SNIFF_CHARS:
                    self.prefix += line[: cs.SNIFF_CHARS - len(self.prefix)]
                if not meaningful and line.strip():
                    meaningful = True
                consumed.append(line)
                if sum(len(part) for part in consumed) > limit:
                    raise cs._reject(
                        f"CSV records may hold at most {limit} characters", 413
                    )
                yield line

        reader = csv.reader(
            lines(),
            delimiter=",",
            quotechar='"',
            doublequote=True,
            skipinitialspace=False,
            strict=True,
        )
        cells = 0
        index = 0
        try:
            for row in reader:
                if self.rows > self.limits.max_rows:
                    raise cs._reject(
                        f"CSV uploads may have at most {self.limits.max_rows} data rows",
                        413,
                    )
                if len(row) > self.limits.max_columns:
                    raise cs._reject(
                        f"CSV rows may have at most {self.limits.max_columns} columns",
                        413,
                    )
                if any(len(cell) > self.limits.max_cell_chars for cell in row):
                    raise cs._reject(
                        f"CSV cells may hold at most {self.limits.max_cell_chars} characters",
                        413,
                    )
                if row:
                    if self.width is None:
                        self.width = len(row)
                    elif len(row) != self.width:
                        raise cs._reject(
                            f"CSV row {reader.line_num} has {len(row)} fields; "
                            f"expected {self.width}"
                        )
                cells += len(row)
                if cells > self.limits.max_cells:
                    raise cs._reject(
                        f"CSV uploads may have at most {self.limits.max_cells} cells",
                        413,
                    )
                record_raw = "".join(consumed)
                consumed.clear()
                self.rows += 1
                yield _Record(row, record_raw, index, reader.line_num, stream.count)
                index += 1
        except csv.Error as exc:
            if "field larger than field limit" in str(exc):
                raise cs._reject(
                    f"CSV cells may hold at most {self.limits.max_cell_chars} characters",
                    413,
                ) from exc
            raise cs._reject("Invalid CSV format") from exc
        finally:
            self.sha256 = stream.digest.hexdigest()
            text.detach()
            stream.close()
        if self.width is None or not meaningful:
            raise cs._reject("CSV input is empty")


# -- the plan -----------------------------------------------------------------------------------


@dataclass
class _LayoutTracker:
    """Whether writing the rows back reproduces the source, per line ending and quoting."""

    crlf: bool = False
    trailing: bool = False
    exact: Dict[Tuple[str, int], bool] = field(
        default_factory=lambda: {
            (terminator, quoting): True
            for terminator in ("\r\n", "\n")
            for quoting in (csv.QUOTE_MINIMAL, csv.QUOTE_ALL)
        }
    )

    def add(self, record: _Record) -> None:
        if "\r\n" in record.raw:
            self.crlf = True
        self.trailing = record.raw.endswith(("\n", "\r"))
        for (terminator, quoting), still in self.exact.items():
            if not still:
                continue
            rendered = cs._render([record.row], terminator, quoting, True)
            if not (
                rendered == record.raw
                or (
                    not self.trailing
                    and rendered.endswith(terminator)
                    and rendered[: -len(terminator)] == record.raw
                )
            ):
                self.exact[(terminator, quoting)] = False

    def layout(self, bom: bool) -> cs.Layout:
        terminator = "\r\n" if self.crlf else "\n"
        for quoting in (csv.QUOTE_MINIMAL, csv.QUOTE_ALL):
            if self.exact[(terminator, quoting)]:
                return cs.Layout(bom, terminator, quoting, self.trailing, exact=True)
        return cs.Layout(bom, terminator, csv.QUOTE_MINIMAL, self.trailing, exact=False)


@dataclass
class _Context:
    """Everything a cell's decision depends on besides its own value."""

    has_header: bool
    header: List[str]
    classes: List[cs.ColumnClass]
    code_columns: set
    sex_columns: set
    unit_columns: frozenset
    analyte_columns: frozenset
    categorical: Dict[int, str]
    term_codes: Dict[int, frozenset]
    term_displays: Dict[int, frozenset]
    roles: Optional[dataset_privacy.ColumnRoles]
    tabular_columns: Tuple[tabular_privacy.ColumnPolicy, ...]
    this_year: int
    qi_enabled: bool = False
    #: Pairs screening refused: their displays and lettered codes go to the
    #: models, their other codes are kept as the source wrote them.
    refused_displays: Dict[int, frozenset] = field(default_factory=dict)
    refused_codes: Dict[int, frozenset] = field(default_factory=dict)

    def column_class(self, column: int) -> cs.ColumnClass:
        return self.classes[column] if column < len(self.classes) else cs.TEXT_COLUMN


# Decision kinds. Each cell is exactly one of these, from its column and value.
NULL = "null"
KEEP = "keep"
PROTECTED = "protected"
CODED = "coded"
TERMINOLOGY = "terminology"
CATEGORICAL = "categorical"
REPLACE = "replace"
AGE90 = "age90"
MODEL = "model"
CLINICAL = "clinical_invariant"
CONTENT = "content_invariant"
HEADER_KEEP = "header_vocabulary"
HOLD = "hold"


@dataclass(frozen=True)
class Decision:
    kind: str
    entity_type: str = ""
    provider: bool = False
    counter: str = ""
    hold: str = ""


def _neighbours(row: Optional[Sequence[str]], column: int) -> Tuple[str, ...]:
    if row is None:
        return ()
    return tuple(
        str(row[index]) for index in (column - 1, column + 1) if 0 <= index < len(row)
    )


def _decide(
    context: _Context,
    row_index: int,
    column: int,
    value: str,
    row: Optional[Sequence[str]] = None,
) -> Decision:
    """The whole-file sanitizer's per-cell rule, unchanged, as a pure decision."""
    if context.has_header and row_index == 0:
        if not value.strip():
            return Decision(NULL)
        shared_header = (
            context.tabular_columns[column]
            if column < len(context.tabular_columns)
            else None
        )
        if cs.header_is_vocabulary(value) or tabular_privacy.is_declared_schema_header(
            value, shared_header
        ):
            return Decision(HEADER_KEEP, counter="vocabulary_header_cells")
        return Decision(MODEL)
    column_class = context.column_class(column)
    stripped = value.strip()
    if cs._is_null(value):
        return Decision(NULL, counter="null_cells")
    policy = column_class.policy

    # Resolve clinical numeric ambiguity before the shared stable-syntax
    # recognizers.  A phone-shaped vital or a long bare integer could be a
    # measurement or an identifier, so it is retained only in the internal
    # candidate and held for exact residual verification.
    if policy == cs.POLICY_CLINICAL:
        if cs._PHONE_SHAPED.match(stripped):
            return Decision(HOLD, hold=cs.REASON_AMBIGUOUS_NUMBER)
        if (
            cs._measurement(value)
            and not cs._has_letter(stripped)
            and cs._longest_integer_run(stripped)
            >= cs.AMBIGUOUS_CLINICAL_INTEGER_DIGITS
        ):
            return Decision(
                PROTECTED,
                hold=cs.REASON_AMBIGUOUS_NUMBER,
                counter="protected_numeric_cells",
            )

    # Preserve the established UUID accounting and replacement path.  The
    # shared policy recognizes the same value as an identifier, but this
    # branch must run first so existing reports do not silently change shape.
    if cs._UUID.fullmatch(stripped) and policy not in (
        cs.POLICY_DATE,
        cs.POLICY_BIRTH_YEAR,
    ):
        return Decision(
            REPLACE, column_class.entity_type or "IDENTIFIER", counter="uuid_cells"
        )

    shared_column = (
        context.tabular_columns[column]
        if column < len(context.tabular_columns)
        else tabular_privacy.ColumnPolicy(column, tabular_privacy.ROLE_UNKNOWN)
    )
    shared = tabular_privacy.occurrence_decision(
        value,
        header=context.header[column] if column < len(context.header) else "",
        column=shared_column,
        row_context=_neighbours(row, column),
    )
    if (
        shared.action != tabular_privacy.ACTION_REPLACE
        and tabular_privacy.embedded_fixed_identifier_spans(value)
    ):
        return Decision(MODEL)
    if shared.reason_code == tabular_privacy.REASON_UNRESOLVED_QI:
        return Decision(MODEL, hold=shared.reason_code)
    if shared.action == tabular_privacy.ACTION_REPLACE and shared.entity_type:
        # An unlabelled telephone-shaped measurement in a clinical column is
        # irreducibly ambiguous.  Keep it only in the internal candidate so
        # final-byte verification can report the residual and suppress output.
        if (
            policy == cs.POLICY_CLINICAL
            and shared.entity_type == "PHONE_NUMBER"
            and not re.search(
                r"\b(?:phone|telephone|fax|bleep|pager)\b",
                " ".join(_neighbours(row, column)),
                re.IGNORECASE,
            )
        ):
            return Decision(HOLD)
        clinical_ambiguity = (
            cs.REASON_AMBIGUOUS_NUMBER
            if policy == cs.POLICY_CLINICAL
            and shared.entity_type in {"PHONE_NUMBER", "US_SSN", "SSN"}
            and not re.search(
                r"\b(?:phone|telephone|fax|bleep|pager|ssn|social security)\b",
                " ".join(_neighbours(row, column)),
                re.IGNORECASE,
            )
            else shared.reason_code or ""
        )
        return Decision(
            REPLACE,
            shared.entity_type,
            provider=column_class.provider and shared.entity_type == "PERSON",
            counter="deterministic_cells",
            hold=clinical_ambiguity,
        )
    if shared.clinical_invariant:
        if policy == cs.POLICY_CLINICAL and cs._measurement(value):
            return Decision(PROTECTED, counter="protected_numeric_cells")
        return Decision(CLINICAL, counter="clinical_invariant_cells")
    if shared.content_invariant:
        if (
            shared_column.role == tabular_privacy.ROLE_QUASI_IDENTIFIER
            and cs.is_category_label(context.categorical.get(column), stripped)
        ):
            # Still count and verify the closed category in identifier-only
            # runs; the QI profile may subsequently generalize it.
            return Decision(CATEGORICAL, counter="categorical_quasi_identifier_cells")
        if (
            context.qi_enabled
            and shared_column.role == tabular_privacy.ROLE_QUASI_IDENTIFIER
        ):
            # Stable input to a declared QI transform, not an exact-output
            # invariant once dataset generalization is enabled.
            return Decision(KEEP)
        return Decision(CONTENT, counter="content_invariant_cells")

    if value in context.refused_displays.get(column, ()):
        return Decision(MODEL)
    if value in context.refused_codes.get(column, ()):
        if cs._has_letter(value):
            return Decision(MODEL)
        if cs._longest_integer_run(value) >= cs.AMBIGUOUS_INTEGER_DIGITS:
            return Decision(KEEP, hold=cs.REASON_AMBIGUOUS_NUMBER)
        return Decision(KEEP)
    if value in context.term_codes.get(column, ()):
        return Decision(CODED, counter="terminology_code_cells")
    if value in context.term_displays.get(column, ()) and terminology_words_only(value):
        return Decision(TERMINOLOGY, counter="terminology_display_cells")

    if policy == cs.POLICY_NAME_AFFIX:
        if cs.is_name_affix(stripped):
            return Decision(CODED, counter="coded_name_affix_cells")
        return Decision(REPLACE, "PERSON", counter="authority_cells")

    if policy == cs.POLICY_PHI:
        structured = cs._whole_cell_structured(value)
        entity_type = column_class.entity_type or "IDENTIFIER"
        if column_class.entity_type in ("PERSON", "ADDRESS", "LOCATION") or (
            cs._identifier_shaped(value) or " " not in stripped or structured
        ):
            return Decision(
                REPLACE, entity_type, column_class.provider, counter="authority_cells"
            )
        return Decision(MODEL)

    if policy in (cs.POLICY_DATE, cs.POLICY_BIRTH_YEAR):
        if cs._YEAR_VALUE.match(stripped):
            if (
                policy == cs.POLICY_BIRTH_YEAR
                and context.this_year - int(stripped) > 89
            ):
                return Decision(REPLACE, "DATE", counter="authority_cells")
            return Decision(KEEP)
        if (
            cs._DATE_VALUE.match(stripped)
            or cs._whole_cell_structured(value) == "DATE_TIME"
        ):
            return Decision(REPLACE, "DATE", counter="authority_cells")
        return Decision(MODEL)

    if policy == cs.POLICY_AGE:
        if re.fullmatch(r"\d{1,3}(?:\.\d+)?", stripped):
            if float(stripped) > 89:
                return Decision(AGE90, counter="authority_cells")
            return Decision(KEEP)
        return Decision(MODEL)

    if cs.is_coded_value(
        column,
        stripped,
        context.code_columns,
        context.sex_columns,
        unit_columns=context.unit_columns,
        analyte_columns=context.analyte_columns,
    ):
        return Decision(CODED, counter="coded_clinical_cells")

    if cs.is_category_label(context.categorical.get(column), stripped):
        return Decision(CATEGORICAL, counter="categorical_quasi_identifier_cells")

    structured = cs._whole_cell_structured(value)
    if policy == cs.POLICY_CLINICAL:
        if cs._PHONE_SHAPED.match(stripped):
            return Decision(HOLD, hold=cs.REASON_AMBIGUOUS_NUMBER)
        if cs._measurement(value):
            if (
                not cs._has_letter(stripped)
                and cs._longest_integer_run(stripped)
                >= cs.AMBIGUOUS_CLINICAL_INTEGER_DIGITS
            ):
                return Decision(
                    PROTECTED,
                    hold=cs.REASON_AMBIGUOUS_NUMBER,
                    counter="protected_numeric_cells",
                )
            return Decision(PROTECTED, counter="protected_numeric_cells")

    if structured:
        return Decision(
            REPLACE,
            "DATE" if structured == "DATE_TIME" else structured,
            counter="deterministic_cells",
        )
    if not cs._has_letter(stripped):
        if (
            policy != cs.POLICY_CLINICAL
            and re.fullmatch(r"[-+]?\d+", stripped.replace(",", ""))
            and cs._longest_digit_run(stripped) >= cs.AMBIGUOUS_INTEGER_DIGITS
        ):
            return Decision(HOLD, hold=cs.REASON_AMBIGUOUS_NUMBER)
        if cs._measurement(value):
            return Decision(KEEP)
    return Decision(MODEL)


class _DecisionCache:
    """Decisions per occurrence context, bounded.

    The raw value is intentionally not the whole key.  Equal digits beside a
    contact label and a laboratory label must be independently classified.
    """

    def __init__(self, context: _Context, capacity: int = 50_000):
        self.context = context
        self.capacity = capacity
        self._cache: Dict[Tuple[int, str, Tuple[str, ...]], Decision] = {}

    def __call__(
        self,
        row_index: int,
        column: int,
        value: str,
        row: Optional[Sequence[str]] = None,
    ) -> Decision:
        if self.context.has_header and row_index == 0:
            return _decide(self.context, row_index, column, value, row)
        occurrence = tuple(_fold_context(item) for item in _neighbours(row, column))
        key = (column, value, occurrence)
        found = self._cache.get(key)
        if found is None:
            found = _decide(self.context, row_index, column, value, row)
            if len(value) <= 256:
                if len(self._cache) >= self.capacity:
                    self._cache.clear()
                self._cache[key] = found
        return found


def _fold_context(value: str) -> str:
    # Context is in-memory only.  Bounding it prevents a very wide neighbour
    # from turning a decision-cache key into a second copy of the record.
    return " ".join(str(value).casefold().split())[:128]


# -- findings -----------------------------------------------------------------------------------


class _StreamFindings:
    """Why a CSV is held, cell by cell - bounded: an exact count, the first listed."""

    def __init__(self, header: Sequence[str], spill: JobSpill):
        self._names = {
            column: cell.strip()
            for column, cell in enumerate(header)
            if cs.header_is_vocabulary(cell) and len(cell.strip()) <= 64
        }
        self._seen = SpillSet(spill, "findings")
        self._listed: List[Tuple[int, int, str, Tuple]] = []
        self.count = 0
        self.by_column: Dict[str, int] = {}

    def add(
        self,
        row_index: int,
        column: int,
        category: str,
        detector: str,
        reason: str,
        transformed: bool,
    ) -> None:
        category = (
            re.sub(r"[^A-Z0-9_]", "_", str(category).upper())[:48] or "UNCLASSIFIED"
        )
        detector = re.sub(r"[^A-Za-z0-9_.-]", "_", str(detector))[:48] or "unknown"
        key = f"{row_index}:{column}:{category}:{detector}:{reason}"
        if key in self._seen:
            return
        self._seen.add(key)
        finding_id = hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        self.count += 1
        column_id = f"column_{column}"
        self.by_column[column_id] = self.by_column.get(column_id, 0) + 1
        entry = (
            -row_index,
            -column,
            finding_id,
            (row_index, column, category, detector, reason, transformed),
        )
        if len(self._listed) < cs.MAX_REVIEW_FINDINGS:
            heapq.heappush(self._listed, entry)
        elif entry > self._listed[0]:
            heapq.heapreplace(self._listed, entry)

    def summary(self) -> Dict[str, Any]:
        if not self.count:
            return {}
        listed = []
        for _r, _c, finding_id, item in sorted(
            self._listed, key=lambda e: (-e[0], -e[1], e[2])
        ):
            row_index, column, category, detector, reason, transformed = item
            entry = {
                "finding_id": finding_id,
                "row_index": row_index,
                "column_id": f"column_{column}",
                "category": category,
                "detector_source": detector,
                "reason_code": reason,
                "value_transformed": transformed,
                "reviewable": True,
            }
            if column in self._names:
                entry["column_name"] = self._names[column]
            listed.append(entry)
        return {
            "review_findings": listed,
            "review_finding_count": self.count,
            "review_findings_by_column": dict(sorted(self.by_column.items())),
        }


# -- the engine ---------------------------------------------------------------------------------


@dataclass
class StreamOutcome:
    result: MultiformatResult
    output_sha256: str = ""
    output_size: int = 0
    ledger: Optional[UnitLedger] = None


class _Engine:
    def __init__(
        self,
        source: Source,
        sink_factory: Callable[[], BinaryIO],
        reopen_sink: Callable[[], BinaryIO],
        profile: str,
        limits: Limits,
        text_anonymizer: Optional[Callable],
        residual_scanner: Optional[Callable],
        job_id: str,
        owner: str,
        budget: UnitBudget,
        progress: Optional[Callable[[str, float], None]],
        tabular_config: tabular_privacy.PolicyConfig,
    ):
        self.source = source
        self.sink_factory = sink_factory
        self.reopen_sink = reopen_sink
        self.profile = profile
        self.tabular_config = tabular_config
        self.limits = limits
        self.scanner = residual_scanner or cs._default_residual_scanner
        self.text_anonymizer = text_anonymizer
        self.budget = budget
        self.spill = JobSpill(job_id, owner)
        self.allocator = StreamingAllocator(self.spill)
        self.emitted = EmittedLedger(self.spill, self.allocator)
        self.emitted_for_result: set[str] = set()
        self.replaced = SpillSet(self.spill, "replaced")
        self.replaced_runs = SpillSet(self.spill, "replaced_runs")
        self.replaced_lengths: set = set()
        #: Structured replacements already computed. A plain bounded cache: the
        #: allocator it consults keeps every value consistent whatever it forgets.
        self.structured: Dict[str, str] = {}
        self._survivor_memo: Dict[str, bool] = {}
        self.model = SpillMap(self.spill, "model", sealed=True)
        #: Source offsets the model stage replaced, per distinct cell text, so a
        #: verification can tell a preserved occurrence from a surviving one.
        self.model_spans = SpillMap(self.spill, "model_spans", sealed=True)
        self.flags = SpillMap(self.spill, "flags", sealed=True)
        self.queue = SpillQueue(self.spill, "model_queue")
        self.ledger = UnitLedger(job_id, "csv", progress, PHASE_WEIGHTS)
        self.ledger.total_bytes = max(1, source.size)
        self.result = MultiformatResult(FORMAT_CSV, "text/csv", "")
        self.clock = StageClock("csv")
        self.categories: Dict[str, int] = {}
        self.counts = {
            name: 0
            for name in (
                "terminology_code_cells",
                "terminology_display_cells",
                "authority_cells",
                "deterministic_cells",
                "null_cells",
                "coded_clinical_cells",
                "coded_name_affix_cells",
                "vocabulary_header_cells",
                "categorical_quasi_identifier_cells",
                "uuid_cells",
                "protected_numeric_cells",
                "clinical_invariant_cells",
                "content_invariant_cells",
            )
        }
        self.source_sha256 = ""
        self.generalized: Dict[Tuple[int, int], str] = {}
        self.dataset_roles = None
        self.single_class = False

    # -- passes over the source -----------------------------------------------------------------

    def _records(self, phase: str) -> Iterator[_Record]:
        reader = RecordReader(self.source, self.limits)
        unit = self.ledger.open(phase, 0, 0)
        unit_bytes = unit_cells = unit_chars = 0
        last = None
        for record in reader:
            last = record
            yield record
            unit_bytes = record.byte_end - unit.byte_start
            unit_cells += len(record.row)
            unit_chars += len(record.raw)
            if self.budget.full(unit_bytes, unit_cells, unit_chars, 0):
                self.ledger.close(unit, record.index + 1, record.byte_end)
                unit = self.ledger.open(phase, record.index + 1, record.byte_end)
                unit_bytes = unit_cells = unit_chars = 0
        end_index = (last.index + 1) if last else 0
        self.ledger.close(unit, end_index, self.source.size if last else 0)
        if self.source_sha256 and reader.sha256 != self.source_sha256:
            self.result.error(cs.REASON_INPUT_CHANGED)
        self._reader = reader

    def _check_delimiter(self) -> None:
        with self.source.open() as handle:
            head = handle.read(cs.SNIFF_CHARS * 4)
        encoding = "utf-8-sig" if head.startswith(codecs.BOM_UTF8) else "utf-8"
        try:
            prefix = codecs.getincrementaldecoder(encoding)("strict").decode(
                head, final=False
            )
        except UnicodeDecodeError:
            return
        if cs._unsupported_delimiter(prefix[: cs.SNIFF_CHARS]):
            raise cs._reject(
                "CSV delimiter is not supported; only comma-separated values are accepted",
                415,
            )

    def plan(self) -> None:
        self._check_delimiter()
        tracker = _LayoutTracker()
        head: List[_Record] = []
        evidence: Optional[PairEvidence] = None
        self.data_rows = 0
        for record in self._records("plan"):
            tracker.add(record)
            for cell in record.row:
                self.allocator.reserve_existing(cell)
            if evidence is None:
                head.append(record)
                if len(head) < 51:
                    continue
                evidence = self._start_plan(head)
                continue
            if any(cell.strip() for cell in record.row):
                self.data_rows += 1
            evidence.add(record.row)
        if evidence is None:
            evidence = self._start_plan(head)
        reader = self._reader
        self.source_sha256 = reader.sha256
        self.result.input_sha256 = reader.sha256
        self.layout = tracker.layout(reader.bom)
        if not self.layout.exact:
            self.result.warn("csv_quoting_or_line_endings_normalized")
        self.rows_total = reader.rows
        pairs = evidence.pairs() if self.context.has_header else []
        self.pairs = pairs
        self.context.term_codes = {pair.code_column: pair.codes for pair in pairs}
        self.context.term_displays = {
            pair.display_column: pair.displays for pair in pairs
        }
        if self.allocator.reserved_overflow:
            self.result.hold(REASON_TOKENS_OVERFLOW)

    def _start_plan(self, head: List[_Record]) -> PairEvidence:
        rows = [record.row for record in head]
        if not rows:
            raise cs._reject("CSV input is empty")
        width = max(len(row) for row in rows)
        has_header = cs._looks_like_header(rows)
        classes = (
            [cs.classify_header(cell) for cell in rows[0]]
            if has_header
            else ([cs.TEXT_COLUMN] * width)
        )
        header = rows[0] if has_header else []
        tabular_columns = tabular_privacy.classify_columns(
            header if has_header else [f"column_{index}" for index in range(width)],
            rows[1:] if has_header else rows,
            self.tabular_config,
        )
        # Explicit direct-identifier declarations are release authority.  Map
        # them into the existing cell transformer so sanitizer and audit do
        # not disagree about the same manifest.
        for item in tabular_columns:
            if (
                item.index >= len(classes)
                or item.role != tabular_privacy.ROLE_DIRECT_IDENTIFIER
            ):
                continue
            entity = item.entity_type or "IDENTIFIER"
            policy = cs.POLICY_DATE if entity == "DATE" else cs.POLICY_PHI
            classes[item.index] = cs.ColumnClass(policy, entity)
        code_columns, sex_columns = cs.clinical_code_columns(header, classes)
        categorical: Dict[int, str] = {}
        roles = None
        if has_header:
            roles = tabular_privacy.dataset_roles(tabular_columns)
            categorical = {
                column: kind
                for column, kind in roles.quasi_identifiers.items()
                if (
                    kind in cs.CATEGORY_VOCABULARIES
                    or kind == dataset_privacy.QI_INSTITUTION
                )
                and classes[column].policy == cs.POLICY_TEXT
            }
        self.context = _Context(
            has_header,
            list(header),
            classes,
            code_columns,
            sex_columns,
            cs.measurement_unit_columns(header, classes) if has_header else frozenset(),
            cs.analyte_columns(header, classes) if has_header else frozenset(),
            categorical,
            {},
            {},
            roles,
            tabular_columns,
            date.today().year,
            qi_enabled=self.tabular_config.qi_enabled,
        )
        evidence = PairEvidence(header, classes) if has_header else PairEvidence([], [])
        for record in head[1:] if has_header else head:
            if any(cell.strip() for cell in record.row):
                self.data_rows += 1
            evidence.add(record.row)
        return evidence

    # -- classify -----------------------------------------------------------------------------

    def _replacement(self, entity_type: str, value: str, provider: bool) -> str:
        from services.text_anonymization import replacement_for_value

        name = f"{entity_type}\0{int(provider)}\0{value.strip()}"
        found = self.structured.get(name)
        if found is None:
            found = replacement_for_value(
                entity_type,
                value.strip(),
                profile=self.profile,
                allocator=self.allocator,
                is_provider=provider,
            )
            if len(self.structured) >= 50_000:
                self.structured.clear()
            self.structured[name] = found
        return found

    def _note_replaced(self, value: str) -> None:
        value = value.strip()
        if not value:
            return
        if len(value) >= 4:
            if len(value) > MAX_SURVIVOR_LENGTH:
                self.result.hold(REASON_REPLACED_UNVERIFIABLE)
            else:
                self.replaced.add(value)
                self.replaced_lengths.add(len(value))
        for run in cs._DIGIT_RUN.findall(value):
            if len(run) >= 6:
                self.replaced_runs.add(run)

    def classify(self) -> None:
        decide = _DecisionCache(self.context)
        self.decide = decide
        self.candidates: Dict[str, int] = {}
        self.pair_codes: Dict[int, Dict[str, int]] = {
            pair.code_column: {} for pair in self.pairs
        }
        self.model_occurrences = 0
        for record in self._records("classify"):
            for column, value in enumerate(record.row):
                decision = decide(record.index, column, value, record.row)
                if decision.counter:
                    self.counts[decision.counter] += 1
                if decision.hold:
                    self.result.hold(decision.hold)
                if decision.kind == REPLACE:
                    replacement = self._replacement(
                        decision.entity_type, value, decision.provider
                    )
                    if replacement != value:
                        self._note_replaced(value)
                        self.emitted.add(replacement.strip())
                    self.categories[decision.entity_type] = (
                        self.categories.get(decision.entity_type, 0) + 1
                    )
                elif decision.kind == AGE90:
                    self.categories["AGE_OVER_89"] = (
                        self.categories.get("AGE_OVER_89", 0) + 1
                    )
                elif decision.kind == MODEL:
                    self.model_occurrences += 1
                    self.queue.add(value)
                elif decision.kind == TERMINOLOGY:
                    self.candidates[value] = self.candidates.get(value, 0) + 1
                elif decision.kind == CODED and column in self.pair_codes:
                    codes = self.pair_codes[column]
                    codes[value] = codes.get(value, 0) + 1
        self.result.add_categories(self.categories)
        self.result.summary.update(self.counts)

    def screen_terminology(self) -> None:
        if not self.candidates:
            return
        calls = 0
        failed: set = set()

        def screen(items: Sequence[str]) -> None:
            nonlocal calls
            _deadline_check("csv.terminology_screen")
            calls += 1
            chunk = TEXT_SEPARATOR.join(items)
            found = [
                finding
                for finding in self.scanner(
                    chunk, generated_token_provenance(chunk, frozenset())
                )
                if finding.get("blocking")
            ]
            if all(terminology_display_accepted(item, found) for item in items):
                return
            if len(items) == 1:
                failed.add(items[0])
                return
            middle = len(items) // 2
            screen(items[:middle])
            screen(items[middle:])

        texts = list(self.candidates)
        for start in range(0, len(texts), cs.MAX_BATCH_FIELDS):
            screen(texts[start : start + cs.MAX_BATCH_FIELDS])
        refused = {pair.display_column for pair in self.pairs if pair.displays & failed}
        for value in texts:
            if any(
                value in pair.displays
                for pair in self.pairs
                if pair.display_column in refused
            ):
                self.queue.add(value)
                self.model_occurrences += self.candidates[value]
        for pair in self.pairs:
            if pair.display_column not in refused:
                continue
            self.context.refused_displays[pair.display_column] = (
                self.context.term_displays.pop(pair.display_column, frozenset())
            )
            self.context.refused_codes[pair.code_column] = self.context.term_codes.pop(
                pair.code_column, frozenset()
            )
            for code, seen in self.pair_codes.get(pair.code_column, {}).items():
                if cs._has_letter(code):
                    self.queue.add(code)
                    self.model_occurrences += seen
                elif cs._longest_integer_run(code) >= cs.AMBIGUOUS_INTEGER_DIGITS:
                    self.result.hold(cs.REASON_AMBIGUOUS_NUMBER)
        self.result.summary["terminology_displays_refused"] = len(failed)
        self.result.summary["terminology_screen_calls"] = calls
        self.pairs = [pair for pair in self.pairs if pair.display_column not in refused]
        self.decide = _DecisionCache(self.context)

    # -- model --------------------------------------------------------------------------------

    def run_models(self) -> bool:
        occurrences_note = self.queue.length
        self.result.summary["terminology_pairs"] = [
            {
                "code_column": f"column_{pair.code_column}",
                "display_column": f"column_{pair.display_column}",
                "codes": len(pair.codes),
            }
            for pair in self.pairs
        ]
        total_chars = 0
        for texts in self.queue.slices(self.budget.max_model_values):
            total_chars += sum(len(text) for text in texts)
        occurrences = self.model_occurrences
        self.result.summary.update(
            {
                "text_cells": occurrences,
                "unique_text_cells": occurrences_note,
                "deduplication_ratio": (
                    round(1 - occurrences_note / occurrences, 4) if occurrences else 0.0
                ),
            }
        )
        if occurrences_note > self.limits.max_model_fields:
            self.result.hold(cs.REASON_MODEL_FIELD_LIMIT)
        if total_chars > self.limits.max_model_chars:
            self.result.hold(cs.REASON_MODEL_TEXT_LIMIT)
        if (
            cs.REASON_MODEL_FIELD_LIMIT in self.result.reason_codes
            or cs.REASON_MODEL_TEXT_LIMIT in self.result.reason_codes
        ):
            self.result.summary["model_calls"] = 0
            return False
        anonymizer = cs.CellTextAnonymizer(
            self.profile, self.allocator, self.text_anonymizer
        )
        self.anonymizer = anonymizer
        position = 0
        for texts in self.queue.slices(self.budget.max_model_values):
            unit = self.ledger.open("model", position, position)
            sanitized = anonymizer.run(texts) if texts else {}
            for text in texts:
                value, spans = sanitized.get(text, (text, []))
                for start, end, _entity_type in spans:
                    self._note_replaced(text[start:end])
                if spans:
                    self.model_spans.put(
                        text, [[start, end] for start, end, _t in spans]
                    )
                if value != text:
                    for pattern in (cs._PLACEHOLDER_TOKEN, SURROGATE_PATTERN):
                        for match in pattern.finditer(value):
                            if match.group(0) not in text:
                                self.emitted.add(match.group(0))
                self.model.put(text, value)
            for text, flags in anonymizer.flagged.items():
                self.flags.put(text, flags)
            anonymizer.flagged = {}
            position += len(texts)
            self.ledger.close(unit, position, position)
            self.ledger.advance(
                "model", self.ledger.total_bytes * position / self.queue.length
            )
        if not self.queue.length:
            unit = self.ledger.open("model", 0, 0)
            self.ledger.close(unit, 0, 0)
        self.result.add_categories(anonymizer.categories)
        if anonymizer.review_reasons:
            self.result.hold(cs.REASON_TEXT_REVIEW, *sorted(anonymizer.review_reasons))
        if anonymizer.residual:
            self.result.hold(
                cs.REASON_ANONYMIZER_RESIDUAL,
                *sorted(
                    f"residual_{category.lower()}" for category in anonymizer.residual
                ),
            )
        if anonymizer.mapping_failed:
            self.result.error(cs.REASON_MAPPING)
        self.result.summary.update(
            {
                "model_calls": anonymizer.calls,
                "batches_redone_per_cell": anonymizer.batches_redone,
            }
        )
        return True

    # -- transformation -----------------------------------------------------------------------

    def transform(
        self,
        row_index: int,
        column: int,
        value: str,
        data_position: Optional[int],
        row: Optional[Sequence[str]] = None,
    ) -> Tuple[str, Decision]:
        output, decision = self.planned(row_index, column, value, row)
        if data_position is not None and self.generalized:
            generalized = self.generalized.get((data_position, column))
            if generalized is not None:
                output = generalized
        return output, decision

    def planned(
        self,
        row_index: int,
        column: int,
        value: str,
        row: Optional[Sequence[str]] = None,
    ) -> Tuple[str, Decision]:
        """A cell after its own rule, before any dataset-level generalization."""
        decision = self.decide(row_index, column, value, row)
        output = value
        if decision.kind == REPLACE:
            output = self._replacement(decision.entity_type, value, decision.provider)
        elif decision.kind == AGE90:
            from services.text_anonymization import AGE_AGGREGATE_REPLACEMENT

            output = AGE_AGGREGATE_REPLACEMENT
        elif decision.kind == MODEL:
            found = self.model.get(value)
            output = value if found is None else found
            if row is not None and not (self.context.has_header and row_index == 0):
                confirmed = tabular_privacy.confirmed_row_identifier_spans(
                    value, row, self.context.tabular_columns, column
                )
                # Work on the model's actual output.  The match is literal and
                # bounded; model edits elsewhere cannot shift a source offset
                # into an unrelated clinical span.
                for start, end, entity in confirmed:
                    source = value[start:end]
                    source_column = next(
                        (
                            item.index
                            for item in self.context.tabular_columns
                            if item.index != column
                            and item.role == tabular_privacy.ROLE_DIRECT_IDENTIFIER
                            and item.entity_type == entity
                            and item.index < len(row)
                            and str(row[item.index]).strip().casefold()
                            == source.casefold()
                        ),
                        None,
                    )
                    if source_column is None:
                        continue
                    original = str(row[source_column]).strip()
                    provider = self.context.column_class(source_column).provider
                    token = self._replacement(entity, original, provider)
                    pattern = re.compile(
                        r"(?<!\w)" + re.escape(source) + r"(?!\w)", re.IGNORECASE
                    )
                    output, count = pattern.subn(lambda _match: token, output)
                    if count:
                        self._note_replaced(source)
                        if SURROGATE_PATTERN.fullmatch(
                            token
                        ) or cs._PLACEHOLDER_TOKEN.fullmatch(token):
                            self.emitted.add(token)
                for (
                    start,
                    end,
                    entity,
                ) in tabular_privacy.embedded_fixed_identifier_spans(value):
                    source = value[start:end]
                    token = self._replacement(entity, source, False)
                    output, count = re.subn(
                        re.escape(source), lambda _match: token, output
                    )
                    if count:
                        self._note_replaced(source)
                        if SURROGATE_PATTERN.fullmatch(
                            token
                        ) or cs._PLACEHOLDER_TOKEN.fullmatch(token):
                            self.emitted.add(token)
        return output, decision

    def _data_position(
        self, record: _Record, position: int
    ) -> Tuple[Optional[int], int]:
        if self.context.has_header and record.index == 0:
            return None, position
        if not any(cell.strip() for cell in record.row):
            return None, position
        return position, position + 1

    # -- dataset privacy ----------------------------------------------------------------------

    def dataset_privacy(self) -> None:
        from services.text_anonymization import AGE_AGGREGATE_REPLACEMENT

        roles = self.context.roles
        profile = self.tabular_config.profile
        self.result.summary["csv_privacy_profile"] = profile
        self.result.summary["tabular_policy_version"] = tabular_privacy.POLICY_VERSION
        if profile == PROFILE_IDENTIFIER_REMOVAL:
            # Identifier removal only: every direct identifier and date was
            # already replaced cell by cell; the table's size and shape are
            # not a hold condition.
            self.result.summary["cells_generalized"] = 0
            return
        if not self.context.has_header or roles is None:
            return
        if not (roles.applicable or roles.ambiguous):
            self.result.summary["cells_generalized"] = 0
            return
        columns = sorted(set(roles.quasi_identifiers) | set(roles.sensitive))
        width = (max(columns) + 1) if columns else 0
        projected: List[List[str]] = []
        tuples: Dict[Tuple[str, ...], int] = {}
        sensitive_seen = {
            column: SpillSet(self.spill, f"sensitive.{column}")
            for column in roles.sensitive
        }
        bounded = self.data_rows <= DATASET_EXACT_MAX_ROWS
        position = 0
        for record in self._records("dataset"):
            data_position, position = self._data_position(record, position)
            if data_position is None:
                continue
            row = [""] * width
            for column in columns:
                if column < len(record.row):
                    row[column] = self.planned(
                        record.index, column, record.row[column], record.row
                    )[0]
            if bounded:
                projected.append(row)
            else:
                key = tuple(row[column] for column in sorted(roles.quasi_identifiers))
                if key not in tuples and len(tuples) >= 2:
                    continue
                tuples[key] = tuples.get(key, 0) + 1
                for column in roles.sensitive:
                    folded = dataset_privacy._sensitive_key(row[column], cs._is_null)
                    if folded:
                        sensitive_seen[column].add(folded)
                self._single_row = row
        if bounded:
            roles = self._value_evidenced(roles, projected)
            if not (roles.applicable or roles.ambiguous):
                self.result.summary["cells_generalized"] = 0
                return
            outcome = dataset_privacy.anonymize_dataset(
                projected,
                roles,
                is_null=cs._is_null,
                placeholders=(AGE_AGGREGATE_REPLACEMENT,),
                policy=tabular_privacy.dataset_policy(self.tabular_config),
            )
            for (index, column), _original in outcome.generalized.items():
                self.generalized[(index, column)] = outcome.rows[index][column]
            self.result.summary.update(outcome.summary)
            self.result.summary["cells_generalized"] = len(outcome.generalized)
            if outcome.reason_codes:
                self.result.hold(*outcome.reason_codes)
                return
            self.dataset_roles = roles if outcome.applicable else None
            return
        self._single_class_policy(roles, tuples, sensitive_seen)

    def _value_evidenced(self, roles, projected):
        """Location quasi-identifiers kept only where the sanitized values are places.

        A header word is not evidence by itself: a column called "ward_contact"
        whose values are replaced contacts, or "unit" whose values are
        measurement units, is not a geography.
        """
        from services.clinical_preservation import is_unit_expression
        from services.surrogates import SURROGATE_PATTERN

        keep = {}
        for column, kind in roles.quasi_identifiers.items():
            declared = next(
                (
                    item
                    for item in self.context.tabular_columns
                    if item.index == column and item.explicit
                ),
                None,
            )
            if declared is not None:
                keep[column] = kind
                continue
            # Ordered and postal quasi-identifiers are what they are; a
            # categorical one read from a header word needs its values.
            if kind in (
                dataset_privacy.QI_ZIP,
                dataset_privacy.QI_YEAR,
                dataset_privacy.QI_AGE,
                dataset_privacy.QI_SEX,
            ):
                keep[column] = kind
                continue
            values = [
                row[column].strip()
                for row in projected
                if column < len(row) and row[column].strip()
            ]
            placed = [
                value
                for value in values
                if not (
                    SURROGATE_PATTERN.fullmatch(value)
                    or "@" in value
                    or "/" in value
                    or is_unit_expression(value)
                    or sum(ch.isdigit() for ch in value) >= 6
                )
            ]
            if values and len(placed) * 2 >= len(values):
                keep[column] = kind
        return dataset_privacy.ColumnRoles(keep, roles.sensitive, roles.ambiguous)

    def _single_class_policy(self, roles, tuples, sensitive_seen) -> None:
        """The exact policy for a large table whose rows share one quasi-identifier tuple.

        Mondrian cannot split identical values, so such a table is one class
        and nothing is generalized: k is its row count, l its distinct
        sensitive values, t zero. Any other large table is held, not sampled.
        """
        policy = tabular_privacy.dataset_policy(self.tabular_config)
        from services.text_anonymization import AGE_AGGREGATE_REPLACEMENT

        summary = {
            "dataset_privacy_policy": dataset_privacy.POLICY_VERSION,
            "quasi_identifier_columns": len(roles.quasi_identifiers),
            "sensitive_columns": len(roles.sensitive),
            "k_required": policy.k_anonymity,
            "l_required": policy.l_diversity,
            "t_required": policy.t_closeness,
        }
        self.result.summary.update(summary)
        self.result.summary["cells_generalized"] = 0
        if roles.ambiguous:
            self.result.hold(dataset_privacy.REASON_AMBIGUOUS)
            return
        if len(tuples) != 1:
            self.result.hold(REASON_DATASET_BOUNDS)
            return
        table = dataset_privacy._read_table(
            [self._single_row],
            roles,
            cs._is_null,
            frozenset((AGE_AGGREGATE_REPLACEMENT,)),
        )
        if table is None:
            self.result.hold(dataset_privacy.REASON_AMBIGUOUS)
            return
        if any(
            kind == dataset_privacy.QI_AGE
            and table.keys[column][0] != self._single_row[column]
            and dataset_privacy.age_band(self._single_row[column]) is not None
            for column, kind in table.kinds.items()
        ):
            # Nothing is rewritten on this path, so an age that is not already
            # its band cannot be released as one.
            self.result.hold(REASON_DATASET_BOUNDS)
            return
        l_value = (
            min((spill_set.size for spill_set in sensitive_seen.values()), default=None)
            if policy.l_diversity is not None
            else None
        )
        rows = next(iter(tuples.values()))
        satisfied = (
            rows >= policy.min_rows
            and rows >= policy.k_anonymity
            and (
                policy.l_diversity is None
                or (l_value is not None and l_value >= policy.l_diversity)
            )
        )
        self.result.summary.update(
            {
                "equivalence_classes": 1,
                "k_achieved": rows,
                "l_achieved": l_value,
                "t_achieved": (
                    0.0 if policy.t_closeness is not None and roles.sensitive else None
                ),
                "partitions": 1,
                "generalized_cells": 0,
                "suppressed_cells": 0,
                "dataset_privacy_evaluation": "exact_single_class",
            }
        )
        if not satisfied:
            self.result.hold(dataset_privacy.REASON_IMPOSSIBLE)
            return
        self.dataset_roles = roles
        self.single_class = True

    # -- write --------------------------------------------------------------------------------

    def write(self) -> None:
        sink = self.sink_factory()
        digest = hashlib.sha256()
        size = 0

        def emit(data: bytes) -> None:
            nonlocal size
            digest.update(data)
            size += len(data)
            sink.write(data)

        if self.layout.bom:
            emit(codecs.BOM_UTF8)
        buffer = io.StringIO(newline="")
        writer = csv.writer(
            buffer,
            delimiter=",",
            quotechar='"',
            doublequote=True,
            quoting=self.layout.quoting,
            lineterminator=self.layout.lineterminator,
        )
        output_pairs = OutputPairs(self.pairs)
        flag_findings = []
        position = 0
        pending: Optional[str] = None
        changed_cells = 0
        for record in self._records("write"):
            data_position, position = self._data_position(record, position)
            out_row = []
            for column, value in enumerate(record.row):
                planned, decision = self.planned(
                    record.index, column, value, record.row
                )
                output = planned
                if data_position is not None and self.generalized:
                    output = self.generalized.get((data_position, column), planned)
                out_row.append(output)
                if planned != value:
                    changed_cells += 1
                if decision.kind == MODEL:
                    flags = self.flags.get(value)
                    if flags:
                        flag_findings.append(
                            (record.index, column, flags, output != value)
                        )
            if self.context.has_header and record.index == 0:
                pass
            else:
                output_pairs.add(out_row)
            writer.writerow(out_row)
            if pending is not None:
                emit(pending.encode("utf-8"))
            pending = _render_output(buffer.getvalue())
            buffer.seek(0)
            buffer.truncate(0)
            if len(flag_findings) > 10_000:
                self._record_flags(flag_findings)
                flag_findings = []
        if pending is not None:
            if not self.layout.trailing_newline and pending.endswith(
                self.layout.lineterminator
            ):
                pending = pending[: -len(self.layout.lineterminator)]
            emit(pending.encode("utf-8"))
        sink.flush()
        sink.close()
        self._record_flags(flag_findings)
        self.output_sha256 = digest.hexdigest()
        self.output_size = size
        self.output_pairs = output_pairs
        self.result.summary["cells_changed"] = changed_cells

    def _record_flags(self, items) -> None:
        for row_index, column, flags, transformed in items:
            for code in flags.get("review", []):
                self.findings.add(
                    row_index,
                    column,
                    cs._review_category(code),
                    "first_pass_evidence_model",
                    cs.REASON_TEXT_REVIEW,
                    transformed,
                )
            for category in flags.get("residual", {}):
                self.findings.add(
                    row_index,
                    column,
                    category,
                    "anonymizer_residual_pass",
                    cs.REASON_ANONYMIZER_RESIDUAL,
                    transformed,
                )

    # -- verification -------------------------------------------------------------------------

    def _output_records(self) -> Iterator[List[str]]:
        raw = self.reopen_sink()
        head = raw.read(3)
        if head.startswith(codecs.BOM_UTF8) != self.layout.bom:
            self.result.error(cs.REASON_DIALECT_CHANGED)
        raw.seek(0)
        text = io.TextIOWrapper(
            io.BufferedReader(_Digesting(raw), 1 << 20),
            encoding="utf-8-sig" if head.startswith(codecs.BOM_UTF8) else "utf-8",
            errors="strict",
            newline="",
        )
        prefix: List[str] = []
        prefix_chars = 0
        separators = 0
        consumed: List[str] = []

        def lines() -> Iterator[str]:
            nonlocal prefix_chars
            while True:
                line = text.readline(self.limits.max_record_chars + 1)
                if not line:
                    return
                consumed.append(line)
                if prefix_chars < cs.SNIFF_CHARS:
                    prefix.append(line)
                    prefix_chars += len(line)
                yield line

        reader = csv.reader(
            lines(), delimiter=",", quotechar='"', doublequote=True, strict=True
        )
        try:
            for row in reader:
                separators += "".join(consumed).count(TEXT_SEPARATOR)
                consumed.clear()
                yield row
        finally:
            self.output_separators = separators
            self.output_prefix = "".join(prefix)[: cs.SNIFF_CHARS]
            text.detach()

    def _survivors(self, cell: str) -> bool:
        """Whether a replaced value (four characters or more) occurs whole-word in a cell."""
        if not self.replaced.size:
            return False
        # The replaced set is final by now, so a repeated cell has one answer.
        known = self._survivor_memo.get(cell)
        if known is None:
            if not hasattr(self, "_lengths"):
                self._lengths = sorted(self.replaced_lengths)
            known = whole_word_survivor(cell, self.replaced.__contains__, self._lengths)
            if len(cell) <= 256:
                if len(self._survivor_memo) >= 100_000:
                    self._survivor_memo.clear()
                self._survivor_memo[cell] = known
        return known

    def verify(self) -> int:
        from services.residual_identifier_scan import scan as structured_scan

        source = self._records("verify")
        output = self._output_records()
        text_queue = SpillQueue(self.spill, "verify_text")
        text_meta = SpillMap(self.spill, "verify_meta", sealed=False)
        terminology_queue: Dict[str, List[Tuple[int, int]]] = {}
        structured_cache: Dict[str, list] = {}
        blocking: Dict[str, int] = {}
        verify_pairs = OutputPairs(self.pairs)
        projected: List[List[str]] = []
        dataset_columns: List[int] = []
        if self.dataset_roles is not None:
            dataset_columns = sorted(
                set(self.dataset_roles.quasi_identifiers)
                | set(self.dataset_roles.sensitive)
            )
        single_tuples: Dict[Tuple[str, ...], int] = {}
        position = 0
        source_separators = 0
        rows_seen = 0
        survived = partial = False
        try:
            for record in source:
                source_separators += record.raw.count(TEXT_SEPARATOR)
                try:
                    actual = next(output)
                except StopIteration:
                    self.result.error(cs.REASON_STRUCTURE_CHANGED)
                    return self._scan_queues(
                        text_queue, text_meta, terminology_queue, blocking
                    )
                except (UnicodeDecodeError, csv.Error):
                    self.result.error(cs.REASON_REOPEN_FAILED)
                    return 0
                rows_seen += 1
                if len(actual) != len(record.row):
                    self.result.error(cs.REASON_STRUCTURE_CHANGED)
                    return 0
                data_position, position = self._data_position(record, position)
                is_header = self.context.has_header and record.index == 0
                for column, (value, cell) in enumerate(zip(record.row, actual)):
                    for pattern in (SURROGATE_PATTERN, cs._PLACEHOLDER_TOKEN):
                        for match in pattern.finditer(cell):
                            token = match.group(0)
                            if token in self.emitted:
                                if len(self.emitted_for_result) >= 200_000:
                                    self.result.hold(
                                        "csv_emitted_token_ledger_exceeds_bound"
                                    )
                                else:
                                    self.emitted_for_result.add(token)
                    planned, decision = self.planned(
                        record.index, column, value, record.row
                    )
                    expected = planned
                    if data_position is not None and self.generalized:
                        expected = self.generalized.get(
                            (data_position, column), planned
                        )
                    if cell != expected:
                        self.result.error(cs.REASON_VALUE_CHANGED)
                        if decision.kind in (
                            PROTECTED,
                            CODED,
                            TERMINOLOGY,
                            CATEGORICAL,
                            CLINICAL,
                            CONTENT,
                        ) and (not is_header):
                            self.result.error(cs.REASON_PROTECTED_CHANGED)
                            if decision.kind == CLINICAL:
                                self.result.hold(
                                    tabular_privacy.REASON_CLINICAL_CHANGED
                                )
                            if decision.kind == CONTENT:
                                self.result.hold(tabular_privacy.REASON_CONTENT_CHANGED)
                    if planned != value:
                        # A cell the plan changed must carry no identifying
                        # digits of its source; a generalized range may. Judge
                        # each occurrence: the same digits kept elsewhere in the
                        # cell by a decision (a laboratory result beside a
                        # replaced extension) are not a surviving identifier.
                        if self._unexplained_source_digits(
                            value,
                            cell,
                            record.row,
                            column,
                            is_header,
                            whole=decision.kind != MODEL,
                        ):
                            partial = True
                    if planned != value:
                        # Keep survivor decisions at this row and column.
                        # Equal text in a separately typed clinical/code cell
                        # is not evidence that this replacement failed.
                        original = value.strip()
                        if (
                            original
                            and len(original) >= 4
                            and whole_word_survivor(
                                cell,
                                lambda candidate: candidate == original,
                                [len(original)],
                            )
                        ):
                            survived = True
                        # A surrogate's own number is generated, not a source
                        # digit. Check only the occurrence this plan changed.
                        source_runs = {
                            run for run in cs._DIGIT_RUN.findall(value) if len(run) >= 6
                        }
                        for run in cs._DIGIT_RUN.findall(
                            SURROGATE_PATTERN.sub(" ", cell)
                        ):
                            if len(run) >= 6 and run in source_runs:
                                partial = True
                    generalized = data_position is not None and (
                        (data_position, column) in self.generalized
                    )
                    kind = (
                        decision.kind
                        if not is_header
                        else (HEADER_KEEP if decision.kind == HEADER_KEEP else MODEL)
                    )
                    if kind == PROTECTED:
                        if not cs._measurement(cell):
                            self.result.error(cs.REASON_PROTECTED_CHANGED)
                        continue
                    if kind == CODED:
                        affix = (
                            self.context.column_class(column).policy
                            == cs.POLICY_NAME_AFFIX
                        )
                        if not (
                            generalized
                            or cs._CLINICAL_CODE.fullmatch(cell.strip())
                            or cell.strip().casefold() in cs.SEX_CODES
                            or (affix and cs.is_name_affix(cell))
                            or (
                                column in self.context.unit_columns
                                and is_measurement_unit(cell)
                            )
                            or (
                                column in self.context.analyte_columns
                                and is_analyte_cell(cell)
                            )
                            or cell in self.context.term_codes.get(column, ())
                        ):
                            self.result.error(cs.REASON_PROTECTED_CHANGED)
                        continue
                    if kind == TERMINOLOGY:
                        if cell not in self.context.term_displays.get(column, ()):
                            self.result.error(cs.REASON_PROTECTED_CHANGED)
                            continue
                        coordinates = terminology_queue.setdefault(cell, [])
                        if len(coordinates) < FINDING_COORDINATES:
                            coordinates.append((record.index, column, 0))
                        continue
                    if kind == CATEGORICAL:
                        if not (
                            generalized
                            or cs.is_category_label(
                                self.context.categorical.get(column), cell
                            )
                        ):
                            self.result.error(cs.REASON_PROTECTED_CHANGED)
                        continue
                    if kind == CLINICAL:
                        if cell != value:
                            self.result.hold(tabular_privacy.REASON_CLINICAL_CHANGED)
                        # Deterministic identifier syntax still wins even in a
                        # declared clinical field.  The shared decision made
                        # that check before returning CLINICAL, so exact
                        # preservation is safe to verify without a model.
                        continue
                    if kind == CONTENT:
                        if cell != value:
                            self.result.hold(tabular_privacy.REASON_CONTENT_CHANGED)
                        continue
                    if cs._is_null(cell) or cs._GENERATED_ONLY.match(cell.strip()):
                        continue
                    declared_header = (
                        self.context.tabular_columns[column]
                        if column < len(self.context.tabular_columns)
                        else None
                    )
                    if is_header and (
                        cs.header_is_vocabulary(cell)
                        or tabular_privacy.is_declared_schema_header(
                            cell, declared_header
                        )
                    ):
                        self._structured_findings(
                            cell,
                            record.index,
                            column,
                            structured_cache,
                            structured_scan,
                            blocking,
                            cell != value,
                        )
                        continue
                    column_class = (
                        self.context.column_class(column)
                        if not is_header
                        else (cs.TEXT_COLUMN)
                    )
                    if column_class.policy in (
                        cs.POLICY_PHI,
                        cs.POLICY_NAME_AFFIX,
                    ) and cs._identifier_shaped(
                        cs._PLACEHOLDER_TOKEN.sub("", SURROGATE_PATTERN.sub("", cell))
                    ):
                        self.result.hold(cs.REASON_HEADER_IDENTIFIER)
                    if column_class.policy in (
                        cs.POLICY_DATE,
                        cs.POLICY_BIRTH_YEAR,
                    ) and cs._DATE_VALUE.match(cell.strip()):
                        self.result.hold(cs.REASON_HEADER_IDENTIFIER)
                    if cs._has_letter(cell):
                        text_queue.add(cell)
                        meta = text_meta.get(cell) or {"n": 0, "at": []}
                        meta["n"] += 1
                        if len(meta["at"]) < FINDING_COORDINATES:
                            meta["at"].append(
                                [
                                    record.index,
                                    column,
                                    int(cell != value or generalized),
                                ]
                            )
                        text_meta.put(cell, meta)
                    else:
                        self._structured_findings(
                            cell,
                            record.index,
                            column,
                            structured_cache,
                            structured_scan,
                            blocking,
                            cell != value,
                        )
                if data_position is not None:
                    verify_pairs.add(actual)
                    if dataset_columns:
                        row = [""] * (max(dataset_columns) + 1)
                        for column in dataset_columns:
                            row[column] = actual[column] if column < len(actual) else ""
                        if self.single_class:
                            key = tuple(
                                row[column]
                                for column in sorted(
                                    self.dataset_roles.quasi_identifiers
                                )
                            )
                            single_tuples[key] = single_tuples.get(key, 0) + 1
                        else:
                            projected.append(row)
            try:
                next(output)
                self.result.error(cs.REASON_STRUCTURE_CHANGED)
            except StopIteration:
                pass
        except (UnicodeDecodeError, csv.Error):
            self.result.error(cs.REASON_REOPEN_FAILED)
            return 0
        if cs._unsupported_delimiter(self.output_prefix):
            self.result.error(cs.REASON_DIALECT_CHANGED)
        if self.output_separators > source_separators:
            self.result.error(cs.REASON_MARKER_SURVIVED)
        if survived:
            self.result.hold(cs.REASON_PHI_SURVIVED)
        if partial:
            self.result.hold(cs.REASON_PARTIAL_SURVIVED)
        if [pair.mapping_sha256 for pair in verify_pairs.declared()] != [
            pair.mapping_sha256 for pair in self.output_pairs.declared()
        ]:
            self.result.error(cs.REASON_VALUE_CHANGED)
        if self.dataset_roles is not None:
            if self.single_class:
                satisfied = len(single_tuples) == 1
            else:
                satisfied = dataset_privacy.verify_dataset(
                    projected,
                    self.dataset_roles,
                    is_null=cs._is_null,
                    policy=tabular_privacy.dataset_policy(self.tabular_config),
                ).satisfied
            self.result.summary["dataset_privacy_verified_on_output"] = satisfied
            if not satisfied:
                self.result.error(dataset_privacy.REASON_NOT_VERIFIED)
        return self._scan_queues(text_queue, text_meta, terminology_queue, blocking)

    def _replaced_source_spans(
        self,
        value: str,
        row: Sequence[str],
        column: int,
        is_header: bool,
        whole: bool = False,
    ) -> List[Tuple[int, int]]:
        """Source offsets of this cell that some stage replaced."""
        if whole:
            # A rule that replaces the cell as a whole replaced every offset.
            return [(0, len(value))]
        spans = [
            (int(start), int(end)) for start, end in (self.model_spans.get(value) or [])
        ]
        if not is_header:
            spans.extend(
                (start, end)
                for start, end, _entity in tabular_privacy.confirmed_row_identifier_spans(
                    value, row, self.context.tabular_columns, column
                )
            )
        spans.extend(
            (start, end)
            for start, end, _entity in tabular_privacy.embedded_fixed_identifier_spans(
                value
            )
        )
        return spans

    def _unexplained_source_digits(
        self,
        value: str,
        cell: str,
        row: Sequence[str],
        column: int,
        is_header: bool,
        whole: bool = False,
    ) -> bool:
        """Whether a run of four or more source digits survives its replacement.

        A run counts as surviving only when the written cell holds more
        occurrences of it than the source holds *outside* every replaced span.
        Occurrences kept outside a replaced span were preserved by a decision of
        their own (checked by the residual scan); occurrences inside one must
        be gone, whatever the same digits mean elsewhere.
        """
        runs = {run for run in cs._DIGIT_RUN.findall(value) if len(run) >= 4}
        if not runs:
            return False
        written = SURROGATE_PATTERN.sub(" ", cs._PLACEHOLDER_TOKEN.sub(" ", cell))
        candidates = [run for run in runs if run in written]
        if not candidates:
            return False
        replaced = self._replaced_source_spans(value, row, column, is_header, whole)
        for run in candidates:
            kept = 0
            start = value.find(run)
            while start >= 0:
                end = start + len(run)
                if not any(low < end and start < high for low, high in replaced):
                    kept += 1
                start = value.find(run, start + 1)
            if written.count(run) > kept:
                return True
        return False

    def _structured_findings(
        self, cell, row_index, column, cache, structured_scan, blocking, transformed
    ) -> None:
        entities = cache.get(cell)
        if entities is None:
            entities = [entity.entity_type for entity in structured_scan(cell)]
            if len(cache) > 50_000:
                cache.clear()
            cache[cell] = entities
        for entity_type in entities:
            blocking[entity_type] = blocking.get(entity_type, 0) + 1
            self.findings.add(
                row_index,
                column,
                entity_type,
                "structured_identifier_scan",
                cs.REASON_RESIDUAL,
                transformed,
            )

    def _scan_queues(self, text_queue, text_meta, terminology_queue, blocking) -> int:
        calls = 0
        emitted = self.emitted

        def scan(chunk: str) -> List[Dict[str, Any]]:
            nonlocal calls
            _deadline_check("csv.verify_text")
            calls += 1
            return [
                finding
                for finding in self.scanner(
                    chunk, generated_token_provenance(chunk, emitted)
                )
                if finding.get("blocking")
            ]

        def record(text: str, coordinates, found) -> None:
            for finding in found:
                category = str(finding.get("category") or "UNCLASSIFIED")
                blocking[category] = blocking.get(category, 0) + 1
                for row_index, column, transformed in coordinates:
                    self.findings.add(
                        row_index,
                        column,
                        category,
                        str(finding.get("detector") or "residual_validator"),
                        cs.REASON_RESIDUAL,
                        bool(transformed),
                    )

        position = 0
        for texts in text_queue.slices(self.budget.max_model_values):
            unit = self.ledger.open("verify_text", position, position)
            for text in texts:
                # A cell is the semantic unit of tabular verification. Never
                # create a narrative by joining independent cells: a model
                # can infer a name from the artificial adjacency alone.
                found = scan(text)
                if found:
                    record(text, text_meta.get(text)["at"], found)
            position += len(texts)
            self.ledger.close(unit, position, position, verified=True)
            self.ledger.advance("verify", self.ledger.total_bytes)
        for text, coordinates in terminology_queue.items():
            found = scan(text)
            if found and not terminology_display_accepted(text, found):
                record(text, coordinates, found)
        if blocking:
            self.result.hold(
                cs.REASON_RESIDUAL,
                *sorted(f"residual_{category.lower()}" for category in blocking),
            )
        return calls

    # -- the whole run ------------------------------------------------------------------------

    def run(self) -> StreamOutcome:
        clock = self.clock
        try:
            with clock.stage("parse"):
                self.plan()
            self.findings = _StreamFindings(self.context.header, self.spill)
            with clock.stage("classify"):
                self.classify()
            if self.candidates:
                with clock.stage("terminology_screen"):
                    self.screen_terminology()
            self.result.summary.update(self._shape_summary())
            with clock.stage("anonymize_text"):
                modelled = self.run_models()
            if not modelled:
                return StreamOutcome(clock.finish(self.result), ledger=self.ledger)
            with clock.stage("dataset_privacy"):
                self.dataset_privacy()
            with clock.stage("write"):
                self.write()
            with clock.stage("verify"):
                calls = self.verify()
            self.result.summary["verification_model_calls"] = calls
            self.result.summary.update(self.findings.summary())
            self.result.summary["terminology_provenance"] = pair_provenance(
                self.output_pairs.declared()
            )
            phases = ["plan", "classify", "model", "write", "verify"]
            if not self.ledger.complete(phases):
                self.result.error(REASON_UNITS_INCOMPLETE)
            self.result.summary["streaming"] = {
                **self.ledger.summary(),
                "spill_peak_bytes": self.spill.peak_bytes,
                "spilled_items": self.spill.spilled_items,
                "input_sha256": self.source_sha256,
                "input_bytes": self.source.size,
            }
            self.result.residual_verification_passed = (
                not self.result.errors
                and not any(
                    code.startswith("residual_")
                    or code
                    in (
                        cs.REASON_RESIDUAL,
                        cs.REASON_PHI_SURVIVED,
                        cs.REASON_PARTIAL_SURVIVED,
                        cs.REASON_HEADER_IDENTIFIER,
                    )
                    for code in self.result.reason_codes
                )
            )
            self.result.summary["output_size_bytes"] = self.output_size
            self.result.output_media_type = cs.CSV_MEDIA_TYPE
            self.result.output_extension = ".csv"
            self.result.validators = cs.VALIDATORS
            self.result.generated_tokens = frozenset(self.emitted_for_result)
            categories: Dict[str, int] = {}
            for entity_type, count in self.result.categories_detected.items():
                category = tabular_privacy.category_for_entity(entity_type)
                if category:
                    categories[category] = categories.get(category, 0) + int(count)
            state = (
                "failed_no_output"
                if self.result.errors and not self.output_size
                else "held" if self.result.reason_codes else "ready_candidate"
            )
            self.result.summary["tabular_privacy"] = tabular_privacy.policy_summary(
                self.tabular_config,
                self.context.tabular_columns,
                category_counts=categories,
                transformations={
                    "cells_changed": int(self.result.summary.get("cells_changed") or 0),
                    "cells_generalized": int(
                        self.result.summary.get("cells_generalized") or 0
                    ),
                    "cells_suppressed": int(
                        self.result.summary.get("suppressed_cells") or 0
                    ),
                },
                dataset_summary=self.result.summary,
                unresolved_reasons=(
                    code
                    for code in self.result.reason_codes
                    if code.startswith("tabular_") or "quasi_identifier" in code
                ),
                candidate_state=state,
            )
            return StreamOutcome(
                clock.finish(self.result),
                self.output_sha256,
                self.output_size,
                self.ledger,
            )
        except SpillUnavailable:
            self.result.error(REASON_SPILL_UNAVAILABLE)
            return StreamOutcome(clock.finish(self.result), ledger=self.ledger)
        finally:
            self.spill.close()

    def _shape_summary(self) -> Dict[str, Any]:
        policies: Dict[str, int] = {}
        categories: Dict[str, int] = {}
        for column_class in self.context.classes:
            policies[column_class.policy] = policies.get(column_class.policy, 0) + 1
            if column_class.entity_type:
                categories[column_class.entity_type] = (
                    categories.get(column_class.entity_type, 0) + 1
                )
        layout = self.layout
        return {
            "rows": self.rows_total - (1 if self.context.has_header else 0),
            "columns": len(self.context.classes),
            "has_header": self.context.has_header,
            "column_policies": dict(sorted(policies.items())),
            "identifying_columns": dict(sorted(categories.items())),
            "line_endings": "crlf" if layout.lineterminator == "\r\n" else "lf",
            "byte_order_mark": layout.bom,
            "quoting": "all" if layout.quoting == csv.QUOTE_ALL else "minimal",
            "round_trip_exact": layout.exact,
        }


def anonymize_csv_stream(
    source: Source,
    sink_factory: Callable[[], BinaryIO],
    reopen_sink: Callable[[], BinaryIO],
    profile: str = "strict",
    limits: Optional[Limits] = None,
    text_anonymizer: Optional[Callable] = None,
    residual_scanner: Optional[Callable] = None,
    job_id: str = "",
    owner: str = "",
    budget: Optional[UnitBudget] = None,
    progress: Optional[Callable[[str, float], None]] = None,
    tabular_manifest: Optional[Mapping[str, Any]] = None,
    tabular_profile: Optional[str] = None,
) -> StreamOutcome:
    """Anonymize a CSV from a re-readable source into a sink, and verify the sink."""
    import secrets

    if source.size <= 0:
        raise cs._reject("CSV input is empty")
    limits = limits or streaming_limits(source.size, budget)
    if source.size > limits.max_bytes:
        raise cs._reject(
            f"CSV uploads must be {limits.max_bytes} bytes or smaller", 413
        )
    with source.open() as handle:
        signature = sniff_signature(handle.read(1024))
    if signature in cs._BINARY_SIGNATURES:
        raise cs._reject("File content does not match its declared type", 415)
    try:
        get_privacy_profile(profile)
    except PrivacyProfileError as exc:
        raise cs._reject("Invalid privacy profile for CSV anonymization", 400) from exc
    try:
        tabular_config = tabular_privacy.policy_config(
            tabular_manifest, profile=tabular_profile
        )
    except tabular_privacy.ManifestError as exc:
        raise cs._reject("Invalid tabular privacy policy", 400) from exc
    engine = _Engine(
        source,
        sink_factory,
        reopen_sink,
        profile,
        limits,
        text_anonymizer,
        residual_scanner,
        job_id or secrets.token_hex(12),
        owner,
        budget or UnitBudget.from_environment(),
        progress,
        tabular_config,
    )
    engine.result.signature = signature
    try:
        return engine.run()
    except tabular_privacy.ManifestError as exc:
        raise cs._reject("Tabular policy contradicts CSV schema", 400) from exc
