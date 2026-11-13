"""Scoped source projection for CSV/TXT/XLSX streaming and verification."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

FORMAT_WORKBOOK = "workbook"
REASON_UNSUPPORTED = "audit_format_unsupported"

AUDIT_VERSION = "release-audit-v4"

MAX_AUDIT_LOCATIONS = 10_000

_AUDIT_DUPLICATE_LOCATIONS = 50

FORMAT_TEXT = "text"

FORMAT_CSV = "csv"

CATEGORY_ARTIFACT = "ARTIFACT"

CATEGORY_METADATA = "METADATA"

CATEGORY_PERSON = "PERSON"

CATEGORY_DATE = "DATE"

CATEGORY_IDENTIFYING_TEXT = "IDENTIFYING_TEXT"

CATEGORY_BURNED_IN = "BURNED_IN_TEXT"

CATEGORY_LINK = "LINK"

CATEGORY_QUASI_IDENTIFIER = "QUASI_IDENTIFIER"

CATEGORY_OCR = "OCR"

CATEGORY_UNSPECIFIED = "UNSPECIFIED"

REASON_UNREADABLE = "audit_artifact_unreadable"

REASON_STRUCTURED = "audit_structured_identifier_present"

REASON_MODEL = "audit_model_detected_identifier"

REASON_OCR_INCOMPLETE = "audit_ocr_incomplete"

REASON_WORKBOOK_PROPERTIES = "audit_workbook_properties_present"

REASON_WORKBOOK_AUTHOR = "audit_workbook_comment_author_present"

REASON_WORKBOOK_LINK = "audit_workbook_link_present"

REASON_CSV_DATASET = "audit_csv_dataset_privacy_not_met"

REASON_CSV_ORIGINAL_IDENTIFIER = "audit_csv_original_identifier_present"

REASON_CSV_TERMINOLOGY = "audit_csv_terminology_provenance_rejected"

REASON_CSV_IDENTIFIER_COLUMN = "audit_csv_identifier_under_identifier_header"

REASON_CSV_DATE_COLUMN = "audit_csv_date_under_date_header"

REASON_TABULAR_POLICY = "audit_tabular_policy_provenance_rejected"

REASON_WORKBOOK_PACKAGE = "audit_workbook_package_surface_present"

REASON_WORKBOOK_FORMULA = "audit_workbook_formula_identifier_present"

REASON_WORKBOOK_DATASET = "audit_workbook_dataset_privacy_not_met"

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f]")

_DATASET_EXACT_MAX_ROWS = 200_000

_CATEGORY_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,40}")

READS_EXACT_TEXT = "reads_exact_text"


def artifact_scanner(emitted=frozenset()) -> Callable[[str], Dict[str, int]]:
    """The residual validator over the exact text of a written file.

    ``emitted`` is the ledger of tokens the transformation wrote. With no
    authenticated ledger the empty default trusts no token shape.
    """
    from services.multiformat import generated_token_provenance
    from services.text_anonymization import residual_phi_categories

    def scanner(text: str) -> Dict[str, int]:
        return residual_phi_categories(text, generated_token_provenance(text, emitted))

    setattr(scanner, READS_EXACT_TEXT, True)
    # The streaming ledger supports membership without exposing its contents.
    setattr(scanner, "emitted_tokens", emitted)
    return scanner


@dataclass(frozen=True, order=True)
class AuditFinding:
    category: str
    reason_code: str


@dataclass(frozen=True)
class AuditReport:
    file_format: str
    findings: Tuple[AuditFinding, ...]
    checks: Tuple[str, ...]
    locations: Tuple[Tuple[str, str, str], ...] = ()
    locations_truncated: bool = False

    @property
    def passed(self) -> bool:
        return not self.findings

    def public_dict(self) -> dict:
        return {
            "audit_version": AUDIT_VERSION,
            "file_format": self.file_format,
            "passed": self.passed,
            "findings": [
                {"category": finding.category, "reason_code": finding.reason_code}
                for finding in self.findings
            ],
            "finding_locations": [
                {"category": category, "reason_code": reason, "location": location}
                for category, reason, location in self.locations
            ],
            "finding_locations_truncated": self.locations_truncated,
            "checks": list(self.checks),
        }


def report_lines(fixture_id: str, report: AuditReport) -> List[str]:
    """CI output: the fixture, the category and the reason code - nothing else."""
    if report.passed:
        return [f"{fixture_id}\tNONE\taudit_passed"]
    return [
        f"{fixture_id}\t{finding.category}\t{finding.reason_code}"
        for finding in report.findings
    ]


class _Audit:
    def __init__(
        self,
        text_scanner: Optional[Callable[[str], Iterable[str]]],
        finding_scanner: Optional[Callable[[str], Iterable[dict]]] = None,
        provenance: Optional[dict] = None,
    ):
        self.findings: Set[AuditFinding] = set()
        self.locations: Set[Tuple[str, str, str]] = set()
        self.locations_truncated = False
        self.checks: List[str] = []
        self.text_scanner = text_scanner
        #: Detector-level findings, needed to tell one model's misreading of a
        #: terminology display from two detectors agreeing on a name.
        self.finding_scanner = finding_scanner
        #: What the sanitizer declared it relied on. Checked, never trusted.
        self.provenance = provenance
        # This is a membership-only ledger for tokens emitted by this file's
        # transformation. It may be spill-backed and need not be iterable.
        self.emitted = getattr(text_scanner, "emitted_tokens", frozenset())
        self._detector = None

    def flag(self, category: str, reason: str, location: str = "") -> None:
        self.findings.add(AuditFinding(category, reason))
        if location:
            if len(self.locations) < MAX_AUDIT_LOCATIONS:
                self.locations.add((category, reason, location))
            elif (category, reason, location) not in self.locations:
                self.locations_truncated = True

    def check(self, name: str) -> None:
        if name not in self.checks:
            self.checks.append(name)

    def check_tokens(self, text, places=("",)):
        from services.residual_identifier_scan import foreign_token_entities
        from services.transformation_provenance import generated_token_provenance

        provenance = generated_token_provenance(text, self.emitted)
        for entity in foreign_token_entities(text):
            if provenance.covering(entity.start, entity.end) is None:
                for place in places:
                    self.flag(entity.entity_type, REASON_STRUCTURED, place)

    def scan_text(
        self,
        text: str,
        model: bool = True,
        model_text: Optional[str] = None,
        location: str = "",
        locations: Sequence[str] = (),
    ) -> None:
        from services.artifact_text import fill_placeholders, strip_placeholders
        from services.phi_detection import StructuredPatternDetector
        from services.residual_identifier_scan import scan

        places = tuple(locations) or (location,)
        self.check_tokens(text or "", places)
        cleaned = strip_placeholders(text or "")
        if not cleaned.strip():
            return
        if self._detector is None:
            self._detector = StructuredPatternDetector()
        # The token-distance scan reads each placeholder as one value token so a
        # satisfied label cannot reach past it; the pattern detector is unchanged.
        from services.detection_evidence import is_undated_clock_time

        for entity in [
            *scan(fill_placeholders(text or "")),
            *self._detector.detect(cleaned),
        ]:
            # The same reading the sanitizer applies: a time of day with no
            # date on its line is not an element of a date.
            if entity.entity_type == "TIME" and is_undated_clock_time(
                cleaned, entity.start, entity.end
            ):
                continue
            for place in places:
                self.flag(entity.entity_type, REASON_STRUCTURED, place)
        if model and self.text_scanner is not None:
            if model_text is None:
                # A scanner that knows where the sanitizer's tokens are reads
                # the text as written. Masking the tokens instead rewrites the
                # sentence it checks: "by Dr <PROVIDER_1> at the" becomes "by
                # Dr   at the", and a title rule then takes "at the" for a name.
                model_text = (
                    text
                    if getattr(self.text_scanner, READS_EXACT_TEXT, False)
                    else cleaned
                )
            for category in self.text_scanner(model_text):
                name = str(category)
                for place in places:
                    self.flag(
                        (
                            name
                            if _CATEGORY_NAME.fullmatch(name)
                            else CATEGORY_UNSPECIFIED
                        ),
                        REASON_MODEL,
                        place,
                    )

    def burned_in(self, texts: Iterable[str], uncertain: bool, reason: str) -> None:
        from services.text_anonymization import (
            EXTRACTED_NON_PHI,
            classify_extracted_text,
        )

        for text in texts:
            if text.strip() and classify_extracted_text(text) != EXTRACTED_NON_PHI:
                self.flag(CATEGORY_BURNED_IN, reason)
        if uncertain:
            self.flag(CATEGORY_OCR, REASON_OCR_INCOMPLETE)

    def report(self, file_format: str) -> AuditReport:
        return AuditReport(
            file_format,
            tuple(sorted(self.findings)),
            tuple(self.checks),
            tuple(sorted(self.locations)),
            self.locations_truncated,
        )


def _plain_text(content: bytes, encoding: str = "utf-8") -> str:
    text = content.decode(encoding)
    if _CONTROL.search(text):
        raise ValueError("binary content")
    return text


def _audit_text(state: _Audit, content: bytes, _ocr_backend) -> None:
    state.check("text_channels")
    state.scan_text(_plain_text(content))


def _audit_csv(state: _Audit, content: bytes, _ocr_backend) -> None:
    """Every cell under its column's rule, as the CSV sanitizer verifies its own output.

    Search joins a row into "label: value | label: value", which sets a value
    beside labels it never sat next to - a year range after a date label, a
    code after a postal label - and scanners read those pairs as identifiers.
    The audit reads cells one at a time instead.
    """
    text = _plain_text(content, "utf-8-sig")

    def rows() -> Iterable[List[str]]:
        return csv.reader(io.StringIO(text, newline=""), strict=True)

    _audit_csv_rows(state, rows)


def _csv_file_rows(path) -> Callable[[], Iterable[List[str]]]:
    """A re-readable row source over a CSV file, bounded per record, never loaded whole."""

    def rows() -> Iterable[List[str]]:
        with open(
            path, "r", encoding="utf-8-sig", errors="strict", newline=""
        ) as stream:

            def lines():
                while True:
                    line = stream.readline(4 * 1024 * 1024)
                    if not line:
                        return
                    if _CONTROL.search(line):
                        raise ValueError("binary content")
                    yield line

            yield from csv.reader(lines(), strict=True)

    return rows


class _TextCache:
    """Each distinct value scanned once, however often it recurs. Bounded."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self._seen: Set[Tuple[str, bool]] = set()

    def first(self, text: str, model: bool) -> bool:
        key = (text, model)
        if key in self._seen:
            return False
        if len(self._seen) >= self.capacity:
            self._seen.clear()
        self._seen.add(key)
        return True


def _audit_csv_rows(state: _Audit, rows: Callable[[], Iterable[List[str]]]) -> None:
    """Two passes over any re-readable row source: pair evidence, then every cell."""
    from services import csv_sanitizer as cs
    from services import csv_terminology, dataset_privacy, tabular_privacy
    from services.spill_store import JobSpill, SpillMap, SpillQueue

    head: List[List[str]] = []
    for row in rows():
        head.append(row)
        if len(head) >= 51:
            break
    if not head:
        return
    has_header = cs._looks_like_header(head)
    header = head[0] if has_header else []
    classes = [cs.classify_header(cell) for cell in header]
    # Terminology allowances exist only for pairs the sanitizer declared AND
    # that the artifact's own cells prove again under the one shared policy.
    # With no declaration the audit makes no allowance; with a declaration the
    # artifact contradicts, it fails.
    state.check("csv_terminology")
    declared_provenance = state.provenance
    if (
        isinstance(declared_provenance, dict)
        and "terminology_provenance" in declared_provenance
    ):
        declared_provenance = declared_provenance.get("terminology_provenance")
    parsed, refusal = csv_terminology.parse_declared(declared_provenance, len(head[0]))
    pairs: List = []
    if parsed:
        if not has_header:
            refusal = "terminology_provenance_mismatch"
        else:
            evidence = csv_terminology.PairEvidence(
                header, classes, guard_generated=True
            )
            for index, row in enumerate(rows()):
                if index:
                    evidence.add(row)
            checked, refusal = csv_terminology.check_declared(evidence, parsed)
            pairs = checked or []
    if refusal and refusal != "terminology_provenance_missing":
        state.flag(CATEGORY_ARTIFACT, REASON_CSV_TERMINOLOGY)
    safe_codes = {pair.code_column: pair.codes for pair in pairs}
    safe_displays = {pair.display_column: pair.displays for pair in pairs}
    terminology_texts: dict = {}
    try:
        if isinstance(state.provenance, dict) and state.provenance.get(
            "tabular_privacy"
        ):
            tabular_config = tabular_privacy.config_from_metadata(
                state.provenance.get("tabular_privacy")
            )
        else:
            # A generic audit has no artifact-bound profile to recover. Use
            # the conservative fixed default, never mutable process config.
            tabular_config = tabular_privacy.PolicyConfig()
    except tabular_privacy.ManifestError:
        state.flag(CATEGORY_ARTIFACT, REASON_TABULAR_POLICY)
        tabular_config = tabular_privacy.PolicyConfig()
    try:
        shared_columns = tabular_privacy.classify_columns(
            header, head[1:], tabular_config
        )
    except tabular_privacy.ManifestError:
        state.flag(CATEGORY_ARTIFACT, REASON_TABULAR_POLICY)
        shared_columns = tabular_privacy.classify_columns(
            header, head[1:], tabular_privacy.PolicyConfig()
        )
    roles = tabular_privacy.dataset_roles(shared_columns)
    code_columns, sex_columns = cs.clinical_code_columns(header, classes)
    affix_columns = cs.name_affix_columns(classes)
    unit_columns = cs.measurement_unit_columns(header, classes)
    analyte_columns = cs.analyte_columns(header, classes)
    state.check("csv_cells")
    structured_once = _TextCache()
    dataset_columns = sorted(set(roles.quasi_identifiers) | set(roles.sensitive))
    projected: List[List[str]] = []
    single: dict = {}
    data_rows = 0
    with JobSpill("audit-" + __import__("secrets").token_hex(8)) as spill:
        texts = SpillQueue(spill, "audit_texts")
        text_locations = SpillMap(spill, "audit_text_locations", sealed=False)
        for row_index, row in enumerate(rows()):
            data = row_index >= (1 if has_header else 0)
            for column, cell in enumerate(row):
                location = f"row:{row_index}:column:{column}"
                value = cell.strip()
                state.check_tokens(cell, (location,))
                if cs._is_null(cell) or cs._GENERATED_ONLY.match(value):
                    continue
                if not data and (
                    cs.header_is_vocabulary(value)
                    or tabular_privacy.is_declared_schema_header(
                        value,
                        (
                            shared_columns[column]
                            if column < len(shared_columns)
                            else None
                        ),
                    )
                ):
                    # A column name built from the classifier's words is not free
                    # text; a model reading "zip" alone can call it a person.
                    if structured_once.first(value, False):
                        state.scan_text(value, model=False, location=location)
                    continue
                if data and cs._UUID.fullmatch(value):
                    # The sanitizer replaces every whole-cell UUID; one left over is
                    # an original record identifier.
                    state.flag(
                        CATEGORY_IDENTIFYING_TEXT,
                        REASON_CSV_ORIGINAL_IDENTIFIER,
                        location,
                    )
                    continue
                if data and cell in safe_codes.get(column, ()):
                    # A code of a pair proven on this artifact: exact, code-shaped,
                    # one-to-one with its display. Not an identifier by its length.
                    continue
                if (
                    data
                    and cell in safe_displays.get(column, ())
                    and (csv_terminology.terminology_words_only(cell))
                ):
                    terminology_texts.setdefault(cell, None)
                    continue
                shared_column = (
                    shared_columns[column]
                    if data and column < len(shared_columns)
                    else None
                )
                shared = tabular_privacy.occurrence_decision(
                    cell,
                    header=header[column] if data and column < len(header) else "",
                    column=shared_column,
                    row_context=tuple(
                        row[index]
                        for index in (column - 1, column + 1)
                        if 0 <= index < len(row)
                    ),
                )
                if shared.action == tabular_privacy.ACTION_REPLACE:
                    state.flag(
                        shared.entity_type or CATEGORY_IDENTIFYING_TEXT,
                        REASON_CSV_ORIGINAL_IDENTIFIER,
                        location,
                    )
                    if shared_column is not None and (
                        shared_column.role == tabular_privacy.ROLE_DIRECT_IDENTIFIER
                    ):
                        state.flag(
                            CATEGORY_IDENTIFYING_TEXT,
                            REASON_CSV_IDENTIFIER_COLUMN,
                            location,
                        )
                    continue
                if any(
                    cell[start:end] not in state.emitted
                    for start, end, _entity in tabular_privacy.embedded_fixed_identifier_spans(
                        cell
                    )
                ):
                    state.flag(
                        CATEGORY_IDENTIFYING_TEXT,
                        REASON_CSV_ORIGINAL_IDENTIFIER,
                        location,
                    )
                    continue
                if shared.clinical_invariant or shared.content_invariant:
                    # Stable identifier syntax was checked first by the same
                    # policy.  Preserve exact clinical content without asking
                    # an independent model to repeat the same domain error.
                    continue
                if data and cs.is_coded_value(
                    column,
                    value,
                    code_columns,
                    sex_columns,
                    affix_columns,
                    unit_columns,
                    analyte_columns,
                ):
                    continue
                policy = (
                    classes[column].policy
                    if data and column < len(classes)
                    else (cs.POLICY_TEXT)
                )
                if policy == cs.POLICY_PHI and cs._identifier_shaped(
                    cs._PLACEHOLDER_TOKEN.sub("", cs.SURROGATE_PATTERN.sub("", cell))
                ):
                    state.flag(
                        CATEGORY_IDENTIFYING_TEXT,
                        REASON_CSV_IDENTIFIER_COLUMN,
                        location,
                    )
                if policy in (
                    cs.POLICY_DATE,
                    cs.POLICY_BIRTH_YEAR,
                ) and cs._DATE_VALUE.match(value):
                    state.flag(CATEGORY_DATE, REASON_CSV_DATE_COLUMN, location)
                if (data and column in roles.quasi_identifiers) or not cs._has_letter(
                    value
                ):
                    if structured_once.first(value, False):
                        state.scan_text(value, model=False, location=location)
                else:
                    position, first = texts.add(value)
                    # One model call serves every identical cell, but each
                    # cell is a separate physical occurrence: keep a bounded
                    # list so a finding is attributed to all of them.
                    known = (
                        [] if first else list(text_locations.get(str(position)) or [])
                    )
                    if first or len(known) < _AUDIT_DUPLICATE_LOCATIONS:
                        text_locations.put(str(position), [*known, location])
            if data and has_header and any(cell.strip() for cell in row):
                data_rows += 1
                if dataset_columns and (roles.applicable or roles.ambiguous):
                    projected_row = [""] * (max(dataset_columns) + 1)
                    for column in dataset_columns:
                        projected_row[column] = row[column] if column < len(row) else ""
                    if len(projected) <= _DATASET_EXACT_MAX_ROWS:
                        projected.append(projected_row)
                    key = tuple(
                        projected_row[column]
                        for column in sorted(roles.quasi_identifiers)
                    )
                    if len(single) <= 1 or key in single:
                        single[key] = single.get(key, 0) + 1
                    else:
                        single[None] = 1
        position = 0
        for chunk in texts.slices(512):
            for text in chunk:
                state.scan_text(text, locations=text_locations.get(str(position)) or ())
                position += 1
    for text in terminology_texts:
        # The structured rules always read it. A model reading is allowed only
        # the one misreading the shared policy allows, which needs to know
        # which detector said what; a category-only scanner cannot tell, so
        # then no allowance is made.
        state.scan_text(text, model=state.finding_scanner is None)
        if state.finding_scanner is not None:
            findings = list(state.finding_scanner(text))
            if not csv_terminology.terminology_display_accepted(text, findings):
                for finding in findings:
                    if finding.get("blocking"):
                        name = str(finding.get("category") or "")
                        state.flag(
                            (
                                name
                                if _CATEGORY_NAME.fullmatch(name)
                                else CATEGORY_UNSPECIFIED
                            ),
                            REASON_MODEL,
                        )
    if not has_header:
        return
    if tabular_config.profile == tabular_privacy.PROFILE_IDENTIFIER_REMOVAL:
        # The audit holds a table to the same profile it was released under:
        # identifier removal makes no claim about the table's size or shape.
        return
    state.check("dataset_privacy")
    if not (roles.applicable or roles.ambiguous):
        return
    if data_rows <= _DATASET_EXACT_MAX_ROWS:
        satisfied = dataset_privacy.verify_dataset(
            projected,
            roles,
            is_null=cs._is_null,
            policy=tabular_privacy.dataset_policy(tabular_config),
        ).satisfied
    else:
        # Too large to recompute every class in memory: only a table whose
        # rows all share one quasi-identifier tuple can be proven here.
        satisfied = (
            not roles.ambiguous
            and len(single) == 1
            and None not in single
            and (data_rows >= tabular_config.k)
        )
    if not satisfied:
        state.flag(CATEGORY_QUASI_IDENTIFIER, REASON_CSV_DATASET)


def audit_csv_file(
    path, text_scanner=None, finding_scanner=None, provenance: Optional[dict] = None
) -> AuditReport:
    """``audit_released_artifact`` for a CSV file too large to hold: streamed, same rules."""
    state = _Audit(text_scanner, finding_scanner, provenance)
    try:
        _audit_csv_rows(state, _csv_file_rows(path))
    except Exception:
        state.flag(CATEGORY_ARTIFACT, REASON_UNREADABLE)
    return state.report(FORMAT_CSV)


def audit_text_file(
    path, text_scanner=None, window: int = 32 * 1024, overlap: int = 1024
) -> AuditReport:
    """The text channel of a released note, read in overlapping windows, never whole."""
    import codecs

    from services.text_stream import window_bounds

    state = _Audit(text_scanner)
    state.check("text_channels")
    try:
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        buffer = ""
        with open(path, "rb") as stream:
            done = False
            while True:
                while not done and len(buffer) < window + overlap:
                    data = stream.read(256 * 1024)
                    buffer += decoder.decode(data, final=not data)
                    done = not data
                if _CONTROL.search(buffer):
                    raise ValueError("binary content")
                # Word-aligned, as the verifier's windows are: a word cut at
                # the edge of a window is not a name.
                end, advance = window_bounds(buffer, window + overlap, overlap, done)
                if buffer[:end].strip():
                    state.scan_text(buffer[:end])
                if done and len(buffer) <= window + overlap:
                    break
                buffer = buffer[advance:]
    except Exception:
        state.flag(CATEGORY_ARTIFACT, REASON_UNREADABLE)
    return state.report(FORMAT_TEXT)


def _audit_workbook(state: _Audit, content: bytes, _ocr_backend) -> None:
    import openpyxl
    from services import dataset_privacy, ooxml_package, tabular_privacy
    from services import workbook_sanitizer as rules

    inspection = ooxml_package.inspect(content, output=True)
    state.check("workbook_package")
    if not inspection.acceptable:
        reason = (
            REASON_WORKBOOK_LINK
            if any("link" in problem for problem in inspection.problems)
            else REASON_WORKBOOK_PACKAGE
        )
        state.flag(CATEGORY_ARTIFACT, reason, "workbook:package")

    _audit_workbook_retained_text(state, content)

    workbook = openpyxl.load_workbook(
        io.BytesIO(content), data_only=False, keep_links=True
    )
    cached = openpyxl.load_workbook(
        io.BytesIO(content), data_only=True, keep_links=True
    )
    state.check("workbook_hidden_channels")
    try:
        if isinstance(state.provenance, dict) and state.provenance.get(
            "tabular_privacy"
        ):
            tabular_config = tabular_privacy.config_from_metadata(
                state.provenance.get("tabular_privacy")
            )
        else:
            tabular_config = tabular_privacy.PolicyConfig()
    except tabular_privacy.ManifestError:
        state.flag(CATEGORY_ARTIFACT, REASON_TABULAR_POLICY, "workbook:policy")
        tabular_config = tabular_privacy.PolicyConfig()
    properties = workbook.properties
    allowed = {"", *rules._WRITER_DEFAULTS}
    if any(
        str(getattr(properties, name, "") or "").strip() not in allowed
        for name in rules._CORE_PROPERTY_FIELDS
    ):
        state.flag(
            CATEGORY_METADATA, REASON_WORKBOOK_PROPERTIES, "workbook:core_properties"
        )
    if len(getattr(workbook, "custom_doc_props", None) or ()):
        state.flag(
            CATEGORY_METADATA, REASON_WORKBOOK_PROPERTIES, "workbook:custom_properties"
        )
    if getattr(workbook, "_external_links", None):
        state.flag(CATEGORY_LINK, REASON_WORKBOOK_LINK, "workbook:external_links")
    state.check("workbook_defined_names")
    for defined_index, (name, defined) in enumerate(workbook.defined_names.items()):
        name_location = f"workbook:defined_name:{defined_index}"
        if not rules._builtin_defined_name(name):
            state.scan_text(rules.identifier_words(name), location=name_location)
        formula = str(getattr(defined, "attr_text", "") or "")
        if rules._EXTERNAL_REFERENCE.search(formula):
            state.flag(CATEGORY_LINK, REASON_WORKBOOK_LINK, name_location)
        for literal in rules._literals(formula):
            state.scan_text(literal, location=name_location)
    dataset_checks = []
    for sheet_index, sheet in enumerate(workbook.worksheets):
        state.scan_text(
            sheet.title,
            model=not rules._structural_sheet_title(sheet.title),
            location=f"sheet:{sheet_index}:name",
        )
        headers, rows, row_numbers = rules._sheet_table(
            sheet, tabular_config if sheet_index == 0 else None
        )
        try:
            sheet_config = tabular_privacy.config_for_sheet(
                tabular_config, headers, primary=sheet_index == 0
            )
        except tabular_privacy.ManifestError:
            state.flag(
                CATEGORY_ARTIFACT, REASON_TABULAR_POLICY, f"sheet:{sheet_index}:policy"
            )
            sheet_config = tabular_privacy.PolicyConfig()
        columns = (
            tabular_privacy.classify_columns(headers, rows[:1000], sheet_config)
            if headers
            else ()
        )
        roles = tabular_privacy.dataset_roles(columns)
        has_table_header = bool(row_numbers and row_numbers[0] == 2)
        # Every cell, hidden sheets, rows and columns included: search skips them.
        for row_number, row in enumerate(sheet.iter_rows(), start=1):
            row_values = ["" if cell.value is None else str(cell.value) for cell in row]
            for column, cell in enumerate(row):
                location = f"sheet:{sheet_index}:cell:{cell.coordinate}"
                if isinstance(cell.value, str):
                    state.check_tokens(cell.value, (location,))
                if cell.hyperlink is not None:
                    state.flag(CATEGORY_LINK, REASON_WORKBOOK_LINK, location)
                if cell.comment is not None:
                    if (cell.comment.author or "").strip() not in (
                        "",
                        rules.NEUTRAL_AUTHOR,
                    ):
                        state.flag(
                            CATEGORY_PERSON,
                            REASON_WORKBOOK_AUTHOR,
                            location + ":comment_author",
                        )
                    state.scan_text(
                        cell.comment.text or "", location=location + ":comment"
                    )
                if cell.data_type == "f" or (
                    isinstance(cell.value, str) and cell.value.startswith("=")
                ):
                    formula = str(cell.value)
                    if rules._EXTERNAL_REFERENCE.search(formula):
                        state.flag(CATEGORY_LINK, REASON_WORKBOOK_LINK, location)
                    for literal in rules._literals(formula):
                        before = len(state.findings)
                        state.scan_text(literal, location=location + ":formula")
                        if len(state.findings) > before:
                            state.flag(
                                CATEGORY_IDENTIFYING_TEXT,
                                REASON_WORKBOOK_FORMULA,
                                location + ":formula",
                            )
                    cached_cell = cached[sheet.title][cell.coordinate]
                    if isinstance(cached_cell.value, str):
                        state.scan_text(
                            cached_cell.value, location=location + ":formula_cache"
                        )
                    continue
                if isinstance(cell.value, str):
                    if has_table_header and row_number == 1:
                        state.scan_text(cell.value, model=False, location=location)
                        continue
                    if cell.value.strip() in state.emitted:
                        continue
                    policy = columns[column] if column < len(columns) else None
                    shared = tabular_privacy.occurrence_decision(
                        cell.value,
                        header=headers[column] if column < len(headers) else "",
                        column=policy,
                        row_context=tuple(
                            row_values[index]
                            for index in (column - 1, column + 1)
                            if 0 <= index < len(row_values)
                        ),
                    )
                    if shared.action == tabular_privacy.ACTION_REPLACE:
                        state.flag(
                            shared.entity_type or CATEGORY_IDENTIFYING_TEXT,
                            REASON_STRUCTURED,
                            location,
                        )
                    elif any(
                        cell.value[start:end] not in state.emitted
                        for start, end, _entity in tabular_privacy.embedded_fixed_identifier_spans(
                            cell.value
                        )
                    ):
                        state.flag(
                            CATEGORY_IDENTIFYING_TEXT, REASON_STRUCTURED, location
                        )
                    elif shared.content_invariant:
                        continue
                    elif not shared.clinical_invariant:
                        # As for a CSV: a quasi-identifier cell (an age band, a
                        # generalized category) and a cell with no letters are
                        # read by the structured rules only. A model reads an age
                        # band such as "30–39" as part of an SSN; the dataset
                        # check below judges these columns by their equivalence
                        # classes.
                        is_quasi = (
                            has_table_header
                            and row_number > 1
                            and column in roles.quasi_identifiers
                        )
                        state.scan_text(
                            cell.value,
                            model=not (
                                is_quasi or not any(ch.isalpha() for ch in cell.value)
                            ),
                            location=location,
                        )
        for attribute in rules._HEADER_FOOTER:
            block = getattr(sheet.HeaderFooter, attribute, None)
            for part in ("left", "center", "right"):
                item = getattr(block, part, None) if block is not None else None
                if item is not None and item.text:
                    state.scan_text(
                        item.text, location=f"sheet:{sheet_index}:{attribute}:{part}"
                    )
        for validation_index, validation in enumerate(
            sheet.data_validations.dataValidation
        ):
            for field in rules._VALIDATION_FIELDS:
                value = getattr(validation, field, None)
                if value:
                    state.scan_text(
                        str(value),
                        location=f"sheet:{sheet_index}:validation:{validation_index}:{field}",
                    )
            for formula in (validation.formula1, validation.formula2):
                for literal in rules._literals(str(formula or "")):
                    state.scan_text(
                        literal,
                        location=f"sheet:{sheet_index}:validation:{validation_index}:formula",
                    )
        for defined_index, (name, defined) in enumerate(sheet.defined_names.items()):
            name_location = f"sheet:{sheet_index}:defined_name:{defined_index}"
            if not rules._builtin_defined_name(name):
                state.scan_text(rules.identifier_words(name), location=name_location)
            formula = str(getattr(defined, "attr_text", "") or "")
            for literal in rules._literals(formula):
                state.scan_text(literal, location=name_location)
        if tabular_config.qi_enabled and (roles.applicable or roles.ambiguous):
            check = dataset_privacy.verify_dataset(
                rows,
                roles,
                is_null=lambda value: not str(value).strip(),
                policy=tabular_privacy.dataset_policy(tabular_config),
            )
            dataset_checks.append(check)
            if not check.satisfied:
                state.flag(
                    CATEGORY_QUASI_IDENTIFIER,
                    REASON_WORKBOOK_DATASET,
                    f"sheet:{sheet_index}:table",
                )
    if dataset_checks:
        state.check("workbook_dataset_privacy")
    workbook.close()
    cached.close()


def _audit_workbook_retained_text(state: _Audit, content: bytes) -> None:
    """Scan supported ZIP text which a worksheet reader may not expose.

    Unused shared strings, orphan comments, and VML text boxes are retained
    package content even when openpyxl does not project them into cells.  The
    package allowlist alone cannot assert that their text is de-identified.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    from services import ooxml_package, tabular_privacy

    state.check("workbook_retained_package_text")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for name in archive.namelist():
                shared = name == "xl/sharedStrings.xml"
                comments = bool(
                    re.fullmatch(r"xl/(?:comments\d*|comments/comment\d*)\.xml", name)
                )
                vml = bool(
                    re.fullmatch(
                        r"xl/drawings/(?:vmlDrawing|commentsDrawing)\d*\.vml", name
                    )
                )
                if not (shared or comments or vml):
                    continue
                raw = archive.read(name)
                if len(raw) > ooxml_package.MAX_XML_PART_BYTES:
                    state.flag(CATEGORY_ARTIFACT, REASON_UNREADABLE, f"package:{name}")
                    continue
                root = ET.fromstring(raw)
                for node_index, node in enumerate(root.iter()):
                    local = node.tag.rsplit("}", 1)[-1]
                    if shared and local != "si":
                        continue
                    if comments and local not in {"comment", "author"}:
                        continue
                    if vml and local.lower() not in {"textbox", "textpath"}:
                        continue
                    value = "".join(node.itertext()).strip()
                    if not value or value in state.emitted:
                        continue
                    location = f"package:{name}:node:{node_index}"
                    if comments and local == "author":
                        from services import workbook_sanitizer as rules

                        if value != rules.NEUTRAL_AUTHOR:
                            state.flag(
                                CATEGORY_PERSON, REASON_WORKBOOK_AUTHOR, location
                            )
                    unmatched = [
                        (value[start:end], entity)
                        for start, end, entity in tabular_privacy.embedded_fixed_identifier_spans(
                            value
                        )
                        if value[start:end] not in state.emitted
                    ]
                    if unmatched:
                        state.flag(
                            CATEGORY_IDENTIFYING_TEXT, REASON_STRUCTURED, location
                        )
                    state.scan_text(value, location=location)
    except (OSError, ValueError, ET.ParseError, zipfile.BadZipFile):
        state.flag(CATEGORY_ARTIFACT, REASON_UNREADABLE)


def audit_released_artifact(
    content,
    file_format,
    ocr_backend=None,
    text_scanner=None,
    finding_scanner=None,
    provenance=None,
):
    state = _Audit(text_scanner, finding_scanner, provenance)
    handlers = {"text": _audit_text, "csv": _audit_csv, "workbook": _audit_workbook}
    handler = handlers.get(file_format)
    if handler is None:
        state.flag(CATEGORY_ARTIFACT, REASON_UNSUPPORTED)
    else:
        try:
            handler(state, bytes(content), ocr_backend)
        except Exception:
            state.flag(CATEGORY_ARTIFACT, REASON_UNREADABLE)
    return state.report(file_format)
