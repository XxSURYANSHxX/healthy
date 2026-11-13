"""XLSX workbook anonymization: every supported text channel rewritten, the rest held.

A workbook carries identity well beyond its visible cells: hidden sheets and
rows, cell comments and their authors, headers and footers, hyperlinks,
data-validation prompts, defined names, formula text, document and custom
properties, and links to other workbooks. This handler rewrites what it can
rewrite safely and holds what it cannot.

* String cells, comments, headers, footers and validation messages go through
  full_v1 and are written back anonymized; numbers, formulas, formatting and
  sheet structure are left as they were.
* Hyperlinks, links to external workbooks, document properties, custom
  properties and comment authors are removed.
* Typed date cells follow the profile's date strategy; under strict they are
  removed.
* An identifier in a sheet name, a defined name or a formula's own text cannot
  be rewritten without breaking the references that use it, and a formula that
  reaches into another workbook cannot be verified, so those hold the workbook.
  So does a large integer that reads like an identifier.
* Macro-enabled workbooks and objects the installed library would silently
  drop on save - charts, drawings, images, pivot tables, slicers, threaded
  comments, controls, external data connections - are held without an output,
  rather than returned with content quietly missing.

The output is reopened and inspected: sheets, order, dimensions and formulas
unchanged, numbers unchanged, strings equal to their anonymized text, no
hyperlink, external link, property or comment author left, and every text
channel read again by the residual validator.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from io import BytesIO
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from services import dataset_privacy, ooxml_package, tabular_privacy
from services.clinical_measurement import (
    line_reads_as_clinical_narrative,
    line_reads_as_measurement,
)
from services.clinical_vocabulary import (
    CLINICAL_SERVICE_TERMS,
    is_clinical_field_label,
    normalize,
)
from services.multiformat import (
    FORMAT_WORKBOOK,
    SIGNATURE_ZIP,
    MultiformatError,
    MultiformatResult,
    StageClock,
    TextChannelSanitizer,
    verify_extracted_text,
)
from services.privacy_profiles import PrivacyProfileError, get_privacy_profile
from services.text_anonymization import TextAnonymizationError

MAX_WORKBOOK_BYTES = 32 * 1024 * 1024
MAX_SHEETS = 100
MAX_CELLS = 200_000
#: An integer this large reads like an account, phone or record number.
IDENTIFIER_LIKE_INTEGER = 1_000_000

XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_STRING_LITERAL = re.compile(r'"((?:[^"]|"")*)"')
_EXTERNAL_REFERENCE = re.compile(r"\[\d+\]|https?://|file:|\\\\", re.IGNORECASE)
_HEADER_FOOTER = (
    "oddHeader",
    "oddFooter",
    "evenHeader",
    "evenFooter",
    "firstHeader",
    "firstFooter",
)
_CORE_PROPERTY_FIELDS = (
    "creator",
    "title",
    "subject",
    "description",
    "keywords",
    "category",
    "lastModifiedBy",
    "identifier",
    "contentStatus",
    "version",
)
_VALIDATION_FIELDS = ("prompt", "error", "promptTitle", "errorTitle")
#: Stamped in place of the save time. The value is arbitrary and fixed; what
#: matters is that it is not the moment this ran, because that is both a fact
#: about whoever sanitized the file and a reason identical input would produce
#: different bytes.
FIXED_TIMESTAMP = datetime(2000, 1, 1, 0, 0, 0)
#: Written in place of a removed comment author, which the format requires.
NEUTRAL_AUTHOR = "anonymized"
#: Values the writer fills in by itself when a property is left empty.
_WRITER_DEFAULTS = frozenset({"openpyxl", NEUTRAL_AUTHOR})

#: Workbook navigation roles and clinical-service words are field provenance,
#: not identity. Every word must be structural; one unrecognised token sends
#: the complete title through the normal identifier detectors.
_SHEET_ROLE_WORDS = frozenset(
    {
        "sheet",
        "data",
        "archive",
        "summary",
        "result",
        "results",
        "report",
        "reports",
        "note",
        "notes",
        "laboratory",
        "labs",
        "input",
        "output",
        "table",
        "form",
        "record",
        "records",
        "clinical",
        "medical",
    }
)


def _structural_sheet_title(title: str) -> bool:
    tokens = [token for token in re.split(r"[^a-z0-9]+", normalize(title)) if token]
    return bool(tokens) and all(
        token in _SHEET_ROLE_WORDS
        or token in CLINICAL_SERVICE_TERMS
        or is_clinical_field_label(token)
        for token in tokens
    )


REASON_CONTAINER = "workbook_container_unsafe"
REASON_MACROS = "workbook_macros_present"
#: A macro-enabled upload was converted. Reported as a warning rather than
#: a hold, because the file is released - without its macros, which the
#: person downloading it has to be told.
REASON_MACROS_REMOVED = "workbook_macros_removed"
REASON_UNSUPPORTED_OBJECT = "workbook_unsupported_object_present"
REASON_UNPARSEABLE = "workbook_unparseable"
REASON_SHEET_LIMIT = "workbook_sheet_limit_exceeded"
REASON_CELL_LIMIT = "workbook_cell_limit_exceeded"
REASON_PHI_IN_FORMULA = "workbook_phi_in_formula_text"
REASON_EXTERNAL_FORMULA = "workbook_formula_references_external_workbook"
REASON_PHI_IN_SHEET_NAME = "workbook_phi_in_sheet_name"
REASON_PHI_IN_DEFINED_NAME = "workbook_phi_in_defined_name"
REASON_REMOVED_DEFINED_NAME = "workbook_removed_defined_name_uncertain"
REASON_NUMERIC_IDENTIFIER = "workbook_numeric_identifier_suspected"
REASON_TEXT_REVIEW = "workbook_text_requires_review"
REASON_ANONYMIZER_RESIDUAL = "privacy_requirements_not_met"
REASON_TEXT_LIMIT = "workbook_text_channel_exceeds_scan_limit"
REASON_WRITE_FAILED = "workbook_output_write_failed"
REASON_REOPEN_FAILED = "workbook_output_reopen_failed"
REASON_STRUCTURE_CHANGED = "workbook_structure_changed"
REASON_VALUE_CHANGED = "workbook_value_changed"
#: An original value found anywhere in the written package, not only in
#: the grid: a shared string, a comment, a property or a relationship.
REASON_VALUE_SURVIVED = "workbook_original_value_survived_in_package"
REASON_HIDDEN_CHANNEL = "workbook_hidden_channel_survived"
REASON_RESIDUAL = "workbook_residual_phi_detected"
REASON_CLINICAL_CHANGED = tabular_privacy.REASON_CLINICAL_CHANGED
REASON_QI_NOT_VERIFIED = "workbook_dataset_privacy_not_verified"

VALIDATORS = (
    "full_v1_text_anonymization",
    "workbook_channel_rewrite",
    "workbook_hidden_channel_removal",
    "workbook_reopened_output_inspection",
    "residual_phi_rescan",
)


def _load_openpyxl():
    import openpyxl

    return openpyxl


def _settings(profile: str) -> Dict[str, Any]:
    try:
        return get_privacy_profile(profile)
    except PrivacyProfileError as exc:
        raise MultiformatError(
            "Invalid privacy profile for workbook anonymization"
        ) from exc


def _identifier_like(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return abs(value) >= IDENTIFIER_LIKE_INTEGER
    if isinstance(value, float) and value.is_integer():
        return abs(value) >= IDENTIFIER_LIKE_INTEGER
    return False


def _literals(formula: str) -> List[str]:
    return [
        match.group(1).replace('""', '"') for match in _STRING_LITERAL.finditer(formula)
    ]


class _Slot:
    """Somewhere a rewritten string goes back to."""

    def __init__(self, owner: Any, attribute: str, is_cell: bool = False):
        self.owner = owner
        self.attribute = attribute
        self.is_cell = is_cell

    def write(self, text: str) -> None:
        if self.is_cell:
            self.owner.value = text
            self.owner.data_type = "s"
        else:
            setattr(self.owner, self.attribute, text)


class _FormulaGroup:
    """String literals of one formula, rewritten without touching references."""

    def __init__(
        self,
        owner: Any,
        attribute: str,
        formula: str,
        key: Optional[Tuple[str, str]] = None,
    ):
        self.owner = owner
        self.attribute = attribute
        self.formula = formula
        self.key = key
        self.values = _literals(formula)
        self.rewritten = list(self.values)

    def apply(self) -> str:
        iterator = iter(self.rewritten)

        def replace(_match) -> str:
            value = next(iterator).replace('"', '""')
            return f'"{value}"'

        output = _STRING_LITERAL.sub(replace, self.formula)
        setattr(self.owner, self.attribute, output)
        return output


class _FormulaSlot:
    is_cell = False

    def __init__(self, group: _FormulaGroup, index: int):
        self.group = group
        self.index = index

    def write(self, text: str) -> None:
        self.group.rewritten[self.index] = text


@dataclass
class _WorkbookTablePlan:
    sheet: Any
    columns: Tuple[tabular_privacy.ColumnPolicy, ...]
    roles: dataset_privacy.ColumnRoles
    data_rows: Tuple[int, ...]


@dataclass
class _WorkbookPolicyState:
    columns: Dict[str, Tuple[tabular_privacy.ColumnPolicy, ...]] = field(
        default_factory=dict
    )
    plans: List[_WorkbookTablePlan] = field(default_factory=list)
    protected_cells: Set[int] = field(default_factory=set)
    clinical_values: Dict[int, str] = field(default_factory=dict)
    content_values: Dict[int, Any] = field(default_factory=dict)
    replaced_originals: List[str] = field(default_factory=list)
    emitted_tokens: Set[str] = field(default_factory=set)
    category_counts: Dict[str, int] = field(default_factory=dict)
    transformations: Dict[str, int] = field(
        default_factory=lambda: {
            "cells_changed": 0,
            "cells_generalized": 0,
            "cells_suppressed": 0,
        }
    )
    dataset_summaries: List[Dict[str, Any]] = field(default_factory=list)


def _sheet_table(
    sheet: Any, config: Optional[tabular_privacy.PolicyConfig] = None
) -> Tuple[List[str], List[List[str]], List[int]]:
    """A conservative row-one table view; formulas are never data values here."""
    from services import csv_sanitizer as cs

    width = int(getattr(sheet, "max_column", 0) or 0)
    height = int(getattr(sheet, "max_row", 0) or 0)
    if width <= 0 or height <= 0:
        return [], [], []
    headers = [str(sheet.cell(1, column).value or "") for column in range(1, width + 1)]
    # Treat row one as a schema only when it is populated, textual and followed
    # by repeated records.  A key/value worksheet (labels in column A and
    # values in B) is not a table whose first row declares every B value.
    textual_header = any(value.strip() for value in headers) and not any(
        sheet.cell(1, column).value is not None
        and not isinstance(sheet.cell(1, column).value, str)
        for column in range(1, width + 1)
    )
    header_width = sum(bool(value.strip()) for value in headers)
    recognized = 0
    if textual_header and header_width:
        schema = tabular_privacy.classify_columns(headers)
        declared = {
            item.index
            for item in (config.columns if config is not None else ())
            if item.index is not None
            and (item.header is not None or item.header_sha256 is not None)
            and item.index < len(headers)
            and tabular_privacy.header_matches(item, headers[item.index])
        }
        recognized = sum(
            bool(value.strip())
            and (
                index in declared
                or item.role != tabular_privacy.ROLE_UNKNOWN
                or cs.header_is_vocabulary(value)
                or value.strip().casefold() == "note"
            )
            for index, (value, item) in enumerate(zip(headers, schema))
        )
        textual_header = recognized * 3 >= header_width * 2 and not any(
            re.search(
                r"\b\d{3}[-.\s]\d{4}\b|@|https?://|<[A-Z][A-Z0-9_]*>", value, re.I
            )
            for value in headers
        )
    record_like = 0
    if textual_header and header_width >= 2:
        for row_number in range(2, min(height, 12) + 1):
            populated = sum(
                sheet.cell(row_number, column).value is not None
                for column in range(1, width + 1)
            )
            if populated >= max(2, (header_width + 1) // 2):
                record_like += 1
    minimum_records = 1 if width <= 2 and recognized == header_width else 2
    if not textual_header or header_width < 2 or record_like < minimum_records:
        headers = [f"column_{index}" for index in range(width)]
        first = 1
    else:
        first = 2
    rows: List[List[str]] = []
    row_numbers: List[int] = []
    for row_number in range(first, height + 1):
        values = [
            (
                ""
                if sheet.cell(row_number, column).value is None
                else str(sheet.cell(row_number, column).value)
            )
            for column in range(1, width + 1)
        ]
        if any(value.strip() for value in values):
            rows.append(values)
            row_numbers.append(row_number)
    return headers, rows, row_numbers


def _apply_tabular_policy(
    workbook: Any,
    result: MultiformatResult,
    profile: str,
    config: tabular_privacy.PolicyConfig,
    allocator: Any,
) -> _WorkbookPolicyState:
    """Apply occurrence decisions and the configured QI policy to every sheet."""
    from services.text_anonymization import (
        AGE_AGGREGATE_REPLACEMENT,
        replacement_for_value,
    )

    state = _WorkbookPolicyState()
    privacy_policy = tabular_privacy.dataset_policy(config)
    for sheet_index, sheet in enumerate(workbook.worksheets):
        headers, rows, row_numbers = _sheet_table(
            sheet, config if sheet_index == 0 else None
        )
        if not headers:
            state.columns[sheet.title] = ()
            continue
        sheet_config = tabular_privacy.config_for_sheet(
            config, headers, primary=sheet_index == 0
        )
        columns = tabular_privacy.classify_columns(headers, rows[:1000], sheet_config)
        state.columns[sheet.title] = columns
        if row_numbers and row_numbers[0] == 2:
            # A proven table header is schema, not a first-row narrative. The
            # models can misread short field names such as "ssn" or "icd10" as
            # values and destroy the mapping needed by final-byte audit.
            for index, header in enumerate(headers):
                cell = sheet.cell(1, index + 1)
                state.protected_cells.add(id(cell))
                state.content_values[id(cell)] = header
        roles = tabular_privacy.dataset_roles(columns)
        for row_values, row_number in zip(rows, row_numbers):
            for index, original in enumerate(row_values):
                cell = sheet.cell(row_number, index + 1)
                value = cell.value
                if isinstance(value, str):
                    allocator.reserve_existing(value)
                neighbours = tuple(
                    row_values[near]
                    for near in (index - 1, index + 1)
                    if 0 <= near < len(row_values)
                )
                decision = tabular_privacy.occurrence_decision(
                    value,
                    header=headers[index],
                    column=columns[index],
                    row_context=neighbours,
                )
                if decision.reason_code:
                    result.hold(decision.reason_code)
                if (
                    decision.action == tabular_privacy.ACTION_REPLACE
                    and decision.entity_type
                ):
                    source = "" if value is None else str(value)
                    provider = bool(
                        decision.entity_type == "PERSON"
                        and frozenset(tabular_privacy.header_tokens(headers[index]))
                        & {"provider", "physician", "doctor", "clinician", "nurse"}
                    )
                    replacement = replacement_for_value(
                        decision.entity_type,
                        source,
                        profile=profile,
                        allocator=allocator,
                        is_provider=provider,
                    )
                    if replacement != source:
                        state.replaced_originals.append(source)
                        state.transformations["cells_changed"] += 1
                        cell.value = replacement
                        cell.data_type = "s"
                        if re.fullmatch(r"<[A-Z][A-Z0-9_]*>", replacement):
                            state.emitted_tokens.add(replacement)
                    state.protected_cells.add(id(cell))
                    state.category_counts[decision.entity_type] = (
                        state.category_counts.get(decision.entity_type, 0) + 1
                    )
                elif decision.clinical_invariant:
                    state.protected_cells.add(id(cell))
                    state.clinical_values[id(cell)] = value
                elif decision.content_invariant:
                    state.protected_cells.add(id(cell))
                    # A declared QI is stable input to generalization, not an
                    # exact-output invariant in this profile. The written
                    # class is verified below.
                    if not (
                        config.qi_enabled
                        and columns[index].role == tabular_privacy.ROLE_QUASI_IDENTIFIER
                    ):
                        state.content_values[id(cell)] = value
                elif (
                    isinstance(value, (datetime, date))
                    and columns[index].explicit
                    and columns[index].role == tabular_privacy.ROLE_NON_SENSITIVE
                ):
                    # Explicitly declared document/fixture dates are not dates
                    # about an individual.  Their exact typed value is an
                    # invariant of the workbook rather than patient data.
                    state.protected_cells.add(id(cell))
                    state.content_values[id(cell)] = value

                if (
                    isinstance(value, str)
                    and decision.action != tabular_privacy.ACTION_REPLACE
                ):
                    inherited = tabular_privacy.confirmed_row_identifier_spans(
                        value, row_values, columns, index
                    )
                    if inherited and decision.clinical_invariant:
                        result.hold("tabular_identifier_clinical_collision")
                    elif inherited:
                        rewritten = value
                        for start, end, entity in reversed(inherited):
                            source = value[start:end]
                            source_column = next(
                                (
                                    item
                                    for item in columns
                                    if item.index != index
                                    and item.role
                                    == tabular_privacy.ROLE_DIRECT_IDENTIFIER
                                    and item.entity_type == entity
                                    and item.index < len(row_values)
                                    and str(row_values[item.index]).strip().casefold()
                                    == source.casefold()
                                ),
                                None,
                            )
                            if source_column is None:
                                continue
                            original = str(row_values[source_column.index]).strip()
                            provider = bool(
                                entity == "PERSON"
                                and set(
                                    tabular_privacy.header_tokens(
                                        headers[source_column.index]
                                    )
                                )
                                & {
                                    "provider",
                                    "physician",
                                    "doctor",
                                    "clinician",
                                    "nurse",
                                }
                            )
                            token = replacement_for_value(
                                entity,
                                original,
                                profile=profile,
                                allocator=allocator,
                                is_provider=provider,
                            )
                            rewritten = rewritten[:start] + token + rewritten[end:]
                            state.replaced_originals.append(source)
                            state.category_counts[entity] = (
                                state.category_counts.get(entity, 0) + 1
                            )
                            if re.fullmatch(r"<[A-Z][A-Z0-9_]*>", token):
                                state.emitted_tokens.add(token)
                        if rewritten != value:
                            cell.value = rewritten
                            state.transformations["cells_changed"] += 1
                    fixed = tabular_privacy.embedded_fixed_identifier_spans(value)
                    if fixed and decision.clinical_invariant:
                        result.hold("tabular_identifier_clinical_collision")
                    elif fixed:
                        changed = False
                        for start, end, entity in fixed:
                            source = value[start:end]
                            token = replacement_for_value(
                                entity, source, profile=profile, allocator=allocator
                            )
                            rewritten, count = re.subn(
                                re.escape(source), lambda _match: token, str(cell.value)
                            )
                            if count:
                                cell.value = rewritten
                                changed = True
                                state.replaced_originals.append(source)
                                state.category_counts[entity] = (
                                    state.category_counts.get(entity, 0) + count
                                )
                                if re.fullmatch(r"<[A-Z][A-Z0-9_]*>", token):
                                    state.emitted_tokens.add(token)
                        if changed and not inherited:
                            state.transformations["cells_changed"] += 1

        if not config.qi_enabled or not (roles.applicable or roles.ambiguous):
            continue
        projected = [
            [
                (
                    ""
                    if sheet.cell(row_number, column + 1).value is None
                    else str(sheet.cell(row_number, column + 1).value)
                )
                for column in range(len(headers))
            ]
            for row_number in row_numbers
        ]
        outcome = dataset_privacy.anonymize_dataset(
            projected,
            roles,
            is_null=lambda value: not str(value).strip(),
            placeholders=(AGE_AGGREGATE_REPLACEMENT,),
            policy=privacy_policy,
        )
        state.dataset_summaries.append(outcome.summary)
        if outcome.reason_codes:
            result.hold(*outcome.reason_codes)
        for row_index, column in outcome.generalized:
            cell = sheet.cell(row_numbers[row_index], column + 1)
            cell.value = outcome.rows[row_index][column]
            cell.data_type = "s"
            state.protected_cells.add(id(cell))
            # A QI category can be generalized in one equivalence class and
            # legitimately remain in another. The exact-output dataset check
            # proves k/l/t for every written cell; the package-wide original
            # search is for direct identifiers, whose recurrence is a leak.
            state.transformations["cells_generalized"] += 1
            if outcome.rows[row_index][column] == dataset_privacy.SUPPRESSED:
                state.transformations["cells_suppressed"] += 1
        if not outcome.reason_codes:
            state.plans.append(
                _WorkbookTablePlan(sheet, columns, roles, tuple(row_numbers))
            )
    return state


#: The date written on every entry of the archive. A ZIP records when each
#: member was added, which is the moment of the save.
_FIXED_ZIP_DATE = (2000, 1, 1, 0, 0, 0)
_CORE_TIMESTAMP = re.compile(
    rb"(<dcterms:(?:created|modified)[^>]*>)[^<]*(</dcterms:(?:created|modified)>)"
)
_FIXED_CORE_TIMESTAMP = b"2000-01-01T00:00:00Z"


def _repacked(output: bytes) -> bytes:
    """The same parts, written again with no record of when.

    Two runs over identical input produced different bytes, in two places:
    the writer stamps dcterms:modified with the moment of the save whatever
    the properties were set to, and a ZIP records a date on every member.
    Neither says anything about the document and both say something about
    whoever sanitized it.

    Nothing is added, removed or reordered here - the parts are the ones the
    allowlist accepted, re-emitted - so this repack is not a way for
    something unverified to enter. What it buys is an artifact a caller can
    hash today and hash again tomorrow.
    """
    source = zipfile.ZipFile(BytesIO(output))
    target = BytesIO()
    try:
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as written:
            for info in source.infolist():
                data = source.read(info.filename)
                if info.filename == "docProps/core.xml":
                    # A function rather than a backreference: the opening
                    # and closing tags are put back explicitly, so the
                    # substitution cannot silently drop them.
                    data = _CORE_TIMESTAMP.sub(
                        lambda match: (
                            match.group(1) + _FIXED_CORE_TIMESTAMP + match.group(2)
                        ),
                        data,
                    )
                entry = zipfile.ZipInfo(info.filename, date_time=_FIXED_ZIP_DATE)
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = info.external_attr
                written.writestr(entry, data)
    finally:
        source.close()
    return target.getvalue()


def _row_reading(row, value: str) -> str:
    """One row as "head: value", or "" when the row has no head.

    A spreadsheet puts the label in one cell and the value in the next, and a
    channel built cell by cell throws that relationship away. This puts it
    back for the one question that needs it: is this cell a finding or a
    party.
    """
    head = ""
    for cell in row:
        text = cell.value
        if isinstance(text, str) and text.strip():
            if text == value:
                break
            if is_clinical_field_label(text):
                head = text.strip()
                break
    if not head:
        return ""
    return f"{head}: {value}"


def anonymize_workbook_bytes(
    content: bytes,
    profile: str = "strict",
    *,
    tabular_manifest: Optional[Mapping[str, Any]] = None,
    tabular_profile: Optional[str] = None,
) -> MultiformatResult:
    """Sanitize an .xlsx workbook into a new file and verify the file itself."""
    if not isinstance(content, (bytes, bytearray)):
        raise MultiformatError("Workbook content must be bytes", status_code=500)
    if not content:
        raise MultiformatError("Uploaded workbook is empty")
    if len(content) > MAX_WORKBOOK_BYTES:
        raise MultiformatError(
            f"Workbook uploads must be {MAX_WORKBOOK_BYTES} bytes or smaller",
            status_code=413,
        )
    raw = bytes(content)
    if not raw.startswith(b"PK\x03\x04"):
        raise MultiformatError("File is not an .xlsx workbook", status_code=415)

    settings = _settings(profile)
    try:
        tabular_config = tabular_privacy.policy_config(
            tabular_manifest, profile=tabular_profile
        )
    except tabular_privacy.ManifestError as exc:
        raise MultiformatError(
            "Invalid tabular privacy policy", status_code=400
        ) from exc
    result = MultiformatResult(FORMAT_WORKBOOK, XLSX_MEDIA_TYPE, SIGNATURE_ZIP)
    clock = StageClock("workbook")

    # Every part and every relationship, against an allowlist. What came
    # before listed the parts known to be trouble and let the rest through,
    # which is a statement about what had been thought of rather than about
    # what is in the file.
    # A macro-enabled package is asked about by its bytes, not its name: a
    # file called .xlsx that carries a macro project is still one.
    macro_enabled = ooxml_package.is_macro_enabled(raw)
    inspection = ooxml_package.inspect(raw, allow_macro_project=macro_enabled)
    result.summary["package"] = inspection.safe_summary()
    result.summary["macro_enabled_input"] = macro_enabled
    if inspection.unknown_parts:
        result.summary["unsupported_parts"] = list(inspection.unknown_parts)
    if not inspection.acceptable:
        if ooxml_package.REASON_NOT_A_WORKBOOK in inspection.problems:
            raise MultiformatError("File is not an .xlsx workbook", status_code=415)
        # The package inspector speaks about packages; a caller asking about
        # a workbook already knows workbook_macros_present. Both are reported
        # so the established code keeps its meaning and the new one adds the
        # detail.
        if ooxml_package.REASON_MACRO_PROJECT in inspection.problems:
            result.hold(REASON_MACROS)
        unsupported = {
            ooxml_package.REASON_UNKNOWN_PART,
            ooxml_package.REASON_EMBEDDED_OBJECT,
            ooxml_package.REASON_ACTIVE_CONTENT,
        }
        if unsupported & set(inspection.problems):
            result.hold(REASON_UNSUPPORTED_OBJECT)
        if ooxml_package.REASON_CONTAINER in inspection.problems:
            result.hold(REASON_CONTAINER)
            result.hold(*inspection.problems)
            # An encrypted or foreign container cannot be opened: no copy.
            return clock.finish(result)
        result.hold(*inspection.problems)
        # A surface this profile does not read (a picture, a chart, an
        # embedded object, a control wired to a macro) holds the workbook. The
        # cells are still sanitized and a macro-free workbook without those
        # objects is written as the owner's review copy - never released.
        result.summary["best_effort_review_copy"] = True

    if macro_enabled:
        # The project is dropped rather than carried: openpyxl is asked for a
        # macro-free load, nothing executes, and vbaProject.bin is never read
        # or copied. The artifact is an .xlsx and the caller is told so.
        result.warn(REASON_MACROS_REMOVED)
        result.summary["macro_parts_removed"] = list(inspection.macro_parts)

    try:
        _sanitize(raw, result, clock, profile, settings, tabular_config)
    except tabular_privacy.ManifestError as exc:
        raise MultiformatError(
            "Tabular policy contradicts workbook schema", status_code=400
        ) from exc
    return clock.finish(result)


def _sanitize(raw, result, clock, profile, settings, tabular_config) -> None:
    openpyxl = _load_openpyxl()
    with clock.stage("parse"):
        try:
            workbook = openpyxl.load_workbook(
                BytesIO(raw), data_only=False, keep_links=False, keep_vba=False
            )
        except Exception:
            workbook = None
    if workbook is None:
        result.error(REASON_UNPARSEABLE)
        return
    try:
        _process(
            workbook, raw, result, clock, profile, settings, openpyxl, tabular_config
        )
    finally:
        workbook.close()


def _process(
    workbook,
    raw,
    result,
    clock,
    profile,
    settings,
    openpyxl,
    tabular_config: tabular_privacy.PolicyConfig,
) -> None:
    sheets = list(workbook.worksheets)
    if len(workbook.sheetnames) > MAX_SHEETS:
        result.hold(REASON_SHEET_LIMIT)
        return

    rewrite_texts: List[str] = []
    rewrite_slots: List[_Slot] = []
    #: Text this pass decided was clinical and left in place. The same word
    #: can be a heading in one sheet and a bare first cell in another, where
    #: it has no head to be read with and is rewritten; finding it in the
    #: package afterwards is then a disagreement between two cells, not a
    #: value that escaped, and reporting it as a leak held a clean workbook.
    kept_clinical: set = set()
    detect_texts: List[Tuple[str, str]] = []
    counts = {
        "sheets": len(workbook.sheetnames),
        "hidden_sheets": 0,
        "string_cells": 0,
        "formula_cells": 0,
        "date_cells_removed": 0,
        "comments": 0,
        "hyperlinks_removed": 0,
        "header_footer_parts": 0,
        "validation_messages": 0,
        "defined_names": 0,
        "document_properties_removed": 0,
        "custom_properties_removed": 0,
    }
    formulas: Dict[Tuple[str, str], str] = {}
    formula_groups: List[_FormulaGroup] = []
    numbers: Dict[Tuple[str, str], Any] = {}
    removed_defined_names: List[Tuple[Any, str]] = []
    from services.surrogates import SurrogateAllocator

    # One allocator for every workbook occurrence, deterministic rules and
    # model-backed text alike.  Classification remains occurrence-scoped;
    # only confirmed identities share a token.
    allocator = SurrogateAllocator()
    policy_state = _apply_tabular_policy(
        workbook, result, profile, tabular_config, allocator
    )
    result.summary["tabular_policy_version"] = tabular_privacy.POLICY_VERSION
    result.summary["workbook_surfaces"] = list(tabular_privacy.WORKBOOK_SURFACES)

    with clock.stage("extract"):
        cells = 0
        for sheet in sheets:
            if sheet.sheet_state != "visible":
                counts["hidden_sheets"] += 1
            if _structural_sheet_title(sheet.title):
                counts["structural_sheet_names_kept"] = (
                    counts.get("structural_sheet_names_kept", 0) + 1
                )
            else:
                detect_texts.append(("sheet_name", sheet.title))
            for row in sheet.iter_rows():
                for cell in row:
                    value = cell.value
                    if value is not None:
                        cells += 1
                        if cells > MAX_CELLS:
                            result.hold(REASON_CELL_LIMIT)
                            return
                    key = (sheet.title, cell.coordinate)
                    if cell.data_type == "f" or (
                        isinstance(value, str) and value.startswith("=")
                    ):
                        counts["formula_cells"] += 1
                        formula = str(value)
                        formulas[key] = formula
                        if _EXTERNAL_REFERENCE.search(formula):
                            result.hold(REASON_EXTERNAL_FORMULA)
                        group = _FormulaGroup(cell, "value", formula, key)
                        if group.values:
                            formula_groups.append(group)
                            for index, literal in enumerate(group.values):
                                rewrite_texts.append(literal)
                                rewrite_slots.append(_FormulaSlot(group, index))
                                detect_texts.append(("formula", literal))
                    elif isinstance(value, str):
                        counts["string_cells"] += 1
                        if id(cell) in policy_state.protected_cells:
                            kept_clinical.add(value.strip())
                        row_line = _row_reading(row, value)
                        if (
                            id(cell) not in policy_state.protected_cells
                            and row_line
                            and (
                                line_reads_as_measurement(row_line)
                                or line_reads_as_clinical_narrative(row_line)
                            )
                        ):
                            # "Diagnosis | D64.9", "Analyte | haemoglobin",
                            # "Unit | g/L". Read on its own a value cell is
                            # unanswerable - D64.9 is identifier-shaped,
                            # haemoglobin is a proper-noun-shaped word - and
                            # read that way all three were rewritten away.
                            # Read with the head of its own row it is a
                            # finding. Keeping it does not skip verification:
                            # the residual pass still reads the written file,
                            # so a name kept here is caught there.
                            kept = counts.get("clinical_cells_kept", 0)
                            counts["clinical_cells_kept"] = kept + 1
                            kept_clinical.add(value.strip())
                        elif id(
                            cell
                        ) not in policy_state.protected_cells and _column_keeps(
                            sheet, cell, value
                        ):
                            # A column the header row names as analytes or
                            # units keeps what reads as one by the same rule a
                            # CSV column does; anything person- or
                            # identifier-shaped still goes to the detectors.
                            kept = counts.get("clinical_cells_kept", 0)
                            counts["clinical_cells_kept"] = kept + 1
                            kept_clinical.add(value.strip())
                        elif id(
                            cell
                        ) not in policy_state.protected_cells and is_clinical_field_label(
                            value
                        ):
                            # A column of headings is the structure of the
                            # sheet, not its content. Sent for rewriting,
                            # "Analyte" came back as <ORGANIZATION_1> and
                            # "Telephone" as a phone number - the sheet lost
                            # the words that say what its values are.
                            kept = counts.get("field_labels_kept", 0)
                            counts["field_labels_kept"] = kept + 1
                            kept_clinical.add(value.strip())
                        elif id(cell) not in policy_state.protected_cells:
                            rewrite_texts.append(value)
                            rewrite_slots.append(_Slot(cell, "value", is_cell=True))
                    elif isinstance(value, (datetime, date)) and not isinstance(
                        value, time
                    ):
                        if id(cell) not in policy_state.protected_cells:
                            cell.value = _date_value(value, settings)
                            counts["date_cells_removed"] += 1
                    elif isinstance(value, (int, float)) and not isinstance(
                        value, bool
                    ):
                        numbers[key] = value
                        if id(
                            cell
                        ) not in policy_state.protected_cells and _identifier_like(
                            value
                        ):
                            result.hold(REASON_NUMERIC_IDENTIFIER)
                    if cell.hyperlink is not None:
                        link = cell.hyperlink
                        detect_texts.extend(
                            ("hyperlink", str(part))
                            for part in (
                                link.target,
                                link.tooltip,
                                link.display,
                                link.location,
                            )
                            if part
                        )
                    if cell.comment is not None:
                        counts["comments"] += 1
                        rewrite_texts.append(cell.comment.text or "")
                        rewrite_slots.append(_Slot(cell.comment, "text"))
                        detect_texts.append(
                            ("comment_author", cell.comment.author or "")
                        )
            for attribute in _HEADER_FOOTER:
                block = getattr(sheet.HeaderFooter, attribute, None)
                for part in ("left", "center", "right"):
                    item = getattr(block, part, None) if block is not None else None
                    if item is not None and item.text:
                        counts["header_footer_parts"] += 1
                        rewrite_texts.append(item.text)
                        rewrite_slots.append(_Slot(item, "text"))
            for validation in sheet.data_validations.dataValidation:
                for field in _VALIDATION_FIELDS:
                    text = getattr(validation, field, None)
                    if text:
                        counts["validation_messages"] += 1
                        rewrite_texts.append(text)
                        rewrite_slots.append(_Slot(validation, field))
                for attribute, formula in (
                    ("formula1", validation.formula1),
                    ("formula2", validation.formula2),
                ):
                    if formula:
                        group = _FormulaGroup(validation, attribute, str(formula))
                        if group.values:
                            formula_groups.append(group)
                            for index, literal in enumerate(group.values):
                                rewrite_texts.append(literal)
                                rewrite_slots.append(_FormulaSlot(group, index))
                                detect_texts.append(("formula", literal))
            for name, defined in list(sheet.defined_names.items()):
                counts["defined_names"] += 1
                if _builtin_defined_name(name):
                    continue
                removed = not _defined_name_referenced(workbook, name, defined)
                kind = "removed_defined_name" if removed else "defined_name"
                detect_texts.append((kind, identifier_words(name)))
                _defined_name_texts(
                    defined,
                    detect_texts,
                    result,
                    "removed_defined_name_formula" if removed else "formula",
                )
                if not removed and _literals(
                    str(getattr(defined, "attr_text", "") or "")
                ):
                    group = _FormulaGroup(defined, "attr_text", str(defined.attr_text))
                    formula_groups.append(group)
                    for index, literal in enumerate(group.values):
                        rewrite_texts.append(literal)
                        rewrite_slots.append(_FormulaSlot(group, index))
                if removed:
                    removed_defined_names.append((sheet.defined_names, name))
        for name, defined in list(workbook.defined_names.items()):
            counts["defined_names"] += 1
            if _builtin_defined_name(name):
                continue
            removed = not _defined_name_referenced(workbook, name, defined)
            kind = "removed_defined_name" if removed else "defined_name"
            detect_texts.append((kind, identifier_words(name)))
            _defined_name_texts(
                defined,
                detect_texts,
                result,
                "removed_defined_name_formula" if removed else "formula",
            )
            if not removed and _literals(str(getattr(defined, "attr_text", "") or "")):
                group = _FormulaGroup(defined, "attr_text", str(defined.attr_text))
                formula_groups.append(group)
                for index, literal in enumerate(group.values):
                    rewrite_texts.append(literal)
                    rewrite_slots.append(_FormulaSlot(group, index))
            if removed:
                removed_defined_names.append((workbook.defined_names, name))
        for field in _CORE_PROPERTY_FIELDS:
            value = getattr(workbook.properties, field, None)
            if value:
                counts["document_properties_removed"] += 1
                detect_texts.append(("property", str(value)))
        for prop in list(workbook.custom_doc_props.props):
            counts["custom_properties_removed"] += 1
            detect_texts.append(
                ("property", f"{prop.name} {getattr(prop, 'value', '')}")
            )
    if counts["date_cells_removed"]:
        result.warn("workbook_date_cells_removed")

    with clock.stage("anonymize_text"):
        rewrite = TextChannelSanitizer(profile, allocator)
        detect = TextChannelSanitizer(profile, allocator)
        try:
            # Workbook surfaces are model-isolation boundaries. The worker
            # may batch two independently indexed inputs, but every policy
            # decision and replacement is still made on one exact surface.
            rewritten = rewrite.sanitize_isolated(
                rewrite_texts,
                [
                    f"rewrite:{index}:{type(slot).__name__}"
                    for index, slot in enumerate(rewrite_slots)
                ],
            )
            detected = detect.sanitize_isolated(
                [text for _kind, text in detect_texts],
                [
                    f"detect:{kind}:{index}"
                    for index, (kind, _text) in enumerate(detect_texts)
                ],
            )
        except TextAnonymizationError as exc:
            if exc.detail != "text_channel_exceeds_scan_limit":
                raise
            result.hold(REASON_TEXT_LIMIT)
            return
    result.summary["model_calls"] = rewrite.calls + detect.calls
    # Keep the detector's review verdict attached to the exact workbook
    # occurrence it read. Sheet indexes and coordinates are safe to publish;
    # sheet names and cell values are not. Non-cell surfaces keep a stable
    # ordinal until the package-specific locator is available.
    sheet_ordinals = {id(sheet): index for index, sheet in enumerate(sheets)}
    review_findings: List[Dict[str, Any]] = []

    def note_finding(
        location: Dict[str, Any], reason: str, category: str, changed: bool
    ) -> None:
        finding = {
            **location,
            "reason_code": reason,
            "category": category,
            "value_transformed": changed,
        }
        review_findings.append(finding)

    for ordinal, (slot, outcome) in enumerate(zip(rewrite_slots, rewritten)):
        location = {
            "surface": "workbook_cell" if slot.is_cell else "workbook_surface",
            "surface_ordinal": ordinal,
        }
        if slot.is_cell:
            location.update(
                {
                    "sheet_index": sheet_ordinals[id(slot.owner.parent)],
                    "row": slot.owner.row,
                    "column_index": slot.owner.column - 1,
                }
            )
        for reason in outcome.review_reasons:
            note_finding(location, reason, "UNCLASSIFIED", outcome.changed)
        for category in outcome.residual_categories:
            note_finding(
                location, "anonymizer_residual_pass", category, outcome.changed
            )
    for ordinal, ((kind, _text), outcome) in enumerate(zip(detect_texts, detected)):
        location = {"surface": kind, "surface_ordinal": ordinal}
        for reason in outcome.review_reasons:
            note_finding(location, reason, "UNCLASSIFIED", outcome.changed)
        for category in outcome.residual_categories:
            note_finding(
                location, "anonymizer_residual_pass", category, outcome.changed
            )
    result.summary["review_finding_count"] = len(review_findings)
    result.summary["review_findings"] = review_findings[:1000]
    result.summary["review_findings_truncated"] = len(review_findings) > 1000
    # Only what actually changed is a needle. A header like "Contact" or an
    # analyte like "sodium" goes through the sanitizer and comes back
    # untouched, and it is supposed to still be in the package - searching for
    # it would report the clinical content as a leak.
    replaced_originals = list(policy_state.replaced_originals) + [
        original
        for original, sanitized in zip(rewrite_texts, rewritten)
        if original != getattr(sanitized, "text", sanitized)
        and original.strip() not in kept_clinical
    ]
    emitted = allocator.emitted()
    written_tokens = set(emitted) | policy_state.emitted_tokens
    for original, sanitized in zip(rewrite_texts, rewritten):
        text = getattr(sanitized, "text", str(sanitized))
        for token in re.findall(r"<[A-Z][A-Z0-9_]*>", text):
            if token not in original:
                written_tokens.add(token)
    result.add_categories(policy_state.category_counts)
    result.add_categories(rewrite.categories)
    result.add_categories(detect.categories)
    if rewrite.review_reasons:
        result.hold(REASON_TEXT_REVIEW, *sorted(rewrite.review_reasons))
    if detect.review_reasons:
        result.hold(REASON_TEXT_REVIEW, *sorted(detect.review_reasons))
    if rewrite.residual:
        result.hold(
            REASON_ANONYMIZER_RESIDUAL,
            *sorted(f"residual_{category.lower()}" for category in rewrite.residual),
        )
    hold_for = {
        "formula": REASON_PHI_IN_FORMULA,
        "sheet_name": REASON_PHI_IN_SHEET_NAME,
        "defined_name": REASON_PHI_IN_DEFINED_NAME,
    }
    for (kind, _text), outcome in zip(detect_texts, detected):
        if outcome.changed and kind in hold_for:
            result.hold(hold_for[kind])
        elif outcome.changed and kind.startswith("removed_defined_name"):
            result.warn(REASON_REMOVED_DEFINED_NAME)

    with clock.stage("sanitize"):
        for slot, outcome in zip(rewrite_slots, rewritten):
            if outcome.changed:
                slot.write(outcome.text)
        for group in formula_groups:
            rewritten_formula = group.apply()
            if group.key is not None:
                formulas[group.key] = rewritten_formula
        # A sheet named with an identifier holds the file, and its review
        # copy never carries the name: the sheet gets a neutral title and
        # every formula and defined name that pointed at it follows.
        renames = _neutral_sheet_titles(
            workbook,
            [
                text
                for (kind, text), outcome in zip(detect_texts, detected)
                if kind == "sheet_name" and outcome.changed
            ],
        )
        if renames:
            _apply_sheet_renames(workbook, renames, formulas, numbers)
            counts["sheet_names_replaced"] = len(renames)
        for sheet in sheets:
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.hyperlink is not None:
                        cell.hyperlink = None
                        counts["hyperlinks_removed"] += 1
                    if cell.comment is not None:
                        cell.comment.author = NEUTRAL_AUTHOR
        for field in _CORE_PROPERTY_FIELDS:
            if getattr(workbook.properties, field, None):
                setattr(workbook.properties, field, None)
        # The writer stamps "created" and "modified" with the moment of the
        # save, which is both a fact about the person sanitizing the file and
        # the reason two runs over identical input produced different bytes.
        # Pinning them is the removal, and it makes the artifact
        # reproducible: a caller can hash the download and hash it again
        # tomorrow.
        workbook.properties.created = FIXED_TIMESTAMP
        workbook.properties.modified = FIXED_TIMESTAMP
        workbook.custom_doc_props.props.clear()
        workbook._external_links = []
        for container, name in removed_defined_names:
            if name in container:
                del container[name]
        if removed_defined_names:
            counts["defined_names_removed"] = len(removed_defined_names)
        for sheet in sheets:
            # Pictures and charts are not read by this profile, so none is
            # carried into any copy.
            if getattr(sheet, "_images", None):
                counts["images_removed"] = counts.get("images_removed", 0) + len(
                    sheet._images
                )
                sheet._images = []
            if getattr(sheet, "_charts", None):
                counts["charts_removed"] = counts.get("charts_removed", 0) + len(
                    sheet._charts
                )
                sheet._charts = []
    result.summary.update(counts)

    with clock.stage("write"):
        try:
            buffer = BytesIO()
            workbook.save(buffer)
            output = _repacked(buffer.getvalue())
        except Exception:
            result.error(REASON_WRITE_FAILED)
            return
    expected_strings = {
        (slot.owner.parent.title, slot.owner.coordinate): outcome.text
        for slot, outcome in zip(rewrite_slots, rewritten)
        if slot.is_cell
    }
    protected_expected: Dict[Tuple[str, str], Any] = {}
    clinical_expected: Dict[Tuple[str, str], Any] = {}
    content_expected: Dict[Tuple[str, str], Any] = {}
    for sheet in sheets:
        for row in sheet.iter_rows():
            for cell in row:
                if id(cell) in policy_state.protected_cells:
                    key = (sheet.title, cell.coordinate)
                    protected_expected[key] = cell.value
                    if id(cell) in policy_state.clinical_values:
                        clinical_expected[key] = policy_state.clinical_values[id(cell)]
                    if id(cell) in policy_state.content_values:
                        content_expected[key] = policy_state.content_values[id(cell)]

    with clock.stage("verify"):
        texts = _verify_output(
            output,
            result,
            workbook.sheetnames,
            formulas,
            numbers,
            expected_strings,
            openpyxl,
            originals=replaced_originals,
            protected_expected=protected_expected,
            clinical_expected=clinical_expected,
            content_expected=content_expected,
            policy_state=policy_state,
            tabular_config=tabular_config,
        )
        if texts is None:
            return
        outcome = verify_extracted_text(texts, frozenset(written_tokens), isolate=True)
    if not outcome.passed:
        result.hold(
            REASON_RESIDUAL,
            *sorted(
                f"residual_{category.lower()}"
                for category in outcome.blocking_categories
            ),
        )
    result.residual_verification_passed = outcome.passed and not result.errors
    result.summary["output_size_bytes"] = len(output)
    result.output_bytes = output
    result.output_digest = hashlib.sha256(output).hexdigest()
    result.output_size = len(output)
    result.generated_tokens = frozenset(written_tokens)
    result.output_media_type = XLSX_MEDIA_TYPE
    result.output_extension = ".xlsx"
    result.validators = VALIDATORS
    category_slugs: Dict[str, int] = {}
    for entity_type, count in result.categories_detected.items():
        category = tabular_privacy.category_for_entity(entity_type)
        if category:
            category_slugs[category] = category_slugs.get(category, 0) + int(count)
    all_columns = tuple(
        column for columns in policy_state.columns.values() for column in columns
    )
    state = "held" if result.reason_codes else "ready_candidate"
    result.summary["tabular_privacy"] = tabular_privacy.policy_summary(
        tabular_config,
        all_columns,
        category_counts=category_slugs,
        transformations=policy_state.transformations,
        dataset_summary=result.summary,
        unresolved_reasons=(
            code
            for code in result.reason_codes
            if code.startswith("tabular_") or "quasi_identifier" in code
        ),
        candidate_state=state,
    )


def _date_value(value: Any, settings: Dict[str, Any]) -> Any:
    strategy = settings["date_strategy"]
    if strategy == "shift":
        return value - timedelta(days=17)
    if strategy == "generalize":
        return datetime(value.year, 1, 1)
    return None


def _column_keeps(sheet: Any, cell: Any, value: str) -> bool:
    """Whether a column's header names it as analytes or units and the cell is one."""
    from services.clinical_preservation import is_unit_expression
    from services.csv_sanitizer import header_tokens
    from services.csv_terminology import (
        is_analyte_cell,
        is_analyte_header,
        is_measurement_unit,
        is_unit_header,
    )

    row = getattr(cell, "row", None)
    column = getattr(cell, "column", None)
    if not row or row <= 1 or not column:
        return False
    header = sheet.cell(1, column).value
    if not isinstance(header, str) or not header.strip():
        return False
    tokens = header_tokens(header)
    stripped = value.strip()
    if is_analyte_header(tokens):
        return is_analyte_cell(stripped)
    if is_unit_header(tokens):
        return is_measurement_unit(stripped) or is_unit_expression(stripped)
    return False


_IDENTIFIER_BREAK = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[_.]+")


def identifier_words(name: str) -> str:
    """A code identifier read as the words it is built from.

    A defined name is a formula symbol - "ReferenceRange", "lab_units" - and
    a model reads the unbroken symbol as one proper noun. Split at case
    changes, underscores and dots, it reads as what it says; a person's name
    written as a symbol ("JaneDoe") becomes "Jane Doe" and keeps its name
    shape, and any change to it still holds the workbook.
    """
    return " ".join(part for part in _IDENTIFIER_BREAK.split(name or "") if part)


def _builtin_defined_name(name: str) -> bool:
    """Names whose semantics are defined by OOXML rather than user prose."""
    return (name or "").casefold().startswith("_xlnm.")


def _defined_name_referenced(workbook: Any, name: str, current: Any) -> bool:
    """Whether removing a user-defined name could change workbook behavior."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.])" + re.escape(name) + r"(?![A-Za-z0-9_.])",
        re.IGNORECASE,
    )
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if (
                    isinstance(cell.value, str)
                    and cell.value.startswith("=")
                    and pattern.search(cell.value)
                ):
                    return True
        for validation in sheet.data_validations.dataValidation:
            if any(
                pattern.search(str(value))
                for value in (validation.formula1, validation.formula2)
                if value
            ):
                return True
    containers = [workbook.defined_names] + [
        sheet.defined_names for sheet in workbook.worksheets
    ]
    return any(
        defined is not current
        and pattern.search(str(getattr(defined, "attr_text", "") or ""))
        for container in containers
        for _other_name, defined in list(container.items())
    )


def _defined_name_texts(
    defined: Any, detect_texts: List[Tuple[str, str]], result, kind: str = "formula"
) -> None:
    formula = str(getattr(defined, "attr_text", "") or "")
    if _EXTERNAL_REFERENCE.search(formula):
        if kind.startswith("removed_defined_name"):
            result.warn(REASON_REMOVED_DEFINED_NAME)
        else:
            result.hold(REASON_EXTERNAL_FORMULA)
    detect_texts.extend((kind, literal) for literal in _literals(formula))


#: A value shorter than this matches too much of an XML package to mean
#: anything - "96" appears in a style index and a column width.
MIN_SEARCHABLE_VALUE = 5


def _survivors_in_package(output: bytes, originals) -> List[str]:
    """Original values still present anywhere in the written package.

    Searches the decompressed text of every part - worksheets, shared
    strings, comments, properties and the relationship files - rather than
    the reopened grid, because the grid is only one of the places a value
    can be.
    """
    wanted = {
        text.strip()
        for text in originals or ()
        if isinstance(text, str) and len(text.strip()) >= MIN_SEARCHABLE_VALUE
    }
    if not wanted:
        return []
    blob = ooxml_package.readable_text(output)
    folded = blob.casefold()
    return sorted(value for value in wanted if value.casefold() in folded)


def _EXPECTED_OUTPUT_PARTS(names):
    """The subset of an output package's names the allowlist recognises.

    Written as a set difference at the call site so a part that is in the
    archive and not in the allowlist is named in the summary, rather than
    only counted. Nothing unverified is copied into an output; this is what
    proves it for a given file.
    """
    return {
        name
        for name in names
        if any(pattern.match(name) for pattern in ooxml_package._ALLOWED_PARTS)
    }


def _neutral_sheet_titles(workbook, flagged: Sequence[str]) -> Dict[str, str]:
    """A unique neutral title ("Sheet4") for each flagged sheet name."""
    taken = set(workbook.sheetnames)
    renames: Dict[str, str] = {}
    number = 1
    for title in flagged:
        if title in renames or title not in taken:
            continue
        while f"Sheet{number}" in taken:
            number += 1
        renames[title] = f"Sheet{number}"
        taken.add(renames[title])
    return renames


def _renamed_references(formula: str, renames: Dict[str, str]) -> str:
    """A formula with every reference to a renamed sheet pointing at its new title."""
    for old, new in renames.items():
        quoted = "'" + old.replace("'", "''") + "'!"
        formula = formula.replace(quoted, new + "!")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", old):
            formula = re.sub(r"(?<![\w.'])" + re.escape(old) + r"!", new + "!", formula)
    return formula


def _apply_sheet_renames(workbook, renames: Dict[str, str], formulas, numbers) -> None:
    """Rename sheets, follow their references, and re-key the integrity maps."""
    for sheet in workbook.worksheets:
        if sheet.title in renames:
            sheet.title = renames[sheet.title]
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and cell.value.startswith("="):
                    cell.value = _renamed_references(cell.value, renames)
    containers = [workbook.defined_names] + [
        sheet.defined_names for sheet in workbook.worksheets
    ]
    for container in containers:
        for _name, defined in list(container.items()):
            if getattr(defined, "attr_text", None):
                defined.attr_text = _renamed_references(defined.attr_text, renames)
    for mapping, is_formula in ((formulas, True), (numbers, False)):
        items = list(mapping.items())
        mapping.clear()
        for (title, coordinate), value in items:
            if is_formula:
                value = _renamed_references(value, renames)
            mapping[(renames.get(title, title), coordinate)] = value


def _verify_output(
    output,
    result,
    sheetnames,
    formulas,
    numbers,
    expected_strings,
    openpyxl,
    originals=(),
    protected_expected=None,
    clinical_expected=None,
    content_expected=None,
    policy_state=None,
    tabular_config=None,
):
    try:
        with zipfile.ZipFile(BytesIO(output)) as archive:
            names = archive.namelist()
        reopened = openpyxl.load_workbook(
            BytesIO(output), data_only=False, keep_links=True
        )
    except Exception:
        result.error(REASON_REOPEN_FAILED)
        return None

    try:
        # The same allowlist, now as proof rather than as a gate: every part
        # of the artifact has to be one a sanitized spreadsheet may contain,
        # and a hyperlink that was permitted on the way in is not permitted
        # here, because permitting it would mean the removal never happened.
        produced = ooxml_package.inspect(output, output=True)
        result.summary["output_package"] = produced.safe_summary()
        if not produced.acceptable:
            result.error(REASON_HIDDEN_CHANNEL)
        unverified = sorted(set(name for name in names) - _EXPECTED_OUTPUT_PARTS(names))
        if unverified:
            result.summary["unverified_output_parts"] = unverified
            result.error(REASON_HIDDEN_CHANNEL)
        if reopened.sheetnames != list(sheetnames):
            result.error(REASON_STRUCTURE_CHANGED)
            return None
        if any(
            str(getattr(reopened.properties, field, None) or "")
            not in _WRITER_DEFAULTS | {""}
            for field in _CORE_PROPERTY_FIELDS
        ):
            result.error(REASON_HIDDEN_CHANNEL)
        if list(reopened.custom_doc_props.props):
            result.error(REASON_HIDDEN_CHANNEL)

        protected_expected = protected_expected or {}
        clinical_expected = clinical_expected or {}
        content_expected = content_expected or {}
        texts: List[str] = []
        seen_formulas: Dict[Tuple[str, str], str] = {}
        for sheet in reopened.worksheets:
            for row in sheet.iter_rows():
                for cell in row:
                    key = (sheet.title, cell.coordinate)
                    value = cell.value
                    if cell.hyperlink is not None:
                        result.error(REASON_HIDDEN_CHANNEL)
                    if cell.comment is not None:
                        if (cell.comment.author or "") != NEUTRAL_AUTHOR:
                            result.error(REASON_HIDDEN_CHANNEL)
                        texts.append(cell.comment.text or "")
                    if key in formulas:
                        seen_formulas[key] = str(value)
                        texts.extend(_literals(str(value)))
                        continue
                    if key in numbers and value != numbers[key]:
                        result.error(REASON_VALUE_CHANGED)
                    if key in expected_strings:
                        if value != expected_strings[key]:
                            result.error(REASON_VALUE_CHANGED)
                    if key in protected_expected and value != protected_expected[key]:
                        result.error(REASON_VALUE_CHANGED)
                    if key in clinical_expected and value != clinical_expected[key]:
                        result.hold(REASON_CLINICAL_CHANGED)
                    if key in content_expected and value != content_expected[key]:
                        result.hold(tabular_privacy.REASON_CONTENT_CHANGED)
                    if isinstance(value, str) and key not in protected_expected:
                        # Only unprotected free text needs a model pass. The
                        # shared policy already proved protected clinical and
                        # schema cells; joining rows creates cross-cell spans.
                        texts.append(value)
            for attribute in _HEADER_FOOTER:
                block = getattr(sheet.HeaderFooter, attribute, None)
                for part in ("left", "center", "right"):
                    item = getattr(block, part, None) if block is not None else None
                    if item is not None and item.text:
                        texts.append(item.text)
            for validation in sheet.data_validations.dataValidation:
                texts.extend(
                    str(getattr(validation, field))
                    for field in _VALIDATION_FIELDS
                    if getattr(validation, field, None)
                )
        if seen_formulas != formulas:
            result.error(REASON_STRUCTURE_CHANGED)

        if policy_state is not None and tabular_config is not None:
            _verify_dataset_plans(reopened, policy_state, tabular_config, result)

        # The last question, and the one a reader of the grid cannot answer:
        # is the value anywhere in the package at all. A string can sit in
        # sharedStrings, in a worksheet as an inline string, in a comment, in
        # a defined name, in a document property or in a relationship target,
        # and a check that reads only the cells passes by not looking.
        survivors = _survivors_in_package(output, originals)
        if survivors:
            result.summary["values_surviving_in_package"] = len(survivors)
            result.error(REASON_VALUE_SURVIVED)
        return texts
    finally:
        reopened.close()


def _verify_dataset_plans(
    reopened: Any,
    state: _WorkbookPolicyState,
    config: tabular_privacy.PolicyConfig,
    result: MultiformatResult,
) -> None:
    """Recompute QI classes from the workbook bytes that will be downloaded."""
    checks = []
    policy = tabular_privacy.dataset_policy(config)
    for plan in state.plans:
        title = plan.sheet.title
        if title not in reopened.sheetnames:
            result.error(REASON_STRUCTURE_CHANGED)
            return
        sheet = reopened[title]
        width = len(plan.columns)
        rows = [
            [
                (
                    ""
                    if sheet.cell(row_number, column + 1).value is None
                    else str(sheet.cell(row_number, column + 1).value)
                )
                for column in range(width)
            ]
            for row_number in plan.data_rows
        ]
        check = dataset_privacy.verify_dataset(
            rows,
            plan.roles,
            is_null=lambda value: not str(value).strip(),
            policy=policy,
        )
        checks.append(check)
        if not check.satisfied:
            result.error(REASON_QI_NOT_VERIFIED)
    if not checks:
        result.summary["workbook_dataset_privacy_verified_on_output"] = None
        return
    result.summary.update(
        {
            "workbook_dataset_privacy_verified_on_output": all(
                check.satisfied for check in checks
            ),
            "equivalence_classes": sum(check.equivalence_classes for check in checks),
            "k_achieved": min(check.k for check in checks),
            "l_achieved": min(
                (check.l for check in checks if check.l is not None), default=None
            ),
            "t_achieved": max(
                (check.t for check in checks if check.t is not None), default=None
            ),
        }
    )
