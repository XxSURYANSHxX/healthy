"""CSV anonymization that keeps every row, column and untouched cell.

A CSV is a table, not a document. Anonymizing it as one string loses which
value belongs to which column, lets a detector join a name in one cell to the
next cell's first word, and makes the rows impossible to rebuild. This handler
works cell by cell:

1. **Parse** with the standard library, comma-separated, UTF-8 with or without
   a byte-order mark. Other delimiters are refused rather than parsed wrongly,
   and every size and shape limit is enforced before anything is anonymized.
2. **Classify columns** from their headers. A header that names a Safe Harbor
   field - a name, an address, a date of birth, a record number - makes its
   column's values identifiers without asking a model. A header that names a
   clinical measurement protects its numbers.
3. **Plan each cell.** Empty cells, null markers and booleans are left alone.
   Values under an identifying header are replaced with the profile's own
   surrogate or placeholder. A whole-cell email, telephone number, SSN, URL,
   IP address or date is replaced by the deterministic structured rules. A
   number whose integer part could be either an identifier or a measurement
   holds the file; its decimal places never count. Closed-vocabulary values -
   clinical codes, sex codes, name titles and suffixes, race and ethnicity
   categories, US states - are kept for their column, and so is a header
   made only of the classifier's own words. Everything else that carries
   text goes to full_v1.
4. **Run full_v1 once per distinct text**, in batches joined by the proven
   boundary, sharing one surrogate allocator so a person keeps one surrogate
   in every cell. Each replacement is placed back into its own cell by input
   offset; the joined output is never split to recover cells. A replacement
   that crosses a cell boundary sends that batch back in halves, down to one
   value at a time, and a review or residual flag is kept only for the cells
   that raise it on their own.
5. **Write** a new CSV with the source's line endings, quoting style and
   byte-order mark where they round-trip exactly.
6. **Reopen and inspect** with an independently configured parser: the same
   rows and columns, every cell equal to its plan, no replaced value or part
   of an identifier left anywhere, nothing identifier-shaped under an
   identifying header, protected numbers unchanged, and every text cell and
   free-text header read again by the residual validator. A finding in a
   joined batch is attributed by rescanning halves: one a single cell raises
   holds the file with that cell's row and column, and one no single cell
   raises existed only across the separators and is dropped.

Only an output that passes every check can be released, and only with
production detectors. Values, offsets and the surrogate mapping stay in
memory; results carry counts and reason codes.
"""

from __future__ import annotations

import codecs
import csv
import hashlib
import io
import re
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from services import dataset_privacy, tabular_privacy
from services.multiformat import (
    SIGNATURE_BIGTIFF,
    SIGNATURE_DICOM,
    SIGNATURE_GZIP,
    SIGNATURE_NIFTI1,
    SIGNATURE_NIFTI2,
    SIGNATURE_PDF,
    SIGNATURE_TIFF,
    SIGNATURE_ZIP,
    TEXT_BATCH_BYTES,
    TEXT_SEPARATOR,
    MultiformatError,
    MultiformatResult,
    merge_counts,
)
from services.request_deadline import check as _deadline_check
from services.surrogates import SURROGATE_PATTERN, SurrogateAllocator

# -- limits ---------------------------------------------------------------------------------

MAX_CSV_BYTES = 32 * 1024 * 1024
#: Decoded characters. UTF-8 never decodes to more characters than bytes, so
#: this is also a guard against a decoder change.
MAX_CSV_CHARS = 32 * 1024 * 1024
MAX_CSV_ROWS = 200_000
MAX_CSV_COLUMNS = 512
MAX_CSV_CELL_CHARS = 32_768
MAX_CSV_CELLS = 2_000_000
#: Distinct text values, and their total characters, that may go to full_v1
#: for one file. Beyond either, the file is held instead of risking the
#: request deadline part way through.
MAX_MODEL_FIELDS = 20_000
MAX_MODEL_CHARS = 128 * 1024
#: One full_v1 call holds at most this many values.
MAX_BATCH_FIELDS = 512
#: Values searched per compiled expression when an output is checked for survivors.
SURVIVOR_PATTERN_GROUP = 256
#: Characters read to decide whether the file uses another delimiter.
SNIFF_CHARS = 64 * 1024

CSV_MEDIA_TYPE = "text/csv; charset=utf-8"

# -- reason codes ---------------------------------------------------------------------------

REASON_MODEL_FIELD_LIMIT = "csv_model_field_limit_exceeded"
REASON_MODEL_TEXT_LIMIT = "csv_model_text_limit_exceeded"
REASON_AMBIGUOUS_NUMBER = "csv_identifier_or_measurement_ambiguous"
REASON_TEXT_REVIEW = "csv_text_requires_review"
REASON_ANONYMIZER_RESIDUAL = "privacy_requirements_not_met"
REASON_CROSSING = "csv_finding_crosses_cells"
REASON_MAPPING = "csv_replacement_mapping_inconsistent"
REASON_WRITE_FAILED = "csv_output_write_failed"
REASON_REOPEN_FAILED = "csv_output_reopen_failed"
REASON_STRUCTURE_CHANGED = "csv_structure_changed"
REASON_VALUE_CHANGED = "csv_cell_differs_from_plan"
REASON_DIALECT_CHANGED = "csv_output_dialect_changed"
REASON_PHI_SURVIVED = "csv_replaced_value_survived"
REASON_PARTIAL_SURVIVED = "csv_partial_identifier_survived"
REASON_HEADER_IDENTIFIER = "csv_identifier_survived_under_phi_header"
REASON_PROTECTED_CHANGED = "csv_protected_value_changed"
REASON_MARKER_SURVIVED = "csv_batch_boundary_in_output"
REASON_RESIDUAL = "csv_residual_phi_detected"
REASON_INPUT_CHANGED = "csv_input_modified"

VALIDATORS = (
    "csv_structure_round_trip",
    "csv_header_aware_classification",
    "full_v1_text_anonymization",
    "csv_reopened_output_inspection",
    "residual_phi_rescan",
)

_BINARY_SIGNATURES = frozenset(
    {
        SIGNATURE_PDF,
        SIGNATURE_DICOM,
        SIGNATURE_ZIP,
        SIGNATURE_GZIP,
        SIGNATURE_TIFF,
        SIGNATURE_BIGTIFF,
        SIGNATURE_NIFTI1,
        SIGNATURE_NIFTI2,
    }
)
_OTHER_BOMS = (
    codecs.BOM_UTF32_LE,
    codecs.BOM_UTF32_BE,
    codecs.BOM_UTF16_LE,
    codecs.BOM_UTF16_BE,
)
_OTHER_DELIMITERS = (";", "\t", "|")

# -- value shapes ---------------------------------------------------------------------------

NULL_MARKERS: FrozenSet[str] = frozenset(
    {
        "na",
        "n/a",
        "nan",
        "null",
        "none",
        "nil",
        "-",
        "--",
        "?",
        "unknown",
        "not recorded",
        "not available",
        "missing",
    }
)
BOOLEAN_MARKERS: FrozenSet[str] = frozenset(
    {
        "true",
        "false",
        "yes",
        "no",
        "y",
        "n",
        "t",
        "f",
    }
)

_NUMBER = r"(?:[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+)"
_MEASUREMENT = re.compile(
    r"^(?:[<>~≤≥]=?\s*)?"
    + _NUMBER
    + r"(?:\s*(?:-|–|to|/|:)\s*"
    + _NUMBER
    + r")?"
    + r"(?:\s*(?P<unit>%|°\s*[cf]|[a-zµ%°][a-z0-9µ/^·*.%]{0,15}))?$",
    re.IGNORECASE,
)
_PHONE_SHAPED = re.compile(r"^(?:\(?\d{3}\)?[-.\s])?\d{3}[-.]\d{4}$")
#: A whole ICD-10 code. In a diagnosis column it is the diagnosis, not an identifier.
_CLINICAL_CODE = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$")
#: The closed set of values a sex or gender column carries.
SEX_CODES = frozenset({"m", "f", "u", "x", "male", "female", "other", "intersex"})
_SEX_HEADER_WORDS = frozenset({"sex", "gender"})
_MONTHS = (
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|january|february|march|"
    r"april|june|july|august|september|october|november|december)"
)
_DATE_VALUE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[ t]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:z|[+-]\d{2}:?\d{2})?)?"
    r"|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?)?"
    r"|\d{1,2}[\s-]" + _MONTHS + r"\.?[\s,-]*\d{2,4}"
    r"|" + _MONTHS + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|\d{4}[-/]\d{1,2}"
    r"|" + _MONTHS + r"\.?\s+\d{4})$",
    re.IGNORECASE,
)
_YEAR_VALUE = re.compile(r"^\d{4}$")
_PLACEHOLDER_TOKEN = re.compile(r"<REDACTED_[A-Z0-9_]+>")
_GENERATED_ONLY = re.compile(
    r"^(?:<REDACTED_[A-Z0-9_]+>|"
    + SURROGATE_PATTERN.pattern.replace(r"\b", "")
    + r"|90\+)$"
)
_DIGIT_RUN = re.compile(r"\d+")
#: A header label: a word, or up to four joined by spaces, underscores, dots or hyphens.
_LABEL = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[_ .-][A-Za-z0-9]+){0,3}")

#: Whole-cell structured matches replaced without a model call.
_DETERMINISTIC_WHOLE_CELL = frozenset(
    {
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "URL",
        "IP_ADDRESS",
        "DATE_TIME",
    }
)
#: A long integer outside a clinical column could be a record number.
AMBIGUOUS_INTEGER_DIGITS = 7
#: Inside a clinical column, only a much longer integer is ambiguous.
AMBIGUOUS_CLINICAL_INTEGER_DIGITS = 9

# -- header vocabulary --------------------------------------------------------------------

POLICY_PHI = "phi"
POLICY_DATE = "date"
POLICY_BIRTH_YEAR = "birth_year"
POLICY_AGE = "age"
POLICY_CLINICAL = "clinical"
POLICY_TEXT = "text"
#: The parts of a name that are not the name - a title before it, a
#: generational or credential suffix after it.
POLICY_NAME_AFFIX = "name_affix"


def _patterns(*phrases: str) -> Tuple[FrozenSet[str], ...]:
    return tuple(frozenset(phrase.split()) for phrase in phrases)


#: (entity type, policy, header token patterns). A pattern matches when all
#: of its tokens appear in the header; the longest matching pattern wins and
#: ties go to the earlier rule.
HEADER_RULES: Tuple[Tuple[str, str, Tuple[FrozenSet[str], ...]], ...] = (
    # Dates and ages first, so "chart review date" or "account opened date"
    # is a date column on a tie.
    ("DATE", POLICY_BIRTH_YEAR, _patterns("birth year", "yob", "year of birth")),
    (
        "DATE",
        POLICY_DATE,
        _patterns(
            "dob",
            "birth",
            "birthdate",
            "date",
            "datetime",
            "timestamp",
            "dod",
            "death",
            "deceased",
            "died",
            "deathdate",
            "dateofbirth",
            "dateofdeath",
        ),
    ),
    ("AGE_OVER_89", POLICY_AGE, _patterns("age", "ageyears")),
    (
        "VEHICLE_ID",
        POLICY_PHI,
        _patterns(
            "vin",
            "vehicle",
            "plate",
            "license plate",
            "licence plate",
            "registration plate",
            "reg plate",
            "vehicle registration",
            "numberplate",
        ),
    ),
    (
        "DEVICE_ID",
        POLICY_PHI,
        _patterns(
            "device id",
            "device serial",
            "device number",
            "serial",
            "serial number",
            "udi",
            "implant id",
            "implant serial",
            "imei",
            "mac address",
            "serialnumber",
        ),
    ),
    (
        "IP_ADDRESS",
        POLICY_PHI,
        _patterns("ip", "ip address", "ipv4", "ipv6", "ipaddress"),
    ),
    ("EMAIL_ADDRESS", POLICY_PHI, _patterns("email", "e mail", "emailaddress")),
    (
        "URL",
        POLICY_PHI,
        _patterns("url", "uri", "website", "web address", "homepage", "link"),
    ),
    (
        "IMAGE_REFERENCE",
        POLICY_PHI,
        _patterns(
            "photo",
            "photograph",
            "picture",
            "image path",
            "image url",
            "image file",
            "headshot",
            "face image",
            "photo url",
        ),
    ),
    (
        "BIOMETRIC_ID",
        POLICY_PHI,
        _patterns(
            "biometric",
            "fingerprint",
            "retina",
            "iris scan",
            "voiceprint",
            "voice print",
            "face template",
            "palm print",
        ),
    ),
    ("US_SSN", POLICY_PHI, _patterns("ssn", "social security", "socialsecurity")),
    (
        "NATIONAL_ID",
        POLICY_PHI,
        _patterns(
            "national id",
            "national identifier",
            "national insurance",
            "nhs number",
            "nhs",
            "tax id",
            "tin",
            "sin",
        ),
    ),
    (
        "LICENSE_NUMBER",
        POLICY_PHI,
        _patterns(
            "license",
            "licence",
            "drivers",
            "driver license",
            "driving licence",
            "passport",
            "dea",
            "dl number",
        ),
    ),
    ("CERTIFICATE_NUMBER", POLICY_PHI, _patterns("certificate", "cert number")),
    (
        "PHONE_NUMBER",
        POLICY_PHI,
        _patterns(
            "phone",
            "telephone",
            "tel",
            "mobile",
            "cellphone",
            "cell phone",
            "cell number",
            "fax",
            "pager",
            "landline",
            "contact number",
            "contact phone",
            "phonenumber",
        ),
    ),
    (
        "HEALTH_PLAN_ID",
        POLICY_PHI,
        _patterns(
            "health plan",
            "insurance",
            "insurance id",
            "policy number",
            "policy id",
            "beneficiary",
            "medicaid",
            "medicare",
            "member id",
            "member number",
            "subscriber",
            "payer id",
            "group number",
        ),
    ),
    (
        "ACCOUNT_NUMBER",
        POLICY_PHI,
        _patterns(
            "account", "acct", "billing number", "invoice number", "iban", "card number"
        ),
    ),
    (
        "MEDICAL_RECORD_NUMBER",
        POLICY_PHI,
        _patterns(
            "mrn",
            "medical record",
            "record number",
            "record no",
            "chart number",
            "chart no",
            "hospital number",
            "hospital no",
            "medicalrecordnumber",
        ),
    ),
    (
        "PATIENT_ID",
        POLICY_PHI,
        _patterns(
            "patient id",
            "patient identifier",
            "patient number",
            "patient no",
            "pid",
            "pt id",
            "patientid",
            "person id",
        ),
    ),
    (
        "ACCESSION_NUMBER",
        POLICY_PHI,
        _patterns(
            "accession",
            "specimen id",
            "specimen number",
            "sample id",
            "sample number",
            "barcode",
            "requisition",
            "lab number",
            "order number",
        ),
    ),
    (
        "IDENTIFIER",
        POLICY_PHI,
        _patterns(
            "id",
            "identifier",
            "uuid",
            "guid",
            "encounter id",
            "visit id",
            "episode id",
            "admission id",
            "case id",
            "case number",
            "subject id",
            "participant id",
            "study id",
            "enrolment id",
            "enrollment id",
            "tracking number",
            "reference number",
            "npi",
            "employee id",
            "record id",
            "encounter",
        ),
    ),
    (
        "ADDRESS",
        POLICY_PHI,
        _patterns(
            "address", "addr", "street", "address line", "residence", "home address"
        ),
    ),
    (
        "LOCATION",
        POLICY_PHI,
        _patterns(
            "city",
            "town",
            "county",
            "zip",
            "zipcode",
            "postcode",
            "postal",
            "postal code",
            "precinct",
            "geocode",
            "latitude",
            "lat",
            "longitude",
            "lon",
            "lng",
            "suburb",
            "village",
            "birthplace",
            "birth place",
            "neighborhood",
            "neighbourhood",
        ),
    ),
    (
        "PERSON",
        POLICY_PHI,
        _patterns(
            "name",
            "surname",
            "forename",
            "firstname",
            "lastname",
            "fullname",
            "maiden",
            "nickname",
            "guardian",
            "kin",
            "nok",
            "spouse",
            "employer",
            "relative",
            "household member",
        ),
    ),
)

#: Headers that are exactly one of these words name a person: "FIRST", "LAST".
_SINGLE_WORD_PERSON = frozenset({"first", "last", "given", "family", "middle"})
#: Headers that name a name affix column: "PREFIX", "SUFFIX", "name suffix".
_NAME_AFFIX_HEADERS = tuple(
    frozenset(words.split())
    for words in (
        "prefix",
        "suffix",
        "honorific",
        "salutation",
        "name prefix",
        "name suffix",
    )
)
#: The closed set of values an affix column carries verbatim, compared without
#: case or trailing dots. A title or a generational or credential suffix names
#: nobody; anything else in such a column is treated as part of a name.
NAME_AFFIXES = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "miss",
        "mx",
        "dr",
        "prof",
        "sir",
        "dame",
        "madam",
        "rev",
        "fr",
        "jr",
        "sr",
        "ii",
        "iii",
        "iv",
        "v",
        "md",
        "phd",
        "jd",
        "do",
        "dds",
        "dmd",
        "dvm",
        "rn",
        "np",
        "pa",
        "esq",
        "mba",
        "msc",
        "bsc",
        "mbbs",
        "facs",
    }
)
#: Column words the classifier knows without giving them a rule of their own.
_KNOWN_HEADER_WORDS = frozenset(
    {
        "marital",
        "healthcare",
        "prefix",
        "suffix",
        "honorific",
        "salutation",
        "start",
        "stop",
        "patient",
        "encounter",
        "code",
        "description",
        "base",
        "area",
    }
)
#: A whole-cell UUID: a record identifier in any column, replaced by rule.
#: Language models read its hex runs as fragments of names and numbers and
#: return spans that cross into the next cell of a batch.
_UUID = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
_PROVIDER_WORDS = frozenset(
    {
        "provider",
        "physician",
        "doctor",
        "clinician",
        "surgeon",
        "nurse",
        "attending",
        "referring",
        "consultant",
        "practitioner",
        "gp",
    }
)
#: A header that names one of these is about a clinical thing, not a person or
#: an identifier: "drug name", "test id", "icd code".
_CLINICAL_SUBJECTS = frozenset(
    {
        "drug",
        "medication",
        "med",
        "medicine",
        "test",
        "lab",
        "procedure",
        "diagnosis",
        "product",
        "organism",
        "analyte",
        "panel",
        "loinc",
        "icd",
        "icd10",
        "snomed",
        "cpt",
        "ndc",
        "rxnorm",
        "atc",
        "drg",
        "hcpcs",
        "unit",
        "units",
        "item",
        "question",
        "form",
        "ward",
        "department",
        "facility",
        "hospital",
        "clinic",
        "site",
        "brand",
        "generic",
        "file",
        "column",
        "type",
        "status",
        "category",
        "class",
        "specimen type",
        "result",
        "gene",
        "variant",
        "allele",
        "protocol",
        "study arm",
        "cohort",
    }
)
_CLINICAL_SUBJECT_EXEMPT = {
    # A specimen or sample id is an identifier even though "lab" or "test" may
    # also appear: "lab specimen id". A date or an age is a Safe Harbor element
    # whatever it is the date or age of: "diagnosis date", "age at diagnosis".
    "ACCESSION_NUMBER",
    "DATE",
    "AGE_OVER_89",
}
#: A header made only of these words, and at least one provider word, is a
#: column of clinician names: "referring physician", "attending gp".
_PROVIDER_HEADER_WORDS = frozenset(
    {
        "name",
        "by",
        "signed",
        "primary",
        "care",
        "of",
        "record",
        "assigned",
        "treating",
    }
)
#: Tokens that make a column a clinical measurement.
CLINICAL_HEADER_TOKENS: FrozenSet[str] = frozenset(
    {
        "bp",
        "systolic",
        "diastolic",
        "blood",
        "pressure",
        "hr",
        "heart",
        "pulse",
        "rate",
        "temp",
        "temperature",
        "weight",
        "height",
        "bmi",
        "spo2",
        "sao2",
        "oxygen",
        "saturation",
        "glucose",
        "sodium",
        "na",
        "potassium",
        "chloride",
        "bicarbonate",
        "urea",
        "bun",
        "creatinine",
        "egfr",
        "hba1c",
        "a1c",
        "hemoglobin",
        "haemoglobin",
        "hb",
        "hgb",
        "hematocrit",
        "hct",
        "wbc",
        "rbc",
        "platelet",
        "platelets",
        "plt",
        "cholesterol",
        "ldl",
        "hdl",
        "triglycerides",
        "alt",
        "ast",
        "alp",
        "bilirubin",
        "albumin",
        "protein",
        "crp",
        "esr",
        "inr",
        "aptt",
        "troponin",
        "bnp",
        "lactate",
        "ph",
        "pco2",
        "po2",
        "dose",
        "dosage",
        "mg",
        "mcg",
        "ml",
        "iu",
        "score",
        "gcs",
        "apgar",
        "nihss",
        "mmse",
        "ratio",
        "percent",
        "pct",
        "percentage",
        "count",
        "level",
        "value",
        "measurement",
        "volume",
        "duration",
        "los",
        "rr",
        "respiratory",
        "map",
        "qtc",
        "qrs",
        "ef",
        "ejection",
        "fraction",
        "viral",
        "load",
        "cd4",
        "copies",
        "titer",
        "titre",
        "waist",
        "circumference",
        "gestational",
        "weeks",
        "days",
        "hours",
        "minutes",
        "amount",
        "quantity",
        "qty",
        "frequency",
        "min",
        "max",
        "mean",
        "median",
        "sd",
        "heartrate",
        "bloodpressure",
        "systolicbp",
        "diastolicbp",
        "cost",
        "expenses",
        "coverage",
        "units",
        "reading",
        "size",
        "length",
        "width",
        "depth",
        "diameter",
    }
)


@dataclass(frozen=True)
class ColumnClass:
    """What a column's header says about its values."""

    policy: str
    entity_type: Optional[str] = None
    provider: bool = False


TEXT_COLUMN = ColumnClass(POLICY_TEXT)


def header_tokens(header: str) -> List[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", header or "")
    return [token for token in re.split(r"[^a-z0-9]+", spaced.lower()) if token]


#: Every word the header classifier knows. A header made only of these words
#: names a kind of column; any other word - a name, say - makes it free text.
#: Words a column header uses for who did what to a record: "verified_by",
#: "requested_by", "collected". The header is structure; its values still go
#: to the detectors under whatever class the header gives the column.
_WORKFLOW_HEADER_WORDS = frozenset(
    {
        "verified",
        "requested",
        "collected",
        "reported",
        "ordered",
        "reviewed",
        "signed",
        "performed",
        "entered",
        "recorded",
        "authorised",
        "authorized",
        "approved",
        "received",
        "resulted",
        "sampled",
        "validated",
        "released",
        "processed",
    }
)
HEADER_VOCABULARY: FrozenSet[str] = (
    _WORKFLOW_HEADER_WORDS
    | frozenset(
        token
        for _entity, _policy, patterns in HEADER_RULES
        for pattern in patterns
        for token in pattern
    )
    | CLINICAL_HEADER_TOKENS
    | _SINGLE_WORD_PERSON
    | _PROVIDER_WORDS
    | _PROVIDER_HEADER_WORDS
    | (frozenset(token for subject in _CLINICAL_SUBJECTS for token in subject.split()))
    | dataset_privacy.SENSITIVE_TOKENS
    | _SEX_HEADER_WORDS
    | frozenset(
        token for _kind, words in dataset_privacy._VOCABULARY for token in words
    )
    | _KNOWN_HEADER_WORDS
    | frozenset(
        # The shared tabular policy classifies headers by these words too, so a
        # header made of them is a field name whichever classifier read it first.
        token
        for patterns in (
            [
                item
                for _kind, group in (
                    *tabular_privacy._QI_HEADERS,
                    *tabular_privacy._DIRECT_HEADERS,
                )
                for item in group
            ],
            tabular_privacy._SENSITIVE_HEADERS,
            tabular_privacy._UTILITY_HEADERS,
        )
        for pattern in patterns
        for token in pattern
    )
)


#: A past participle: "audited", "witnessed", "countersigned", "taken".
_PARTICIPLE = re.compile(r"[a-z]{3,}(?:ed|en)")


def header_is_vocabulary(header: str) -> bool:
    """Whether every word of a header is one the column classifier knows."""
    tokens = header_tokens(header)
    if len(tokens) == 2 and tokens[1] == "by" and _PARTICIPLE.fullmatch(tokens[0]):
        # "audited_by", "countersigned_by", "witnessed_by": a role column,
        # named by what was done and the word "by", whatever the verb.
        return True
    return bool(tokens) and all(token in HEADER_VOCABULARY for token in tokens)


def classify_header(header: str) -> ColumnClass:
    """The column class for one header, by token vocabulary, not exact spelling."""
    tokens = header_tokens(header)
    token_set = frozenset(tokens)
    if not tokens:
        return TEXT_COLUMN
    if token_set in _NAME_AFFIX_HEADERS and len(tokens) == len(token_set):
        return ColumnClass(POLICY_NAME_AFFIX, "PERSON")
    if len(tokens) == 1 and tokens[0] in _SINGLE_WORD_PERSON:
        return ColumnClass(POLICY_PHI, "PERSON")
    # tabular-privacy-v1 is authoritative for identifiers, dates, ages and
    # manifest-independent quasi-identifiers.  The legacy vocabulary below is
    # retained only for its richer clinical/header compatibility; sanitizer,
    # verifier and audit all enter through this same function.
    shared = tabular_privacy.classify_columns(
        [header], config=tabular_privacy.PolicyConfig()
    )[0]
    if shared.role == tabular_privacy.ROLE_DIRECT_IDENTIFIER:
        entity = shared.entity_type or "IDENTIFIER"
        if entity == "DATE":
            return ColumnClass(POLICY_DATE, entity)
        return ColumnClass(
            POLICY_PHI,
            entity,
            provider=entity == "PERSON" and bool(token_set & _PROVIDER_WORDS),
        )
    if shared.role == tabular_privacy.ROLE_QUASI_IDENTIFIER:
        if shared.qi_kind == dataset_privacy.QI_AGE:
            return ColumnClass(POLICY_AGE, "AGE_OVER_89")
        if shared.qi_kind == dataset_privacy.QI_YEAR:
            return ColumnClass(POLICY_BIRTH_YEAR, "DATE")
        if shared.qi_kind == dataset_privacy.QI_ZIP:
            return ColumnClass(POLICY_PHI, "LOCATION")
    if shared.role == tabular_privacy.ROLE_CLINICAL_UTILITY:
        return ColumnClass(POLICY_CLINICAL)
    clinical_subject = bool(token_set & _CLINICAL_SUBJECTS)
    best: Optional[Tuple[int, int, str, str]] = None
    for order, (entity_type, policy, patterns) in enumerate(HEADER_RULES):
        if clinical_subject and entity_type not in _CLINICAL_SUBJECT_EXEMPT:
            continue
        for pattern in patterns:
            if pattern <= token_set:
                candidate = (len(pattern), -order, entity_type, policy)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    if best is not None:
        _size, _order, entity_type, policy = best
        if entity_type == "PERSON":
            return ColumnClass(
                POLICY_PHI, "PERSON", provider=bool(token_set & _PROVIDER_WORDS)
            )
        return ColumnClass(policy, entity_type)
    if (
        token_set & _PROVIDER_WORDS
        and token_set <= _PROVIDER_WORDS | _PROVIDER_HEADER_WORDS
    ):
        return ColumnClass(POLICY_PHI, "PERSON", provider=True)
    if token_set & CLINICAL_HEADER_TOKENS:
        return ColumnClass(POLICY_CLINICAL)
    return TEXT_COLUMN


# -- parsing -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """How the source was written, so the output can be written the same way."""

    bom: bool
    lineterminator: str
    quoting: int
    trailing_newline: bool
    exact: bool


def _reject(detail: str, status_code: int = 400) -> MultiformatError:
    return MultiformatError(detail, status_code=status_code)


def _sample_rows(sample: str, delimiter: str) -> List[List[str]]:
    cut = max(sample.rfind("\n"), sample.rfind("\r"))
    if cut > 0:
        sample = sample[:cut]
    try:
        reader = csv.reader(io.StringIO(sample, newline=""), delimiter=delimiter)
        return [row for _index, row in zip(range(50), reader) if row]
    except csv.Error:
        return []


def _unsupported_delimiter(text: str) -> bool:
    """Whether the file is separated by something other than commas."""
    sample = text[:SNIFF_CHARS]
    comma_rows = _sample_rows(sample, ",")
    if any(len(row) > 1 for row in comma_rows):
        return False
    for delimiter in _OTHER_DELIMITERS:
        rows = _sample_rows(sample, delimiter)
        widths = {len(row) for row in rows}
        if rows and len(widths) == 1 and next(iter(widths)) > 1:
            return True
    return False


def _render(
    rows: Sequence[Sequence[str]],
    lineterminator: str,
    quoting: int,
    trailing_newline: bool,
) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(
        buffer,
        delimiter=",",
        quotechar='"',
        doublequote=True,
        quoting=quoting,
        lineterminator=lineterminator,
    )
    writer.writerows(rows)
    rendered = buffer.getvalue()
    if not trailing_newline and rendered.endswith(lineterminator):
        rendered = rendered[: -len(lineterminator)]
    return rendered


def clinical_code_columns(header: Sequence[str], column_classes: Sequence[ColumnClass]):
    """(diagnosis code columns, sex code columns) named by a header row."""
    code_columns, sex_columns = set(), set()
    for column, (cell, column_class) in enumerate(zip(header, column_classes)):
        if column_class.policy != POLICY_TEXT:
            continue
        words = frozenset(header_tokens(cell))
        if words & dataset_privacy.SENSITIVE_TOKENS:
            code_columns.add(column)
        if words & _SEX_HEADER_WORDS:
            sex_columns.add(column)
    return code_columns, sex_columns


def measurement_unit_columns(
    header: Sequence[str], column_classes: Sequence[ColumnClass]
) -> FrozenSet[int]:
    """The columns a header names as units of measure: "unit", "units", "uom"."""
    from services.csv_terminology import is_unit_header

    return frozenset(
        column
        for column, (cell, column_class) in enumerate(zip(header, column_classes))
        if column_class.policy in (POLICY_TEXT, POLICY_CLINICAL)
        and is_unit_header(header_tokens(cell))
    )


def analyte_columns(
    header: Sequence[str], column_classes: Sequence[ColumnClass]
) -> FrozenSet[int]:
    """The columns a header names as laboratory analytes or tests."""
    from services.csv_terminology import is_analyte_header

    return frozenset(
        column
        for column, (cell, column_class) in enumerate(zip(header, column_classes))
        if column_class.policy in (POLICY_TEXT, POLICY_CLINICAL)
        and is_analyte_header(header_tokens(cell))
    )


def name_affix_columns(column_classes: Sequence[ColumnClass]) -> FrozenSet[int]:
    """The columns a header names as name prefixes or suffixes."""
    return frozenset(
        column
        for column, column_class in enumerate(column_classes)
        if column_class.policy == POLICY_NAME_AFFIX
    )


_US_STATES = (
    "alabama al|alaska ak|arizona az|arkansas ar|california ca|colorado co|connecticut ct|"
    "delaware de|florida fl|georgia ga|hawaii hi|idaho id|illinois il|indiana in|iowa ia|"
    "kansas ks|kentucky ky|louisiana la|maine me|maryland md|massachusetts ma|michigan mi|"
    "minnesota mn|mississippi ms|missouri mo|montana mt|nebraska ne|nevada nv|"
    "new hampshire nh|new jersey nj|new mexico nm|new york ny|north carolina nc|"
    "north dakota nd|ohio oh|oklahoma ok|oregon or|pennsylvania pa|rhode island ri|"
    "south carolina sc|south dakota sd|tennessee tn|texas tx|utah ut|vermont vt|virginia va|"
    "washington wa|west virginia wv|wisconsin wi|wyoming wy|district of columbia dc|"
    "puerto rico pr|guam gu|american samoa as|us virgin islands vi|"
    "northern mariana islands mp"
)
#: The closed vocabularies of categorical quasi-identifiers: standard race and
#: ethnicity categories, and US states and territories by name or postal code.
#: The dataset-level policy generalizes or suppresses these to reach k, l and
#: t. A language model reading "Massachusetts" or "hispanic" alone calls it an
#: entity, and rewriting it destroys a research variable that Safe Harbor does
#: not require removing. A value outside the vocabulary is text like any other.
CATEGORY_VOCABULARIES: Dict[str, FrozenSet[str]] = {
    dataset_privacy.QI_RACE_ETHNICITY: frozenset(
        {
            "white",
            "black",
            "black or african american",
            "african american",
            "asian",
            "native",
            "american indian",
            "alaska native",
            "american indian or alaska native",
            "native american",
            "hawaiian",
            "native hawaiian",
            "pacific islander",
            "native hawaiian or other pacific islander",
            "other",
            "other race",
            "multiracial",
            "two or more races",
            "hispanic",
            "nonhispanic",
            "non-hispanic",
            "not hispanic",
            "latino",
            "hispanic or latino",
            "not hispanic or latino",
            "unknown",
            "declined",
        }
    ),
    dataset_privacy.QI_LOCATION: frozenset(
        word
        for pair in _US_STATES.split("|")
        for word in (pair.rsplit(" ", 1)[0], pair.rsplit(" ", 1)[1])
    ),
}


def _category_key(value: str) -> str:
    return " ".join(value.strip().casefold().replace("_", " ").split())


def is_category_label(kind: Optional[str], value: str) -> bool:
    """Whether a whole cell is one label of its column's closed vocabulary."""
    if kind == dataset_privacy.QI_INSTITUTION:
        from services.tabular_privacy import is_generic_institution_category

        return is_generic_institution_category(value)
    return _category_key(value) in CATEGORY_VOCABULARIES.get(kind or "", frozenset())


def is_name_affix(value: str) -> bool:
    """Whether a whole cell is one closed-vocabulary title or suffix, and nothing more."""
    return value.strip().rstrip(".").casefold() in NAME_AFFIXES


def is_coded_value(
    column: int,
    value: str,
    code_columns,
    sex_columns,
    affix_columns: FrozenSet[int] = frozenset(),
    unit_columns: FrozenSet[int] = frozenset(),
    analyte_columns: FrozenSet[int] = frozenset(),
) -> bool:
    """Whether a cell is a whole code its column may carry verbatim."""
    stripped = value.strip()
    return bool(
        (column in code_columns and _CLINICAL_CODE.fullmatch(stripped))
        or (column in sex_columns and stripped.casefold() in SEX_CODES)
        or (column in affix_columns and is_name_affix(stripped))
        or (column in unit_columns and is_measurement_unit(stripped))
        or (column in analyte_columns and is_analyte_cell(stripped))
    )


def _is_null(value: str) -> bool:
    stripped = value.strip().lower()
    return not stripped or stripped in NULL_MARKERS or stripped in BOOLEAN_MARKERS


def _measurement(value: str) -> bool:
    match = _MEASUREMENT.match(value.strip())
    if not match:
        return False
    unit = match.group("unit") or ""
    return sum(character.isdigit() for character in unit) <= 3


def _longest_digit_run(value: str) -> int:
    return max((len(run) for run in _DIGIT_RUN.findall(value)), default=0)


def _longest_integer_run(value: str) -> int:
    """The longest run of digits before any decimal point.

    A measurement's fractional digits carry precision, not identity: an
    expense of ``5921.341281738`` has four integer digits however many
    decimals the export kept. A record number cast to a float keeps its
    identifying digits before the point, where they are still counted.
    """
    return _longest_digit_run(re.sub(r"(?<=\d)\.\d+", "", value))


def _has_letter(value: str) -> bool:
    return any(character.isalpha() for character in value)


def _identifier_shaped(value: str) -> bool:
    stripped = value.strip()
    return (
        _longest_digit_run(stripped) >= 4
        or "@" in stripped
        or "://" in stripped
        or stripped.lower().startswith("www.")
    )


def _whole_cell_structured(value: str) -> Optional[str]:
    """The structured category that covers this whole cell, if any."""
    from services.phi_detection import StructuredPatternDetector

    stripped = value.strip()
    if not stripped:
        return None
    for entity in StructuredPatternDetector().detect(stripped):
        if (
            entity.entity_type in _DETERMINISTIC_WHOLE_CELL
            and entity.start == 0
            and entity.end == len(stripped)
        ):
            return entity.entity_type
    return None


# -- clinical terminology columns ---------------------------------------------------------------
# The policy lives in services/csv_terminology.py, the one place the sanitizer,
# its own verifier and the artifact audit all read it from.

from services.csv_terminology import (
    MIN_TERMINOLOGY_ROWS,
)
from services.csv_terminology import (
    POLICY_VERSION as TERMINOLOGY_POLICY_VERSION,
)  # noqa: E402,F401 - re-exported for callers
from services.csv_terminology import (
    TERMINOLOGY_ROWS_PER_CODE,
    TERMINOLOGY_WORDS,
    TerminologyPair,
    is_analyte_cell,
    is_analyte_name,
    is_measurement_unit,
    output_pairs,
    pair_provenance,
    terminology_display_accepted,
    terminology_display_guard,
    terminology_findings_explained,
    terminology_pairs,
    terminology_words_only,
)

# -- full_v1 over distinct cell texts ------------------------------------------------------


def _default_text_anonymizer(text: str, profile: str, allocator: SurrogateAllocator):
    from services.text_anonymization import anonymize_clinical_text_with_replacements

    return anonymize_clinical_text_with_replacements(
        text, profile=profile, allocator=allocator
    )


def _default_residual_scanner(text: str, provenance):
    from services.text_anonymization import residual_findings

    return residual_findings(text, provenance)


class CellTextAnonymizer:
    """Runs distinct cell texts through full_v1 in batches, cell boundaries intact.

    Batching joins unrelated cells with a separator, and a model reading the
    join can produce what no single cell contains: a name made of two cells,
    a review flag for a span across the separator. Nothing a batch produces
    is attributed to a cell unless it maps to that cell exactly:

    * Replacements are placed by input offset, never by splitting the output.
      A batch whose replacement crosses a boundary, or whose output does not
      rebuild from its per-cell replacements, is split in half and each half
      redone, down to single cells if need be. The deepest result is used.
    * A batch whose replacements all map cleanly keeps them, but its review
      and residual flags say nothing about which cell raised them. The batch
      is bisected to find the cells that raise them alone; a flag no single
      cell raises existed only across the separators and is dropped, while
      the batch's replacements - never fewer - stand.

    ``flagged`` records, per distinct text, the flags that text raised on its
    own, so every hold can name its cells.
    """

    def __init__(
        self,
        profile: str,
        allocator: SurrogateAllocator,
        anonymize: Optional[Callable] = None,
    ):
        self.profile = profile
        self.allocator = allocator
        self.anonymize = anonymize or _default_text_anonymizer
        self.calls = 0
        self.batches_redone = 0
        self.categories: Dict[str, int] = {}
        self.residual: Dict[str, int] = {}
        self.review_reasons: set = set()
        #: text -> {"review": [codes], "residual": {category: count}}
        self.flagged: Dict[str, Dict[str, Any]] = {}
        self.mapping_failed = False

    def run(
        self, texts: Sequence[str]
    ) -> Dict[str, Tuple[str, List[Tuple[int, int, str]]]]:
        """text -> (sanitized text, replaced input spans)."""
        results: Dict[str, Tuple[str, List[Tuple[int, int, str]]]] = {}
        # A tabular model invocation is a security boundary: it contains one
        # cell and cannot create or attribute a span across a record boundary.
        # Model implementations may still batch their internal chunks (the
        # bounded CUDA acceptance sets that batch size to two).
        for text in texts:
            self._single(text, results)
        return results

    def _call(self, text: str):
        _deadline_check("csv.anonymize_text")
        self.calls += 1
        return self.anonymize(text, self.profile, self.allocator)

    @staticmethod
    def _flags(result: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
        return (
            sorted(result.get("review_required_reasons") or []),
            dict(result.get("residual_phi_categories") or {}),
        )

    def _flag(self, text: str, result: Dict[str, Any]) -> None:
        """Record the flags one cell raised on its own."""
        review, residual = self._flags(result)
        if not (review or residual):
            return
        self.review_reasons.update(review)
        merge_counts(self.residual, residual)
        entry = self.flagged.setdefault(text, {"review": [], "residual": {}})
        entry["review"] = sorted(set(entry["review"]) | set(review))
        merge_counts(entry["residual"], residual)

    @staticmethod
    def _rebuild(text: str, replacements: Sequence[Tuple[int, int, str, str]]) -> str:
        pieces: List[str] = []
        cursor = 0
        for start, end, _entity_type, replacement in sorted(replacements):
            pieces.append(text[cursor:start])
            pieces.append(replacement)
            cursor = end
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _single(self, text: str, results) -> List[Tuple[int, int, str]]:
        result, replacements = self._call(text)
        rebuilt = self._rebuild(text, replacements)
        if rebuilt != result["anonymized_text"]:
            self.mapping_failed = True
        merge_counts(self.categories, result.get("detected_entities", {}))
        self._flag(text, result)
        results[text] = (rebuilt, [(s, e, t) for s, e, t, _r in replacements])
        return results[text][1]

    def _batch(self, batch: List[str], results) -> None:
        if len(batch) == 1:
            self._single(batch[0], results)
            return
        joined = TEXT_SEPARATOR.join(batch)
        bounds: List[Tuple[int, int]] = []
        cursor = 0
        for text in batch:
            bounds.append((cursor, cursor + len(text)))
            cursor += len(text) + len(TEXT_SEPARATOR)
        result, replacements = self._call(joined)

        per_item: List[List[Tuple[int, int, str, str]]] = [[] for _ in batch]
        touched: set = set()
        for start, end, entity_type, replacement in replacements:
            owner = next(
                (
                    k
                    for k, (low, high) in enumerate(bounds)
                    if low <= start and end <= high
                ),
                None,
            )
            if owner is None:
                # The cells this span reaches into, separators included.
                touched.update(
                    k
                    for k, (low, high) in enumerate(bounds)
                    if start < high + len(TEXT_SEPARATOR) and low < end
                )
                continue
            low = bounds[owner][0]
            per_item[owner].append((start - low, end - low, entity_type, replacement))

        rebuilt = [self._rebuild(text, items) for text, items in zip(batch, per_item)]
        middle = len(batch) // 2
        if self._rebuild(joined, replacements) != result["anonymized_text"]:
            # The answer does not rebuild from its own replacements: nothing
            # in it is trusted, and each half is redone on its own.
            self.batches_redone += 1
            self._batch(batch[:middle], results)
            self._batch(batch[middle:], results)
            return
        if touched:
            # A replacement crossed a cell boundary. Only the cells it reached
            # are redone, each on its own; every other cell's replacements
            # mapped exactly and stand. Redoing both halves instead repeats the
            # same crossing at every level - on a column of record UUIDs that
            # was 147 model calls for 159 distinct values.
            self.batches_redone += 1
            merge_counts(self.categories, result.get("detected_entities", {}))
            untouched = [k for k in range(len(batch)) if k not in touched]
            for k in untouched:
                results[batch[k]] = (
                    rebuilt[k],
                    [(s, e, t) for s, e, t, _r in per_item[k]],
                )
            for k in sorted(touched):
                self._single(batch[k], results)
            review, residual = self._flags(result)
            if (review or residual) and untouched:
                self._attribute([batch[k] for k in untouched])
            return
        merge_counts(self.categories, result.get("detected_entities", {}))
        for text, value, items in zip(batch, rebuilt, per_item):
            results[text] = (value, [(s, e, t) for s, e, t, _r in items])
        review, residual = self._flags(result)
        if review or residual:
            # Every replacement mapped to its cell and stands. Which cells
            # raised the flags is found by asking smaller parts.
            self._attribute(batch[:middle])
            self._attribute(batch[middle:])

    def _attribute(self, batch: List[str]) -> None:
        """Find the cells of a batch that raise a review or residual flag alone.

        Only the flags are read; the replacements of these calls are not used.
        """
        if len(batch) == 1:
            result, _replacements = self._call(batch[0])
            self._flag(batch[0], result)
            return
        result, _replacements = self._call(TEXT_SEPARATOR.join(batch))
        review, residual = self._flags(result)
        if not (review or residual):
            return
        middle = len(batch) // 2
        self._attribute(batch[:middle])
        self._attribute(batch[middle:])


# -- the handler ------------------------------------------------------------------------


def _looks_like_header(rows: List[List[str]]) -> bool:
    first = [cell.strip() for cell in rows[0]]
    data = [row for row in rows[1:51] if row]
    if not first or any(not cell or len(cell) > 64 for cell in first):
        return False
    if len({cell.lower() for cell in first}) != len(first):
        return False
    if any(
        _measurement(cell) or _DATE_VALUE.match(cell) or _whole_cell_structured(cell)
        for cell in first
    ):
        return False
    if any(classify_header(cell) != TEXT_COLUMN for cell in first):
        return True
    for index, cell in enumerate(first):
        column = [row[index].strip() for row in data if row[index].strip()]
        if column and all(
            _measurement(value) or _DATE_VALUE.match(value) for value in column
        ):
            return True
    # Labels the vocabulary does not know - "diagnosis", "note" - still read as
    # a header when every cell is a short label and at least one column's label
    # never recurs below it. A wrong call costs nothing: header cells are
    # anonymized as text like any other cell.
    if data and all(_LABEL.fullmatch(cell) for cell in first):
        return any(
            cell.casefold() not in {row[index].strip().casefold() for row in data}
            for index, cell in enumerate(first)
        )
    return False


#: Review findings listed per file; the count is always exact.
MAX_REVIEW_FINDINGS = 200


def _review_category(code: str) -> str:
    """The detector category a first-pass review reason names."""
    match = re.fullmatch(r"ambiguous_([a-z0-9_]+)_requires_review", code)
    if match:
        return match.group(1).upper()
    return "AGE" if code == "uncertain_age_reference" else "UNCLASSIFIED"


def anonymize_csv_bytes(
    content: bytes,
    profile: str = "strict",
    text_anonymizer: Optional[Callable] = None,
    residual_scanner: Optional[Callable] = None,
    tabular_manifest: Optional[Mapping[str, Any]] = None,
    tabular_profile: Optional[str] = None,
) -> MultiformatResult:
    """Anonymize a CSV cell by cell into a new file, and verify that file.

    The same bounded-memory engine as a streamed upload (``services.csv_stream``)
    reads these bytes; only the whole-file size limits of this entry point
    differ. There is one implementation of every CSV privacy rule.
    """
    from services import csv_stream

    if not isinstance(content, (bytes, bytearray)):
        raise _reject("CSV content must be bytes", 500)
    if not content:
        raise _reject("CSV input is empty")
    if len(content) > MAX_CSV_BYTES:
        raise _reject(f"CSV uploads must be {MAX_CSV_BYTES} bytes or smaller", 413)
    raw = bytes(content)
    sink = io.BytesIO()

    class _Keep(io.BytesIO):
        def close(self) -> None:
            sink.write(self.getvalue())
            super().close()

    outcome = csv_stream.anonymize_csv_stream(
        csv_stream.Source.of_bytes(raw),
        _Keep,
        lambda: io.BytesIO(sink.getvalue()),
        profile=profile,
        limits=csv_stream.whole_file_limits(),
        text_anonymizer=text_anonymizer,
        residual_scanner=residual_scanner,
        tabular_manifest=tabular_manifest,
        tabular_profile=tabular_profile,
    )
    result = outcome.result
    if outcome.output_size and sink.getvalue():
        output = sink.getvalue()
        if hashlib.sha256(output).hexdigest() != outcome.output_sha256:
            result.error(REASON_WRITE_FAILED)
        result.output_bytes = output
        result.output_digest = outcome.output_sha256
        result.output_size = len(output)
    if hashlib.sha256(bytes(content)).hexdigest() != hashlib.sha256(raw).hexdigest():
        result.error(REASON_INPUT_CHANGED)
    return result


def _generated_digits(cell: str) -> str:
    """The digits inside generated tokens, which may repeat an original's digits."""
    return " ".join(match.group(0) for match in SURROGATE_PATTERN.finditer(cell))
