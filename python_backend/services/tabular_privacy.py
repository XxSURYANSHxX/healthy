"""Shared, versioned privacy policy for CSV and XLSX artifacts.

This module owns the decisions that must agree across tabular sanitization,
verification and the independent release audit.  It deliberately does not own
model loading: ambiguous names and locations still go through ``full_v1``.
Stable syntax, labelled context, column roles, quasi-identifier configuration
and clinical invariants are decided here before a model is asked.

The policy is Safe Harbor-oriented engineering, not legal certification or an
Expert Determination.  Every public structure returned by this module contains
only category names, column ordinals, counts and reason codes; source values
never leave the caller.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

POLICY_VERSION = "tabular-privacy-v1"

PROFILE_ENV_VAR = "BIOBLOCK_TABULAR_PRIVACY_PROFILE"
PROFILE_IDENTIFIER_REMOVAL = "identifier_removal"
PROFILE_SAFE_HARBOR_PLUS_QI = "safe_harbor_plus_qi"
# Accepted only so an existing deployment does not silently change behavior.
PROFILE_RISK_BASED_LEGACY = "risk_based"
PROFILES = frozenset({PROFILE_IDENTIFIER_REMOVAL, PROFILE_SAFE_HARBOR_PLUS_QI})

ROLE_DIRECT_IDENTIFIER = "direct_identifier"
ROLE_QUASI_IDENTIFIER = "quasi_identifier"
ROLE_SENSITIVE_CLINICAL = "sensitive_clinical"
ROLE_CLINICAL_UTILITY = "clinical_utility"
ROLE_NON_SENSITIVE = "non_sensitive"
ROLE_UNKNOWN = "unknown"
COLUMN_ROLES = frozenset(
    {
        ROLE_DIRECT_IDENTIFIER,
        ROLE_QUASI_IDENTIFIER,
        ROLE_SENSITIVE_CLINICAL,
        ROLE_CLINICAL_UTILITY,
        ROLE_NON_SENSITIVE,
        ROLE_UNKNOWN,
    }
)

ACTION_KEEP = "keep"
ACTION_REPLACE = "replace"
ACTION_GENERALIZE = "generalize"
ACTION_SUPPRESS = "suppress"
ACTION_HOLD = "hold"

REASON_UNRESOLVED_QI = "tabular_quasi_identifier_unresolved"
REASON_UNSUPPORTED_RISK = "tabular_risk_condition_unsupported"
REASON_CLINICAL_CHANGED = "clinical_utility_changed"
REASON_CONTENT_CHANGED = "tabular_nonclinical_content_changed"
REASON_FORGED_TOKEN = "tabular_forged_replacement_token"
REASON_MANIFEST_INVALID = "tabular_policy_manifest_invalid"


@dataclass(frozen=True)
class SafeHarborCategory:
    number: int
    slug: str
    label: str
    entity_types: Tuple[str, ...]


SAFE_HARBOR_CATEGORIES: Tuple[SafeHarborCategory, ...] = (
    SafeHarborCategory(1, "names", "Names", ("PERSON",)),
    SafeHarborCategory(
        2,
        "geography",
        "Geographic subdivisions smaller than a state",
        ("ADDRESS", "LOCATION"),
    ),
    SafeHarborCategory(
        3,
        "dates",
        "Date elements related to an individual",
        ("DATE", "DATE_TIME", "AGE_OVER_89"),
    ),
    SafeHarborCategory(4, "telephone_numbers", "Telephone numbers", ("PHONE_NUMBER",)),
    SafeHarborCategory(5, "fax_numbers", "Fax numbers", ("FAX_NUMBER",)),
    SafeHarborCategory(6, "email_addresses", "Email addresses", ("EMAIL_ADDRESS",)),
    SafeHarborCategory(
        7, "social_security_numbers", "Social Security numbers", ("US_SSN", "SSN")
    ),
    SafeHarborCategory(
        8,
        "medical_record_numbers",
        "Medical-record numbers",
        ("MEDICAL_RECORD_NUMBER", "PATIENT_ID"),
    ),
    SafeHarborCategory(
        9,
        "health_plan_numbers",
        "Health-plan beneficiary numbers",
        ("HEALTH_PLAN_ID", "INSURANCE_ID"),
    ),
    SafeHarborCategory(10, "account_numbers", "Account numbers", ("ACCOUNT_NUMBER",)),
    SafeHarborCategory(
        11,
        "certificate_license_numbers",
        "Certificate and licence numbers",
        ("CERTIFICATE_NUMBER", "LICENSE_NUMBER"),
    ),
    SafeHarborCategory(
        12,
        "vehicle_identifiers",
        "Vehicle identifiers and serial numbers",
        ("VEHICLE_ID",),
    ),
    SafeHarborCategory(
        13,
        "device_identifiers",
        "Device identifiers and serial numbers",
        ("DEVICE_ID",),
    ),
    SafeHarborCategory(14, "urls", "URLs", ("URL",)),
    SafeHarborCategory(15, "ip_addresses", "IP addresses", ("IP_ADDRESS",)),
    SafeHarborCategory(
        16, "biometric_identifiers", "Biometric identifiers", ("BIOMETRIC_ID",)
    ),
    SafeHarborCategory(
        17,
        "full_face_images",
        "Full-face photographs and comparable images",
        ("IMAGE_REFERENCE", "FULL_FACE_IMAGE"),
    ),
    SafeHarborCategory(
        18,
        "other_unique_identifiers",
        "Other unique identifying numbers, characteristics or codes",
        ("IDENTIFIER", "OTHER_UNIQUE_ID", "ACCESSION_NUMBER", "USERNAME"),
    ),
)

_CATEGORY_BY_ENTITY = {
    entity: category.slug
    for category in SAFE_HARBOR_CATEGORIES
    for entity in category.entity_types
}


class ManifestError(ValueError):
    """A manifest cannot safely configure this policy."""


def _fold(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def header_tokens(header: str) -> Tuple[str, ...]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", header or "")
    return tuple(token for token in re.split(r"[^a-z0-9]+", spaced.casefold()) if token)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _manifest_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _optional_int(value: Any, name: str, minimum: int = 1) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ManifestError(f"{name} must be an integer of at least {minimum}")
    return value


def _optional_float(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{name} must be a number between 0 and 1")
    number = float(value)
    if not 0 <= number <= 1:
        raise ManifestError(f"{name} must be a number between 0 and 1")
    return number


@dataclass(frozen=True)
class ColumnDeclaration:
    index: Optional[int]
    header: Optional[str]
    role: str
    entity_type: Optional[str] = None
    qi_kind: Optional[str] = None
    transformation: str = "auto"
    header_sha256: Optional[str] = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ColumnDeclaration":
        if not isinstance(value, Mapping):
            raise ManifestError("each column declaration must be an object")
        index = value.get("index")
        header = value.get("header")
        header_sha256 = value.get("header_sha256")
        if header_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", str(header_sha256)
        ):
            raise ManifestError("column header digest is invalid")
        if index is None and not isinstance(header, str) and header_sha256 is None:
            raise ManifestError("a column declaration needs an index or header")
        if index is not None and (
            isinstance(index, bool) or not isinstance(index, int) or index < 0
        ):
            raise ManifestError("column index must be a non-negative integer")
        if header is not None and (not isinstance(header, str) or not header.strip()):
            raise ManifestError("column header must be a non-empty string")
        role = str(value.get("role") or "")
        if role not in COLUMN_ROLES:
            raise ManifestError("column role is not supported")
        entity_type = value.get("entity_type")
        qi_kind = value.get("qi_kind")
        if entity_type is not None and not re.fullmatch(
            r"[A-Z][A-Z0-9_]{1,63}", str(entity_type)
        ):
            raise ManifestError("entity_type is invalid")
        if qi_kind is not None and str(qi_kind) not in {
            "age",
            "year",
            "sex",
            "zip",
            "location",
            "race_ethnicity",
            "occupation",
            "institution",
            "language",
            "marital_status",
            "facility",
            "service_line",
        }:
            raise ManifestError("qi_kind is not supported")
        transformation = str(value.get("transformation") or "auto")
        if transformation not in {"auto", "generalize", "suppress"}:
            raise ManifestError("column transformation is not supported")
        normalized_header = header.strip() if isinstance(header, str) else None
        digest = (
            hashlib.sha256(_fold(normalized_header).encode()).hexdigest()
            if normalized_header
            else None
        )
        if digest and header_sha256 and digest != header_sha256:
            raise ManifestError("column header digest does not match")
        return cls(
            index,
            normalized_header,
            role,
            str(entity_type) if entity_type else None,
            str(qi_kind) if qi_kind else None,
            transformation,
            digest or header_sha256,
        )


@dataclass(frozen=True)
class PolicyConfig:
    profile: str = PROFILE_IDENTIFIER_REMOVAL
    k: int = 5
    l: Optional[int] = 2
    t: Optional[float] = None
    columns: Tuple[ColumnDeclaration, ...] = ()
    source: str = "default"
    manifest_sha256: Optional[str] = None

    @property
    def qi_enabled(self) -> bool:
        return self.profile == PROFILE_SAFE_HARBOR_PLUS_QI

    def public_metadata(self) -> Dict[str, Any]:
        return {
            "policy_version": POLICY_VERSION,
            "profile": self.profile,
            "manifest_sha256": self.manifest_sha256,
            "k_required": self.k if self.qi_enabled else None,
            "l_required": self.l if self.qi_enabled else None,
            "t_required": self.t if self.qi_enabled else None,
            "configuration_source": self.source,
        }


def normalize_profile(value: Optional[str]) -> str:
    profile = (value or PROFILE_IDENTIFIER_REMOVAL).strip().casefold()
    if profile == PROFILE_RISK_BASED_LEGACY:
        return PROFILE_SAFE_HARBOR_PLUS_QI
    if profile not in PROFILES:
        raise ManifestError("unsupported tabular privacy profile")
    return profile


def policy_config(
    manifest: Optional[Mapping[str, Any]] = None,
    *,
    profile: Optional[str] = None,
    allow_environment_manifest: bool = False,
) -> PolicyConfig:
    """Validate an artifact-bound manifest or use the fixed identifier default.

    The compatibility keyword is ignored: mutable process environment never
    selects the profile or the manifest for an artifact.
    """
    source = "argument" if manifest is not None else "default"
    loaded: Optional[Mapping[str, Any]] = manifest
    if loaded is None:
        # A profile-only environment setting is mutable process state, not an
        # artifact declaration. Callers may pass an explicit profile, or bind
        # a versioned deployment manifest before processing the artifact.
        requested = profile
        selected = normalize_profile(requested)
        return PolicyConfig(
            profile=selected,
            t=(
                0.2
                if (requested or "").strip().casefold() == PROFILE_RISK_BASED_LEGACY
                else None
            ),
            source=source,
        )
    if str(loaded.get("policy_version") or POLICY_VERSION) != POLICY_VERSION:
        raise ManifestError("tabular policy version is not supported")
    requested = profile or loaded.get("profile")
    selected = normalize_profile(requested)
    k = _optional_int(loaded.get("k", 5), "k") or 5
    l_value = _optional_int(loaded.get("l", 2), "l")
    t_value = _optional_float(
        loaded.get(
            "t",
            (
                0.2
                if str(requested or "").strip().casefold() == PROFILE_RISK_BASED_LEGACY
                else None
            ),
        ),
        "t",
    )
    raw_columns = loaded.get("columns") or ()
    if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
        raise ManifestError("columns must be a list")
    columns = tuple(ColumnDeclaration.from_dict(item) for item in raw_columns)
    seen_indices = [item.index for item in columns if item.index is not None]
    if len(seen_indices) != len(set(seen_indices)):
        raise ManifestError("a column index may be declared only once")
    return PolicyConfig(
        profile=selected,
        k=k,
        l=l_value,
        t=t_value,
        columns=columns,
        source=source,
        manifest_sha256=_manifest_digest(loaded),
    )


def config_from_metadata(metadata: Optional[Mapping[str, Any]]) -> PolicyConfig:
    """Re-validate a sanitizer's value-free declaration for final-byte audit."""
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("policy_version") != POLICY_VERSION
    ):
        raise ManifestError("tabular policy provenance is missing or invalid")
    raw_columns = metadata.get("manifest_columns")
    inferred_fallback = raw_columns is None
    raw_columns = (
        (metadata.get("column_roles") or ()) if inferred_fallback else raw_columns
    )
    columns: List[ColumnDeclaration] = []
    if not isinstance(raw_columns, Sequence) or isinstance(raw_columns, (str, bytes)):
        raise ManifestError("tabular column provenance is invalid")
    for item in raw_columns:
        if not isinstance(item, Mapping):
            raise ManifestError("tabular column provenance is invalid")
        raw_index = item.get("index")
        if raw_index is None:
            match = re.fullmatch(r"column_(\d+)", str(item.get("column") or ""))
            if not match:
                raise ManifestError("tabular column provenance is invalid")
            raw_index = int(match.group(1))
        columns.append(
            ColumnDeclaration.from_dict(
                {
                    "index": raw_index,
                    "header_sha256": item.get("header_sha256"),
                    "role": item.get("role"),
                    "entity_type": item.get("entity_type"),
                    "qi_kind": item.get("qi_kind"),
                    "transformation": item.get("transformation") or "auto",
                }
            )
        )
    manifest = {
        "policy_version": POLICY_VERSION,
        "profile": metadata.get("profile"),
        "k": metadata.get("k_required") or 5,
        "l": metadata.get("l_required"),
        "t": metadata.get("t_required"),
        "columns": [
            {
                "index": item.index,
                "header_sha256": item.header_sha256,
                "role": item.role,
                "entity_type": item.entity_type,
                "qi_kind": item.qi_kind,
                "transformation": item.transformation,
            }
            for item in columns
        ],
    }
    config = policy_config(manifest, allow_environment_manifest=False)
    return PolicyConfig(
        profile=config.profile,
        k=config.k,
        l=config.l,
        t=config.t,
        columns=config.columns,
        source="audit_provenance",
        manifest_sha256=(
            str(metadata.get("manifest_sha256"))
            if metadata.get("manifest_sha256")
            else None
        ),
    )


def config_for_sheet(
    config: PolicyConfig, headers: Sequence[str], *, primary: bool = False
) -> PolicyConfig:
    """Keep only declarations whose schema actually belongs to this sheet.

    A workbook-wide column ordinal is not authority over every hidden sheet.
    Named declarations match their own field name; legacy index-only
    declarations apply only to the primary worksheet.
    """
    selected = []
    for item in config.columns:
        if item.index is not None:
            if item.index >= len(headers):
                if primary:
                    raise ManifestError("declared tabular column index is absent")
                continue
            if item.header is None and item.header_sha256 is None:
                if primary:
                    selected.append(item)
            elif header_matches(item, headers[item.index]):
                selected.append(item)
            elif primary:
                raise ManifestError("declared tabular header contradicts artifact")
        elif (item.header is not None or item.header_sha256 is not None) and sum(
            header_matches(item, header) for header in headers
        ) == 1:
            selected.append(item)
        elif primary:
            raise ManifestError("declared tabular header is absent or ambiguous")
    return replace(config, columns=tuple(selected))


def header_matches(item: ColumnDeclaration, header: str) -> bool:
    """Match a declaration without retaining the raw header in audit metadata."""
    if item.header is not None:
        return _fold(header) == _fold(item.header)
    if item.header_sha256 is not None:
        return hashlib.sha256(_fold(header).encode()).hexdigest() == item.header_sha256
    return False


@dataclass(frozen=True)
class ColumnPolicy:
    index: int
    role: str
    entity_type: Optional[str] = None
    qi_kind: Optional[str] = None
    transformation: str = "auto"
    explicit: bool = False
    unresolved: bool = False

    def public_dict(self) -> Dict[str, Any]:
        return {
            "column": f"column_{self.index}",
            "role": self.role,
            "entity_type": self.entity_type,
            "qi_kind": self.qi_kind,
            "transformation": self.transformation,
            "explicit": self.explicit,
            "unresolved": self.unresolved,
        }


def _patterns(*phrases: str) -> Tuple[frozenset[str], ...]:
    return tuple(frozenset(phrase.split()) for phrase in phrases)


# Longest matching pattern wins.  Stable, labelled identifiers are before
# broad words such as ``name`` and ``id``.
_DIRECT_HEADERS: Tuple[Tuple[str, Tuple[frozenset[str], ...]], ...] = (
    (
        "DATE",
        _patterns(
            "dob",
            "date of birth",
            "birth date",
            "death date",
            "admission date",
            "discharge date",
            "collection date",
            "service date",
            "timestamp",
        ),
    ),
    ("FAX_NUMBER", _patterns("fax", "fax number")),
    (
        "PHONE_NUMBER",
        _patterns(
            "phone",
            "telephone",
            "mobile",
            "cell phone",
            "pager",
            "bleep",
            "contact number",
            "direct line",
            "extension",
        ),
    ),
    ("EMAIL_ADDRESS", _patterns("email", "email address", "e mail")),
    ("US_SSN", _patterns("ssn", "social security")),
    (
        "MEDICAL_RECORD_NUMBER",
        _patterns(
            "mrn", "medical record", "record number", "chart number", "hospital number"
        ),
    ),
    (
        "HEALTH_PLAN_ID",
        _patterns(
            "health plan",
            "beneficiary",
            "insurance id",
            "member id",
            "member number",
            "policy number",
            "subscriber id",
        ),
    ),
    (
        "ACCOUNT_NUMBER",
        _patterns("account", "account number", "billing number", "iban"),
    ),
    (
        "LICENSE_NUMBER",
        _patterns(
            "license number",
            "licence number",
            "driver license",
            "gmc",
            "nmc",
            "hcpc",
            "gdc",
            "gphc",
            "npi",
            "dea",
            "provider number",
            "registration number",
            "provider registration",
        ),
    ),
    ("CERTIFICATE_NUMBER", _patterns("certificate number", "certificate id")),
    (
        "VEHICLE_ID",
        _patterns(
            "vin",
            "vehicle id",
            "vehicle serial",
            "license plate",
            "licence plate",
            "registration plate",
        ),
    ),
    (
        "DEVICE_ID",
        _patterns(
            "device id",
            "device serial",
            "serial number",
            "udi",
            "imei",
            "implant serial",
        ),
    ),
    ("URL", _patterns("url", "uri", "website", "web address", "homepage")),
    ("IP_ADDRESS", _patterns("ip", "ip address", "ipv4", "ipv6")),
    (
        "BIOMETRIC_ID",
        _patterns(
            "biometric",
            "fingerprint",
            "voiceprint",
            "voice print",
            "retina scan",
            "iris scan",
            "face template",
        ),
    ),
    (
        "IMAGE_REFERENCE",
        _patterns(
            "full face photo",
            "photograph",
            "headshot",
            "image url",
            "image path",
            "photo url",
            "photo path",
            "photo reference",
            "image reference",
            "image file",
            "photo file",
        ),
    ),
    (
        "ADDRESS",
        _patterns("street address", "home address", "address line", "residence"),
    ),
    (
        "LOCATION",
        _patterns(
            "city",
            "town",
            "county",
            "precinct",
            "geocode",
            "latitude",
            "longitude",
            "suburb",
            "village",
            "neighborhood",
        ),
    ),
    (
        "PERSON",
        _patterns(
            "patient name",
            "person name",
            "full name",
            "first name",
            "last name",
            "surname",
            "forename",
            "maiden name",
            "relative name",
            "spouse name",
            "guardian name",
            "employer name",
            "household member",
            "household name",
            "next of kin",
        ),
    ),
    (
        "PATIENT_ID",
        _patterns("patient id", "patient number", "participant id", "subject id"),
    ),
    (
        "ACCESSION_NUMBER",
        _patterns(
            "accession",
            "specimen id",
            "sample id",
            "requisition",
            "barcode",
            "order number",
        ),
    ),
    (
        "IDENTIFIER",
        _patterns(
            "identifier",
            "uuid",
            "guid",
            "encounter id",
            "visit id",
            "case id",
            "tracking number",
            "reference number",
            "employee id",
            "unique code",
            "unique id",
        ),
    ),
)

_QI_HEADERS: Tuple[Tuple[str, Tuple[frozenset[str], ...]], ...] = (
    ("age", _patterns("age", "age years", "age band")),
    ("year", _patterns("year", "birth year", "yob")),
    ("sex", _patterns("sex", "gender")),
    ("zip", _patterns("zip", "zipcode", "postal code", "postcode", "postal area")),
    ("race_ethnicity", _patterns("race", "ethnicity", "ancestry")),
    ("marital_status", _patterns("marital status")),
    ("occupation", _patterns("occupation", "profession", "employment", "job title")),
    ("language", _patterns("language", "preferred language")),
    ("location", _patterns("state", "province", "region", "district", "territory")),
    ("facility", _patterns("facility", "hospital", "clinic", "institution", "site")),
    ("service_line", _patterns("service line", "department", "ward")),
)

_SENSITIVE_HEADERS = _patterns(
    "diagnosis",
    "condition",
    "disease",
    "disorder",
    "icd",
    "snomed",
    "medication",
    "procedure",
    "pathology",
    "result interpretation",
    "clinical status",
    "severity",
)
_UTILITY_HEADERS = _patterns(
    "loinc",
    "rxnorm",
    "cvx",
    "test",
    "analyte",
    "lab value",
    "result",
    "unit",
    "reference range",
    "dose",
    "route",
    "frequency",
    "vital sign",
    "stain",
    "sequence",
    "device type",
    "procedure name",
    "medication name",
    "vaccine",
    "systolic",
    "diastolic",
    "blood pressure",
    "bp",
    "heart rate",
    "pulse",
    "temperature",
    "weight",
    "height",
    "bmi",
    "spo2",
    "glucose",
    "sodium",
    "potassium",
    "creatinine",
    "hba1c",
    "hemoglobin",
    "haemoglobin",
    "platelet",
    "count",
    "level",
    "measurement",
    # Named analytes: a column whose header is the name of a laboratory test
    # holds that test's name or result, not a person.
    "testosterone",
    "estradiol",
    "cortisol",
    "insulin",
    "thyroxine",
    "tsh",
    "ferritin",
    "cholesterol",
    "triglycerides",
    "ldl",
    "hdl",
    "albumin",
    "bilirubin",
    "urea",
    "calcium",
    "magnesium",
    "troponin",
    "lactate",
    "haematocrit",
    "hematocrit",
)


def _best_header_match(tokens: frozenset[str], rules) -> Optional[str]:
    best: Optional[Tuple[int, int, str]] = None
    for order, (kind, patterns) in enumerate(rules):
        for pattern in patterns:
            if pattern <= tokens:
                candidate = (len(pattern), -order, kind)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    return best[2] if best else None


def _declared_for(
    index: int, header: str, config: PolicyConfig, header_counts: Mapping[str, int]
) -> Optional[ColumnDeclaration]:
    by_index = next((item for item in config.columns if item.index == index), None)
    if by_index is not None:
        if (
            by_index.header is None and by_index.header_sha256 is None
        ) or header_matches(by_index, header):
            return by_index
        raise ManifestError("declared tabular header contradicts artifact")
    folded = _fold(header)
    named = [
        item
        for item in config.columns
        if item.index is None and header_matches(item, header)
    ]
    if not named:
        return None
    # A declaration by a duplicate name is ambiguous; require an ordinal.
    if header_counts.get(folded, 0) != 1 or len(named) != 1:
        raise ManifestError("duplicate headers must be declared by column index")
    return named[0]


_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SSN = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_PHONE = re.compile(r"^(?:\+\d[\d ()-]{6,}\d|(?:\(?\d{3}\)?[-.\s])?\d{3}[-.\s]\d{4})$")
_URL = re.compile(r"^(?:https?://|www\.)\S+$", re.IGNORECASE)
_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
_YEAR = re.compile(r"^(?:18|19|20|21)\d{2}$")
_DATE = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4})$",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"^<[A-Z][A-Z0-9_]*_[1-9][0-9]*>$")
_EMBEDDED_TOKEN = re.compile(r"(?<!\w)<[A-Z][A-Z0-9_]*_[1-9][0-9]*>(?!\w)")
_EMBEDDED_EMAIL = re.compile(r"(?<![\w@])[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w@])")
_EMBEDDED_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_EMBEDDED_URL = re.compile(r"(?<!\w)(?:https?://|www\.)[^\s;,]+", re.IGNORECASE)
_EMBEDDED_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


def is_document_version_date(
    value: str, header: str, column: Optional[ColumnPolicy]
) -> bool:
    """An explicitly declared, structurally validated document date.

    A generic ``date`` or a self-descriptive header alone is not enough to
    exempt a date about a person. The declaration and the exact field meaning
    must agree, and the value must be a valid ISO calendar date.
    """
    if (
        column is None
        or not column.explicit
        or column.role != ROLE_NON_SENSITIVE
        or header_tokens(header) != ("document", "version", "date")
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip())
    ):
        return False
    try:
        date.fromisoformat(value.strip())
    except ValueError:
        return False
    return True


def stable_identifier_type(value: str) -> Optional[str]:
    """A whole-value identifier with stable syntax, independent of a model."""
    stripped = value.strip()
    if not stripped:
        return None
    if _EMAIL.fullmatch(stripped):
        return "EMAIL_ADDRESS"
    if _SSN.fullmatch(stripped):
        return "US_SSN"
    if _URL.fullmatch(stripped):
        return "URL"
    try:
        ipaddress.ip_address(stripped.strip("[]"))
        return "IP_ADDRESS"
    except ValueError:
        pass
    if _UUID.fullmatch(stripped):
        return "OTHER_UNIQUE_ID"
    if _DATE.fullmatch(stripped):
        return "DATE"
    if _PHONE.fullmatch(stripped) and sum(ch.isdigit() for ch in stripped) >= 7:
        return "PHONE_NUMBER"
    return None


_SEX_VALUES = frozenset(
    {"m", "f", "male", "female", "intersex", "nonbinary", "non-binary"}
)
_RACE_VALUES = frozenset(
    {
        "asian",
        "black",
        "african american",
        "white",
        "native american",
        "alaska native",
        "native hawaiian",
        "pacific islander",
        "multiracial",
        "hispanic",
        "latino",
        "non-hispanic",
        "two or more races",
    }
)
_MARITAL_VALUES = frozenset({"single", "married", "divorced", "widowed", "separated"})


def _value_inferred_role(values: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    cleaned = [_fold(value) for value in values if _fold(value)]
    if not cleaned:
        return None, None
    typed = [stable_identifier_type(value) for value in values if _fold(value)]
    nonempty_types = [item for item in typed if item]
    if (
        nonempty_types
        and len(nonempty_types) == len(cleaned)
        and len(set(nonempty_types)) == 1
    ):
        return ROLE_DIRECT_IDENTIFIER, nonempty_types[0]
    observed = frozenset(cleaned)
    if observed <= _SEX_VALUES:
        return ROLE_QUASI_IDENTIFIER, "sex"
    if observed <= _RACE_VALUES:
        return ROLE_QUASI_IDENTIFIER, "race_ethnicity"
    if observed <= _MARITAL_VALUES:
        return ROLE_QUASI_IDENTIFIER, "marital_status"
    return None, None


def classify_columns(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]] = (),
    config: Optional[PolicyConfig] = None,
) -> Tuple[ColumnPolicy, ...]:
    """Classify columns from declarations, headers and bounded value evidence."""
    config = config or policy_config(allow_environment_manifest=False)
    counts: Dict[str, int] = {}
    for header in headers:
        folded = _fold(header)
        counts[folded] = counts.get(folded, 0) + 1
    answer: List[ColumnPolicy] = []
    for index, header in enumerate(headers):
        declared = _declared_for(index, header, config, counts)
        if declared is not None:
            answer.append(
                ColumnPolicy(
                    index,
                    declared.role,
                    declared.entity_type,
                    declared.qi_kind,
                    declared.transformation,
                    True,
                )
            )
            continue
        tokens = frozenset(header_tokens(header))
        direct = _best_header_match(tokens, _DIRECT_HEADERS)
        quasi = _best_header_match(tokens, _QI_HEADERS)
        sensitive = any(pattern <= tokens for pattern in _SENSITIVE_HEADERS)
        utility = any(pattern <= tokens for pattern in _UTILITY_HEADERS)
        if direct:
            qi_kind = (
                "year"
                if direct == "DATE"
                else (
                    (
                        "zip"
                        if tokens & {"zip", "zipcode", "postcode", "postal"}
                        else "location"
                    )
                    if direct == "LOCATION"
                    else None
                )
            )
            answer.append(
                ColumnPolicy(index, ROLE_DIRECT_IDENTIFIER, direct, qi_kind=qi_kind)
            )
            continue
        if quasi and sensitive:
            answer.append(ColumnPolicy(index, ROLE_UNKNOWN, unresolved=True))
            continue
        if quasi:
            answer.append(ColumnPolicy(index, ROLE_QUASI_IDENTIFIER, qi_kind=quasi))
            continue
        if sensitive:
            answer.append(ColumnPolicy(index, ROLE_SENSITIVE_CLINICAL))
            continue
        if utility:
            answer.append(ColumnPolicy(index, ROLE_CLINICAL_UTILITY))
            continue
        values = [str(row[index]) for row in rows if index < len(row)][:1000]
        inferred_role, inferred_kind = _value_inferred_role(values)
        if inferred_role == ROLE_DIRECT_IDENTIFIER:
            answer.append(ColumnPolicy(index, inferred_role, inferred_kind))
            continue
        if inferred_role == ROLE_QUASI_IDENTIFIER:
            answer.append(ColumnPolicy(index, inferred_role, qi_kind=inferred_kind))
            continue
        # A high-cardinality, otherwise unclassified field is a suspected QI,
        # not automatically a direct identifier.  Clinical headers were
        # already removed from this branch so rare diagnoses are not punished.
        present = [_fold(value) for value in values if _fold(value)]
        unique_ratio = len(set(present)) / len(present) if len(present) >= 5 else 0.0
        unresolved = bool(config.qi_enabled and unique_ratio >= 0.8)
        answer.append(ColumnPolicy(index, ROLE_UNKNOWN, unresolved=unresolved))
    return tuple(answer)


_ICD10 = re.compile(r"^[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?$")
_ICD11 = re.compile(r"^[0-9A-Z][0-9A-Z][0-9A-Z](?:\.[0-9A-Z]{1,4})?$", re.IGNORECASE)
_LOINC = re.compile(r"^\d{1,5}-\d$")
_SNOMED = re.compile(r"^\d{6,18}$")
_MEASUREMENT = re.compile(
    r"^(?:[<>~≤≥]=?\s*)?[-+]?\d+(?:\.\d+)?(?:\s*(?:-|–|to|/)\s*[-+]?\d+(?:\.\d+)?)?"
    r"\s*(?:%|mmHg|mU/L|mOsm/kg|mmol/mol|mg|mcg|µg|g|kg|mL|L|IU|U|bpm|°[CF])?$",
    re.IGNORECASE,
)
_UNIT = re.compile(
    r"^(?:%|mmhg|mu/l|mosm/kg|mmol/mol|mg|mcg|µg|g|kg|ml|l|iu|u|bpm|"
    r"[a-zµ]+(?:/[a-z0-9.²³]+)+)$",
    re.IGNORECASE,
)
_DURATION = re.compile(
    r"^(?:return|follow[- ]?up|review)?\s*(?:in\s+)?\d+(?:\.\d+)?\s+"
    r"(?:minutes?|hours?|days?|weeks?|months?|years?)$",
    re.IGNORECASE,
)
_DOSE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|units?)\b", re.IGNORECASE)
_DOSE_SCHEDULE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|ml|iu|units?)" r"(?:\s+(?:once|twice)\s+daily)?",
    re.IGNORECASE,
)
_CLINICAL_EXACT = frozenset(
    {
        "type 2 diabetes mellitus",
        "pcv20",
        "influenza vaccine",
        "pas-d stain",
        "flair",
        "dess",
        "sag pd fs",
        "nasogastric tube",
        "serum or plasma",
        "hba1c",
        "continue treatment and return in 3 months",
        "synthetic",
    }
)


def is_synthetic_disclaimer(
    value: str, header: str, column: Optional[ColumnPolicy] = None
) -> bool:
    """Preserve only a complete, fixed non-patient provenance declaration.

    The column must either carry a conventional provenance header, or be
    declared non-sensitive by the signed schema itself: the declaration, not the
    spelling of a header, says this field is not about a person. The value must
    be exactly the fixed statement in either case, so no other text, and no
    clinical, quasi-identifier or undeclared column, is exempted by this rule.
    """
    declared_non_sensitive = (
        column is not None and column.explicit and column.role == ROLE_NON_SENSITIVE
    )
    return (
        header.casefold() in {"fixture_label", "synthetic_disclaimer"}
        or declared_non_sensitive
    ) and bool(
        re.fullmatch(
            r"SYNTHETIC\s*[-:]\s*NOT A REAL PATIENT", value.strip(), re.IGNORECASE
        )
    )


_GENERIC_INSTITUTION = re.compile(
    r"(?:community|outpatient|inpatient|primary care|urgent care|general|"
    r"rural|urban|district|regional)\s+"
    r"(?:clinic|centre|center|hospital|practice|unit)",
    re.IGNORECASE,
)


def is_generic_institution_category(value: str) -> bool:
    """A facility *type*, not a named organization or employer."""
    return bool(_GENERIC_INSTITUTION.fullmatch(value.strip()))


_CLINICAL_WORDS = frozenset(
    {
        "diagnosis",
        "diagnoses",
        "symptom",
        "symptoms",
        "procedure",
        "medication",
        "dose",
        "route",
        "frequency",
        "laboratory",
        "analyte",
        "result",
        "unit",
        "reference",
        "range",
        "vital",
        "stain",
        "sequence",
        "vaccine",
        "treatment",
        "negative",
        "positive",
        "severity",
        "status",
        "snomed",
        "loinc",
        "rxnorm",
        "cvx",
        "icd",
        "systolic",
        "diastolic",
        "blood",
        "pressure",
        "bp",
        "heart",
        "pulse",
        "temperature",
        "weight",
        "height",
        "bmi",
        "glucose",
        "sodium",
        "potassium",
        "creatinine",
        "hba1c",
        "count",
        "level",
        "measurement",
    }
)


def is_clinical_invariant(
    value: str, header: str = "", row_context: Sequence[str] = ()
) -> bool:
    """Whether exact preservation has deterministic clinical evidence.

    This is intentionally structural and conservative.  A rare value is not
    clinical merely because it is rare, and a deterministic identifier always
    wins before this function is consulted.
    """
    text = value.strip()
    if not text or stable_identifier_type(text):
        return False
    folded = _fold(text)
    header_words = frozenset(header_tokens(header))
    context_words = header_words | frozenset(
        word for item in row_context for word in header_tokens(str(item))
    )
    if folded in _CLINICAL_EXACT:
        return True
    # These are pinned pathology terminology values, not arbitrary strings
    # made safe by a generic CODE column. The matching stain header supplies
    # the occurrence-specific context; mixed notes never match exactly.
    if folded in {"h&e stain", "trichrome stain"} and header_words & {
        "pathology",
        "stain",
    }:
        return True
    if _ICD10.fullmatch(text) and (context_words & {"icd", "icd10", "diagnosis"}):
        return True
    if _ICD11.fullmatch(text) and "icd11" in context_words:
        return True
    if _LOINC.fullmatch(text) and ("loinc" in context_words or "test" in context_words):
        return True
    if _SNOMED.fullmatch(text) and "snomed" in context_words:
        return True
    if _MEASUREMENT.fullmatch(text) and context_words & _CLINICAL_WORDS:
        return True
    if _UNIT.fullmatch(text) and (
        # A unit column: its header says only that ("unit", "units of measure").
        # A header with other words ("unit_provider") is not one, so a unit
        # value there is read like any other text.
        (
            header_words & {"unit", "units", "uom"}
            and header_words
            <= {
                "unit",
                "units",
                "uom",
                "of",
                "measure",
                "measurement",
                "laboratory",
                "lab",
                "result",
                "results",
            }
        )
        or context_words & {"result", "laboratory"}
    ):
        return True
    if _DURATION.fullmatch(text):
        return True
    if _DOSE_SCHEDULE.fullmatch(text) and context_words & {
        "dose",
        "dosage",
        "medication",
        "drug",
        "frequency",
    }:
        return True
    if folded == "reference range":
        return True
    if folded in {
        "twice daily",
        "once daily",
        "intravenous",
        "oral",
    } and context_words & {"medication", "dose", "route", "frequency", "treatment"}:
        return True
    return False


_CONTACT_LABEL = re.compile(
    r"\b(?:phone|telephone|fax|bleep|pager|page(?:\s+(?:on|via|at))?|ext(?:ension)?|"
    r"switchboard|call|contact number|direct line)\b",
    re.IGNORECASE,
)
_RECORD_LABEL = re.compile(
    r"\b(?:mrn|medical record|chart number|patient id|health plan|member id|account|"
    r"accession|specimen id|certificate|licen[cs]e|gmc|nmc|hcpc|gdc|gphc|npi|dea|"
    r"provider number|registration number|vin|serial number|udi)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OccurrenceDecision:
    action: str
    entity_type: Optional[str] = None
    category: Optional[str] = None
    reason_code: Optional[str] = None
    deterministic: bool = False
    clinical_invariant: bool = False
    content_invariant: bool = False


def occurrence_decision(
    value: Any,
    *,
    header: str = "",
    column: Optional[ColumnPolicy] = None,
    row_context: Sequence[str] = (),
) -> OccurrenceDecision:
    """Decide one cell occurrence from only its own bounded context."""
    text = "" if value is None else str(value)
    stripped = text.strip()
    if not stripped:
        return OccurrenceDecision(ACTION_KEEP)
    if _TOKEN.fullmatch(stripped):
        return OccurrenceDecision(
            ACTION_REPLACE,
            "OTHER_UNIQUE_ID",
            _CATEGORY_BY_ENTITY["OTHER_UNIQUE_ID"],
            REASON_FORGED_TOKEN,
            True,
        )
    if is_document_version_date(stripped, header, column):
        return OccurrenceDecision(
            ACTION_KEEP, deterministic=True, content_invariant=True
        )
    if is_synthetic_disclaimer(stripped, header, column):
        return OccurrenceDecision(
            ACTION_KEEP, deterministic=True, content_invariant=True
        )
    stable = stable_identifier_type(stripped)
    if (
        stable == "PHONE_NUMBER"
        and column is not None
        and column.role == ROLE_DIRECT_IDENTIFIER
        and column.entity_type == "FAX_NUMBER"
    ):
        stable = "FAX_NUMBER"
    if stable:
        return OccurrenceDecision(
            ACTION_REPLACE, stable, _CATEGORY_BY_ENTITY.get(stable), deterministic=True
        )
    if column is not None and column.role == ROLE_DIRECT_IDENTIFIER:
        entity = column.entity_type or "IDENTIFIER"
        # A four-digit year is the permitted year representation except when
        # the manifest/header says it is a birth year for someone over 89.
        if entity == "DATE" and _YEAR.fullmatch(stripped):
            return OccurrenceDecision(ACTION_KEEP)
        return OccurrenceDecision(
            ACTION_REPLACE, entity, _CATEGORY_BY_ENTITY.get(entity), deterministic=True
        )
    if (
        column is not None
        and column.role == ROLE_QUASI_IDENTIFIER
        and column.qi_kind in {"institution", "facility"}
        and frozenset(header_tokens(header))
        & {"facility", "institution", "site", "clinic", "hospital", "centre", "center"}
        and is_generic_institution_category(stripped)
    ):
        return OccurrenceDecision(
            ACTION_KEEP, deterministic=True, content_invariant=True
        )
    if column is not None and column.role == ROLE_QUASI_IDENTIFIER:
        # Closed demographic/state vocabularies are category *values*, not
        # personal names. Match the entire value under its explicitly signed
        # QI kind; an unknown value or a name added to one still goes through
        # the model. The QI profile may later generalize these cells.
        from services import csv_sanitizer

        if column.qi_kind in {
            "race_ethnicity",
            "location",
        } and csv_sanitizer.is_category_label(column.qi_kind, stripped):
            return OccurrenceDecision(
                ACTION_KEEP, deterministic=True, content_invariant=True
            )
        if column.qi_kind == "sex" and stripped.casefold() in csv_sanitizer.SEX_CODES:
            return OccurrenceDecision(
                ACTION_KEEP, deterministic=True, content_invariant=True
            )
        if (
            column.qi_kind == "zip"
            and column.explicit
            and re.fullmatch(r"\d{5}(?:-\d{4})?", stripped)
            and frozenset(header_tokens(header))
            & {"zip", "zipcode", "postcode", "postal"}
        ):
            # The signed schema proves this is geography rather than an
            # arbitrary short number. A five-digit ZIP (or ZIP+4) is a
            # sub-state geographic subdivision, a direct-identifier category
            # under both profiles, so this occurrence is replaced
            # deterministically with a typed surrogate. Only this declared
            # column's cell is affected: the same digits as a measurement or in
            # another column are never matched here. Under safe_harbor_plus_qi
            # the QI stage then generalizes or suppresses the written values.
            return OccurrenceDecision(
                ACTION_REPLACE,
                "LOCATION",
                _CATEGORY_BY_ENTITY["LOCATION"],
                deterministic=True,
            )
    if embedded_fixed_identifier_spans(stripped):
        # A mixed narrative is not itself a single replacement value, but it
        # cannot be exempted as clinical text before its exact spans are read.
        return OccurrenceDecision(ACTION_KEEP)
    if (
        column is not None
        and column.role == ROLE_QUASI_IDENTIFIER
        and column.qi_kind == "age"
        and re.fullmatch(r"\d{1,3}", stripped)
    ):
        age = int(stripped)
        if age <= 89:
            # A complete, valid age in a declared age field is not an SSN or
            # telephone number. A mixed value never takes this path.
            return OccurrenceDecision(
                ACTION_KEEP, deterministic=True, content_invariant=True
            )
        if age <= 120:
            return OccurrenceDecision(
                ACTION_REPLACE,
                "AGE_OVER_89",
                _CATEGORY_BY_ENTITY["AGE_OVER_89"],
                deterministic=True,
            )
        return OccurrenceDecision(ACTION_HOLD, reason_code=REASON_UNRESOLVED_QI)
    # Structured clinical columns are exact-preservation surfaces.  Give
    # their own header precedence over labels embedded in adjacent identifier
    # values (for example an MRN beside a creatinine result).
    corroborated_hormone_analyte = (
        column is not None
        and column.role == ROLE_CLINICAL_UTILITY
        and re.fullmatch(r"[a-z]{5,}sterone", stripped.casefold()) is not None
        and _fold(header.replace("_", " ")) == stripped.casefold()
    )
    if (
        column is not None
        and column.role == ROLE_CLINICAL_UTILITY
        and (
            _MEASUREMENT.fullmatch(stripped)
            or corroborated_hormone_analyte
            or is_clinical_invariant(stripped, header, row_context)
        )
    ):
        return OccurrenceDecision(
            ACTION_KEEP, deterministic=True, clinical_invariant=True
        )
    if column is not None and column.role == ROLE_NON_SENSITIVE:
        # A declaration describes the field, not an arbitrary value in it.
        # Unsupported content remains subject to model and artifact scans.
        return OccurrenceDecision(ACTION_KEEP)
    # An adjacent cell can supply a short field label ("Bleep", "MRN"), but
    # an identifier or narrative in that cell is not a label for this one.
    label_neighbours = [
        str(item)
        for item in row_context
        if item is not None
        and not re.search(r"\d", str(item))
        and len(header_tokens(str(item))) <= 4
    ]
    neighbours = " ".join(label_neighbours)
    labelled = f"{header} {neighbours}".strip()
    digits = "".join(character for character in stripped if character.isdigit())
    if digits and _CONTACT_LABEL.search(labelled) and 3 <= len(digits) <= 15:
        entity = (
            "FAX_NUMBER" if re.search(r"\bfax\b", labelled, re.I) else "PHONE_NUMBER"
        )
        return OccurrenceDecision(
            ACTION_REPLACE, entity, _CATEGORY_BY_ENTITY[entity], deterministic=True
        )
    if _RECORD_LABEL.search(labelled) and re.search(r"[A-Za-z0-9]", stripped):
        lowered = labelled.casefold()
        entity = (
            "LICENSE_NUMBER"
            if re.search(
                r"gmc|nmc|hcpc|gdc|gphc|npi|dea|provider|registration|licen[cs]e",
                lowered,
            )
            else (
                "VEHICLE_ID"
                if "vin" in lowered
                else (
                    "DEVICE_ID"
                    if re.search(r"serial|udi", lowered)
                    else (
                        "HEALTH_PLAN_ID"
                        if re.search(r"health plan|member", lowered)
                        else (
                            "ACCOUNT_NUMBER"
                            if "account" in lowered
                            else (
                                "ACCESSION_NUMBER"
                                if re.search(r"accession|specimen", lowered)
                                else "MEDICAL_RECORD_NUMBER"
                            )
                        )
                    )
                )
            )
        )
        return OccurrenceDecision(
            ACTION_REPLACE, entity, _CATEGORY_BY_ENTITY.get(entity), deterministic=True
        )
    if is_clinical_invariant(stripped, header, row_context):
        return OccurrenceDecision(
            ACTION_KEEP, deterministic=True, clinical_invariant=True
        )
    if column is not None and column.role == ROLE_QUASI_IDENTIFIER:
        return OccurrenceDecision(ACTION_GENERALIZE, deterministic=True)
    if column is not None and column.unresolved:
        return OccurrenceDecision(ACTION_HOLD, reason_code=REASON_UNRESOLVED_QI)
    return OccurrenceDecision(ACTION_KEEP)


def confirmed_row_identifier_spans(
    text: str,
    row: Sequence[Any],
    columns: Sequence[ColumnPolicy],
    target_column: int,
) -> Tuple[Tuple[int, int, str], ...]:
    """Exact spans reused from confirmed direct-ID cells in this record only.

    A model may miss an invented or uncommon name even when the same record's
    explicitly typed name cell establishes its identity.  Only literal,
    boundary-delimited occurrences are inherited; a code, measurement or a
    similar-looking value in another record never becomes a row-wide label.
    """
    if not isinstance(text, str) or not text:
        return ()
    candidates: List[Tuple[int, int, int, str]] = []
    for source_column in columns:
        index = source_column.index
        if (
            index == target_column
            or index >= len(row)
            or source_column.role != ROLE_DIRECT_IDENTIFIER
        ):
            continue
        source = "" if row[index] is None else str(row[index]).strip()
        if len(source) < 3 or _TOKEN.fullmatch(source):
            continue
        decided = occurrence_decision(source, column=source_column)
        if decided.action != ACTION_REPLACE or not decided.entity_type:
            continue
        pattern = re.compile(r"(?<!\w)" + re.escape(source) + r"(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), index, decided.entity_type))
    # Longest match wins if two direct fields overlap. The ordinal resolves
    # identical values with different declarations deterministically.
    selected: List[Tuple[int, int, str]] = []
    for start, end, index, entity in sorted(
        candidates, key=lambda item: (-(item[1] - item[0]), item[0], item[2])
    ):
        if not any(
            start < previous_end and previous_start < end
            for previous_start, previous_end, _ in selected
        ):
            selected.append((start, end, entity))
    return tuple(sorted(selected))


def embedded_fixed_identifier_spans(text: str) -> Tuple[Tuple[int, int, str], ...]:
    """Stable identifier syntax inside one narrative, with exact offsets.

    These patterns are independently recognizable without a model or a
    neighbouring cell. They run before any clinical-language exemption.
    """
    if not isinstance(text, str):
        return ()
    candidates: List[Tuple[int, int, str]] = []
    for pattern, entity in (
        (_EMBEDDED_TOKEN, "OTHER_UNIQUE_ID"),
        (_EMBEDDED_EMAIL, "EMAIL_ADDRESS"),
        (_EMBEDDED_SSN, "US_SSN"),
        (_EMBEDDED_URL, "URL"),
        (_EMBEDDED_IPV4, "IP_ADDRESS"),
    ):
        for match in pattern.finditer(text):
            start, end = match.span()
            if entity == "URL":
                while end > start and text[end - 1] in ".)]}":
                    end -= 1
            if entity == "IP_ADDRESS":
                try:
                    ipaddress.ip_address(text[start:end])
                except ValueError:
                    continue
            if start < end:
                candidates.append((start, end, entity))
    selected: List[Tuple[int, int, str]] = []
    for start, end, entity in sorted(
        candidates, key=lambda item: (-(item[1] - item[0]), item[0])
    ):
        if not any(
            start < previous_end and previous_start < end
            for previous_start, previous_end, _ in selected
        ):
            selected.append((start, end, entity))
    return tuple(sorted(selected))


def is_declared_schema_header(value: str, column: Optional[ColumnPolicy]) -> bool:
    """Whether trusted schema provenance may protect a field name as syntax."""
    return bool(
        column is not None
        and column.explicit
        and isinstance(value, str)
        and len(value) <= 128
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z][A-Za-z0-9]*)+", value)
        and not embedded_fixed_identifier_spans(value)
    )


def category_for_entity(entity_type: str) -> Optional[str]:
    return _CATEGORY_BY_ENTITY.get(entity_type)


def dataset_roles(columns: Sequence[ColumnPolicy]):
    """Convert shared column policy to the existing deterministic engine."""
    from services import dataset_privacy

    kind_aliases = {
        "marital_status": dataset_privacy.QI_OCCUPATION,
        "language": dataset_privacy.QI_OCCUPATION,
        "facility": dataset_privacy.QI_INSTITUTION,
        "service_line": dataset_privacy.QI_INSTITUTION,
    }
    quasi = {
        item.index: kind_aliases.get(
            item.qi_kind or "", item.qi_kind or dataset_privacy.QI_LOCATION
        )
        for item in columns
        if item.role == ROLE_QUASI_IDENTIFIER
        or (item.role == ROLE_DIRECT_IDENTIFIER and item.qi_kind is not None)
    }
    sensitive = tuple(
        item.index for item in columns if item.role == ROLE_SENSITIVE_CLINICAL
    )
    ambiguous = tuple(item.index for item in columns if item.unresolved)
    return dataset_privacy.ColumnRoles(quasi, sensitive, ambiguous)


def dataset_policy(config: PolicyConfig):
    from services import dataset_privacy

    return dataset_privacy.PrivacyPolicy(
        k_anonymity=config.k,
        l_diversity=config.l,
        t_closeness=config.t,
        min_rows=config.k,
    )


def policy_summary(
    config: PolicyConfig,
    columns: Sequence[ColumnPolicy],
    *,
    category_counts: Optional[Mapping[str, int]] = None,
    transformations: Optional[Mapping[str, int]] = None,
    dataset_summary: Optional[Mapping[str, Any]] = None,
    unresolved_reasons: Iterable[str] = (),
    candidate_state: str = "pending",
) -> Dict[str, Any]:
    """A value-free privacy report suitable for API responses and evidence."""
    dataset_summary = dataset_summary or {}
    qi_columns = [
        f"column_{item.index}" for item in columns if item.role == ROLE_QUASI_IDENTIFIER
    ]
    unresolved = sorted(set(reason for reason in unresolved_reasons if reason))
    return {
        **config.public_metadata(),
        "legal_characterization": (
            "HIPAA Safe Harbor-oriented tabular de-identification with configurable "
            "quasi-identifier protection"
        ),
        "certification_claimed": False,
        "direct_identifier_categories": dict(sorted((category_counts or {}).items())),
        "quasi_identifier_columns": qi_columns,
        # What actually ran, for the page to compare with what the person chose.
        "schema_declared": bool(config.columns),
        "declared_column_count": len(config.columns),
        "declared_columns_applied": sum(item.explicit for item in columns),
        "inferred_column_count": sum(not item.explicit for item in columns),
        "schema_coverage": (
            "complete"
            if columns and all(item.explicit for item in columns)
            else "partial" if any(item.explicit for item in columns) else "inferred"
        ),
        "full_declared_schema_applied": bool(
            config.columns and columns and all(item.explicit for item in columns)
        ),
        "unresolved_columns": [
            f"column_{item.index}" for item in columns if item.unresolved
        ],
        "column_roles": [item.public_dict() for item in columns],
        "manifest_columns": [
            {
                "index": item.index,
                "header_sha256": item.header_sha256,
                "role": item.role,
                "entity_type": item.entity_type,
                "qi_kind": item.qi_kind,
                "transformation": item.transformation,
            }
            for item in config.columns
        ],
        "transformations": dict(sorted((transformations or {}).items())),
        "minimum_equivalence_class_size": dataset_summary.get("k_achieved"),
        "l_diversity": dataset_summary.get("l_achieved"),
        "t_closeness": (
            dataset_summary.get("t_achieved") if config.t is not None else None
        ),
        "unresolved_risk_reasons": unresolved,
        "candidate_state": candidate_state,
    }


WORKBOOK_SURFACES: Tuple[str, ...] = (
    "visible_cells",
    "hidden_cells",
    "very_hidden_cells",
    "comments",
    "comment_authors",
    "document_properties",
    "custom_properties",
    "defined_names",
    "formula_strings",
    "hyperlinks",
    "headers",
    "footers",
    "validation_messages",
    "sheet_names",
    "shared_strings",
    "relationships",
    "cached_formula_values",
    "images",
    "image_alt_text",
    "drawings",
    "text_boxes",
    "chart_titles",
    "external_links",
    "embedded_objects",
    "unused_package_parts",
)


def safe_harbor_coverage() -> List[Dict[str, Any]]:
    return [
        {
            "number": category.number,
            "slug": category.slug,
            "label": category.label,
            "entity_types": list(category.entity_types),
        }
        for category in SAFE_HARBOR_CATEGORIES
    ]
