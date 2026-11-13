"""Build the deterministic, wholly synthetic tabular-hipaa18-v1 corpus."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import zipfile
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.chart import BarChart, Reference
from openpyxl.comments import Comment
from openpyxl.drawing.image import Image as SheetImage
from openpyxl.packaging.custom import StringProperty
from openpyxl.workbook.defined_name import DefinedName
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SEED = 184467
ROWS = 60

DIRECT = {
    "patient_name": ("PERSON", "names"),
    "relative_name": ("PERSON", "names"),
    "employer_name": ("PERSON", "names"),
    "street_address": ("ADDRESS", "geographic_subdivisions"),
    "city": ("LOCATION", "geographic_subdivisions"),
    "patient_date": ("DATE", "dates"),
    "phone": ("PHONE_NUMBER", "telephone_numbers"),
    "fax": ("FAX_NUMBER", "fax_numbers"),
    "email": ("EMAIL_ADDRESS", "email_addresses"),
    "ssn": ("US_SSN", "social_security_numbers"),
    "mrn": ("MEDICAL_RECORD_NUMBER", "medical_record_numbers"),
    "health_plan_id": ("HEALTH_PLAN_ID", "health_plan_numbers"),
    "account_number": ("ACCOUNT_NUMBER", "account_numbers"),
    "license_number": ("LICENSE_NUMBER", "certificate_and_license_numbers"),
    "vehicle_id": ("VEHICLE_ID", "vehicle_identifiers"),
    "device_serial": ("DEVICE_ID", "device_identifiers"),
    "url": ("URL", "urls"),
    "ip_address": ("IP_ADDRESS", "ip_addresses"),
    "biometric_id": ("BIOMETRIC_ID", "biometric_identifiers"),
    "image_reference": ("IMAGE_REFERENCE", "full_face_images"),
    "other_unique_code": ("OTHER_UNIQUE_ID", "other_unique_identifiers"),
    "provider_registration": ("LICENSE_NUMBER", "certificate_and_license_numbers"),
}

QUASI = {
    "age": "age",
    "sex": "sex",
    "race": "race_ethnicity",
    "occupation": "occupation",
    "language": "language",
    "region": "location",
    "postal_area": "zip",
    "facility": "facility",
    "service_line": "service_line",
}

SENSITIVE = {"diagnosis", "icd10", "snomed"}
UTILITY = {
    "loinc_code",
    "loinc_description",
    "rxnorm_code",
    "medication",
    "dose",
    "route",
    "frequency",
    "cvx",
    "vaccine",
    "lab_test",
    "lab_value",
    "unit",
    "reference_range",
    "blood_pressure",
    "pathology_stain",
    "imaging_sequence",
    "device_type",
    "follow_up",
}

HEADERS = (
    list(DIRECT)
    + list(QUASI)
    + [
        "rare_combo",
        "diagnosis",
        "icd10",
        "snomed",
        "loinc_code",
        "loinc_description",
        "rxnorm_code",
        "medication",
        "dose",
        "route",
        "frequency",
        "cvx",
        "vaccine",
        "lab_test",
        "lab_value",
        "unit",
        "reference_range",
        "blood_pressure",
        "pathology_stain",
        "imaging_sequence",
        "device_type",
        "follow_up",
        "note",
        "fixture_label",
        "document_version_date",
    ]
)


def _row(index: int) -> list[str]:
    group = index // 5
    diagnoses = ("Type 2 diabetes mellitus", "Hypertension", "Asthma")
    sequences = ("FLAIR", "DESS", "SAG PD FS")
    stains = ("PAS-D stain", "H&E stain", "Trichrome stain")
    values = {
        "patient_name": f"SyntheticPatient-{index:03d}",
        "relative_name": f"SyntheticRelative-{index:03d}",
        "employer_name": f"SyntheticEmployer-{index % 7:02d}",
        "street_address": f"{9000 + index} Synthetic Way",
        "city": f"InventedCity-{index % 6}",
        "patient_date": f"19{60 + index % 30:02d}-{1 + index % 9:02d}-{1 + index % 27:02d}",
        "phone": f"555-{1000 + index:04d}",
        "fax": f"555-{2000 + index:04d}",
        "email": f"synthetic.patient.{index:03d}@example.invalid",
        "ssn": f"9{10 + index % 90:02d}-{10 + index % 89:02d}-{1000 + index:04d}",
        "mrn": f"SYN-MRN-{410000 + index}",
        "health_plan_id": f"SYN-PLAN-{510000 + index}",
        "account_number": f"SYN-ACCT-{610000 + index}",
        "license_number": f"SYN-LIC-{710000 + index}",
        "vehicle_id": f"SYN-VIN-{810000 + index}",
        "device_serial": f"SYN-DEV-{910000 + index}",
        "url": f"https://synthetic-{index:03d}.example.invalid/record",
        "ip_address": f"192.0.2.{1 + index}",
        "biometric_id": f"SYN-FINGERPRINT-{index:03d}",
        "image_reference": f"synthetic-face-{index:03d}.png",
        "other_unique_code": f"SYN-UNIQUE-{index:06d}",
        "provider_registration": f"SYN-REG-{310000 + index}",
        "age": str(24 + group * 3 + index % 2),
        "sex": ("F", "M")[group % 2],
        "race": ("Synthetic Group A", "Synthetic Group B")[group % 2],
        "occupation": f"SyntheticRole-{group % 3}",
        "language": f"Language-{group % 2}",
        "region": f"Region-{group % 3}",
        "postal_area": f"00{group % 3}0{group % 2}",
        "facility": f"SyntheticFacility-{group % 2}",
        "service_line": f"Service-{group % 3}",
        # Deliberately undeclared and unique: safe_harbor_plus_qi must hold it.
        "rare_combo": f"RareSynthetic-{index:03d}",
        "diagnosis": diagnoses[index % len(diagnoses)],
        "icd10": ("E11.9", "I10", "J45.909")[index % 3],
        "snomed": ("44054006", "38341003", "195967001")[index % 3],
        "loinc_code": "4548-4",
        "loinc_description": "Hemoglobin A1c/Hemoglobin.total in Blood; Serum or Plasma",
        "rxnorm_code": "860975",
        "medication": "Metformin",
        "dose": "500 mg",
        "route": "oral",
        "frequency": "twice daily",
        "cvx": "216",
        "vaccine": ("PCV20", "Influenza vaccine")[index % 2],
        "lab_test": "HbA1c",
        "lab_value": "7.2%",
        "unit": ("mU/L", "mOsm/kg")[index % 2],
        "reference_range": "4.0-5.6%",
        "blood_pressure": "128/78 mmHg",
        "pathology_stain": stains[index % 3],
        "imaging_sequence": sequences[index % 3],
        "device_type": "nasogastric tube",
        "follow_up": "Continue treatment and return in 3 months",
        "note": "",
        "fixture_label": "SYNTHETIC",
        "document_version_date": "2026-09-30",
    }
    direct_headers = list(DIRECT)
    first = direct_headers[(2 * index) % len(direct_headers)]
    second = direct_headers[(2 * index + 7) % len(direct_headers)]
    separator = (
        ", Synthetïc multiline\nsecond line: " if index == 0 else "; paired with "
    )
    values["note"] = (
        f"PCV20-Alice synthetic label; {first.upper()}: {values[first]}"
        f"{separator}{second.lower()}: {values[second]}; unit mU/L; <PATIENT_999>"
    )
    return [values[header] for header in HEADERS]


def _policy_manifest() -> dict:
    columns = []
    for index, header in enumerate(HEADERS):
        if header in DIRECT:
            entity, _category = DIRECT[header]
            columns.append(
                {
                    "index": index,
                    "header": header,
                    "role": "direct_identifier",
                    "entity_type": entity,
                }
            )
        elif header in QUASI:
            columns.append(
                {
                    "index": index,
                    "header": header,
                    "role": "quasi_identifier",
                    "qi_kind": QUASI[header],
                }
            )
        elif header in SENSITIVE:
            columns.append(
                {"index": index, "header": header, "role": "sensitive_clinical"}
            )
        elif header in UTILITY:
            columns.append(
                {"index": index, "header": header, "role": "clinical_utility"}
            )
        elif header in {"fixture_label", "document_version_date"}:
            columns.append({"index": index, "header": header, "role": "non_sensitive"})
    return {
        "policy_version": "tabular-privacy-v1",
        "profile": "safe_harbor_plus_qi",
        "k": 5,
        "l": 2,
        "t": None,
        "columns": columns,
    }


def _face_png() -> io.BytesIO:
    stream = io.BytesIO()
    image = Image.new("RGB", (80, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 5, 72, 75), fill="#e9c8a6", outline="#334155", width=2)
    draw.ellipse((25, 28, 31, 34), fill="#1f2937")
    draw.ellipse((49, 28, 55, 34), fill="#1f2937")
    draw.arc((27, 38, 54, 61), 15, 165, fill="#7f1d1d", width=2)
    image.save(stream, format="PNG")
    stream.seek(0)
    return stream


def _write_xlsx(rows: list[list[str]]) -> None:
    workbook = openpyxl.Workbook()
    visible = workbook.active
    visible.title = "Clinical Data"
    visible.append(HEADERS)
    for row in rows:
        visible.append(row)
    # Duplicate headers and leading-zero codes are intentional format tests.
    visible.cell(1, len(HEADERS) + 1, "duplicate")
    visible.cell(1, len(HEADERS) + 2, "duplicate")
    visible.cell(2, len(HEADERS) + 1, "00123")
    visible.cell(2, len(HEADERS) + 2, "00456")
    visible["A2"].comment = Comment("Jordan Fictional; 555-3000", "SyntheticAuthor-000")
    visible["A63"] = '="Jordan Fictional contacted"'
    visible["A64"] = "=[1]External!A1"
    visible["B2"].hyperlink = "https://synthetic-link.example.invalid/record"
    visible.oddHeader.center.text = "SyntheticPatient-001 | 555-3001"
    visible.oddFooter.center.text = "synthetic.patient.001@example.invalid"
    hidden = workbook.create_sheet("Hidden Notes")
    hidden.sheet_state = "hidden"
    hidden.append(["note", "lab_value"])
    hidden.append(["SyntheticPatient-002 called 555-3002", "4417"])
    vault = workbook.create_sheet("Very Hidden")
    vault.sheet_state = "veryHidden"
    vault.append(["provider_registration", "lab_value"])
    vault.append(["SYN-REG-310003", "4417"])
    workbook.properties.creator = "SyntheticPatient-003"
    workbook.properties.title = "SYN-MRN-410003"
    workbook.properties.created = datetime(2000, 1, 1)
    workbook.properties.modified = datetime(2000, 1, 1)
    workbook.custom_doc_props.append(
        StringProperty(name="SyntheticOwner", value="SyntheticPatient-004")
    )
    workbook.defined_names.add(
        DefinedName("SyntheticPatient_005", attr_text="'Clinical Data'!$A$2")
    )
    chart = BarChart()
    chart.title = "SyntheticPatient-006 clinical chart"
    chart.add_data(
        Reference(
            visible, min_col=HEADERS.index("lab_value") + 1, min_row=1, max_row=3
        ),
        titles_from_data=True,
    )
    visible.add_chart(chart, "A66")
    visible.add_image(SheetImage(_face_png()), "D66")
    path = ROOT / "tabular_hipaa18_v1.xlsx"
    workbook.save(path)
    # OOXML ZIP entry timestamps otherwise reflect the build machine clock.
    source = path.read_bytes()
    canonical = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source), "r") as reader, zipfile.ZipFile(
        canonical, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as writer:
        for name in sorted(reader.namelist()):
            info = zipfile.ZipInfo(name, (2000, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            data = reader.read(name)
            if name == "docProps/core.xml":
                data = re.sub(
                    rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)",
                    rb"\g<1>2000-01-01T00:00:00Z\g<2>",
                    data,
                )
            writer.writestr(info, data)
    path.write_bytes(canonical.getvalue())


def build() -> dict:
    random.seed(SEED)
    rows = [_row(index) for index in range(ROWS)]
    with (ROOT / "tabular_hipaa18_v1.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        csv.writer(handle, lineterminator="\n").writerows([HEADERS, *rows])
    _write_xlsx(rows)
    policy = _policy_manifest()
    (ROOT / "quasi_identifier_policy.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )
    categories = sorted({category for _entity, category in DIRECT.values()})
    manifest = {
        "corpus": "tabular_hipaa18_v1",
        "version": 1,
        "seed": SEED,
        "rows": ROWS,
        "default_profile": "safe_harbor_plus_qi",
        "policy_version": "tabular-privacy-v1",
        "synthetic_only": True,
        "safe_harbor_categories": categories,
        "surfaces": {
            "csv": [
                "dedicated_columns",
                "free_text",
                "mixed_content",
                "unicode",
                "quoted_multiline",
            ],
            "xlsx": [
                "visible_sheet",
                "hidden_sheet",
                "very_hidden_sheet",
                "comments",
                "core_properties",
                "custom_properties",
                "defined_names",
                "formula_strings",
                "hyperlinks",
                "headers_footers",
                "chart_title",
                "embedded_synthetic_face_image",
                "external_formula",
                "duplicate_headers",
                "leading_zero_codes",
            ],
            "csv_embedded_image": "not_applicable_for_surface",
            "xlsx_text_box": "not_supported_by_builder",
        },
    }
    planted = []
    for row_index, row in enumerate(rows, start=2):
        note = row[HEADERS.index("note")]
        for header, (entity, category) in DIRECT.items():
            value = row[HEADERS.index(header)]
            planted.append(
                {
                    "surface": "dedicated_cell",
                    "row": row_index,
                    "column": header,
                    "entity_type": entity,
                    "category": category,
                    "value": value,
                }
            )
            if value in note:
                planted.append(
                    {
                        "surface": "mixed_content_cell",
                        "row": row_index,
                        "column": "note",
                        "entity_type": entity,
                        "category": category,
                        "value": value,
                    }
                )
    workbook_surface_values = [
        "Jordan Fictional",
        "555-3000",
        "SyntheticAuthor-000",
        "SyntheticPatient-001",
        "555-3001",
        "synthetic.patient.001@example.invalid",
        "SyntheticPatient-002",
        "555-3002",
        "SYN-REG-310003",
        "SyntheticPatient-003",
        "SYN-MRN-410003",
        "SyntheticPatient-004",
        "SyntheticPatient_005",
        "SyntheticPatient-006",
        "https://synthetic-link.example.invalid/record",
    ]
    gold = {
        "direct_identifier_columns": [
            {"column": header, "entity_type": entity, "category": category}
            for header, (entity, category) in DIRECT.items()
        ],
        "mixed_content_column": "note",
        "planted_values": planted,
        "workbook_surface_values": workbook_surface_values,
        "workbook_surface_identifiers": len(workbook_surface_values),
    }
    invariants = {
        "exact_values": [
            "Type 2 diabetes mellitus",
            "HbA1c",
            "7.2%",
            "128/78 mmHg",
            "Metformin",
            "500 mg",
            "twice daily",
            "PCV20",
            "Influenza vaccine",
            "Serum or Plasma",
            "mU/L",
            "mOsm/kg",
            "PAS-D stain",
            "FLAIR",
            "DESS",
            "SAG PD FS",
            "nasogastric tube",
            "Continue treatment and return in 3 months",
            "SYNTHETIC",
        ],
    }
    states = {
        "csv": {"identifier_removal": "ready", "safe_harbor_plus_qi": "held"},
        "xlsx": {"identifier_removal": "held", "safe_harbor_plus_qi": "held"},
        "notes": {
            "safe_harbor_plus_qi": "undeclared high-uniqueness column",
            "xlsx": "unsupported image/chart/external-link surfaces are stripped and held",
        },
    }
    for name, payload in (
        ("manifest.json", manifest),
        ("gold.json", gold),
        ("clinical_invariants.json", invariants),
        ("expected_states.json", states),
    ):
        (ROOT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digests = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("tabular_hipaa18_v1.csv", "tabular_hipaa18_v1.xlsx")
    }
    (ROOT / "digests.json").write_text(
        json.dumps(digests, indent=2) + "\n", encoding="utf-8"
    )
    return digests


if __name__ == "__main__":
    print(json.dumps(build(), sort_keys=True))
