import json
import re
from pathlib import Path

ROOT = Path(__file__).parent


def _gold_patterns():
    gold = json.loads((ROOT / "gold.json").read_text(encoding="utf-8"))
    by_value = {}
    for item in gold["planted_values"]:
        by_value[item["value"]] = item["entity_type"]
    # Workbook-only planted surfaces must be part of the detector double too;
    # the cell gold alone does not include comments, headers or properties.
    surface_types = (
        "PERSON",
        "PHONE_NUMBER",
        "PERSON",
        "PERSON",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "PERSON",
        "PHONE_NUMBER",
        "LICENSE_NUMBER",
        "PERSON",
        "MEDICAL_RECORD_NUMBER",
        "PERSON",
        "PERSON",
        "PERSON",
        "URL",
    )
    for value, entity in zip(gold["workbook_surface_values"], surface_types):
        by_value[value] = entity
    by_value["<PATIENT_999>"] = "OTHER_UNIQUE_ID"
    return [
        (re.compile(re.escape(value), re.I), entity)
        for value, entity in sorted(by_value.items(), key=lambda item: -len(item[0]))
    ]


def _mock_anonymizer(text, profile, allocator):
    patterns = _gold_patterns()
    replacements = []
    for pattern, entity in patterns:
        for match in pattern.finditer(text):
            token = allocator.surrogate_for(entity, match.group()) or "<REDACTED>"
            replacements.append((match.start(), match.end(), entity, token))
    accepted = []
    cursor = -1
    for item in sorted(replacements):
        if item[0] >= cursor:
            accepted.append(item)
            cursor = item[1]
    output = text
    for start, end, _entity, token in reversed(accepted):
        output = output[:start] + token + output[end:]
    return (
        {
            "anonymized_text": output,
            "detected_entities": {},
            "residual_phi_categories": {},
            "review_required_reasons": [],
        },
        accepted,
    )


def _clean_scanner(_text, _provenance):
    return []
