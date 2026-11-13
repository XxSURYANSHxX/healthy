"""The release audit reads written text as written, and still finds what survived.

The 25 KB boundary note was held by an audit that masked the sanitizer's own
tokens before scanning: "by Dr <PROVIDER_1> at the" became "by Dr   at the",
and the title rule took "at the" for a name. The audit now reads exact text
and locates tokens only from the run's authenticated ledger. A scanner holding
only bytes trusts no token shape. Every real name below still blocks.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import release_audit  # noqa: E402
from services.text_anonymization import residual_phi_categories  # noqa: E402


def _categories(text, scanner):
    state = release_audit._Audit(scanner)
    state.scan_text(text)
    return {
        finding.category for finding in state.report(release_audit.FORMAT_TEXT).findings
    }


def test_the_masking_artefact_that_held_the_boundary_note_is_gone():
    text = "Seen today by Dr <PROVIDER_1> at the <FACILITY_1>. Sodium 138 mmol/L."

    # The old path: tokens masked, then scanned. It reads a name into the hole.
    assert "PERSON" in _categories(text, residual_phi_categories)
    assert (
        _categories(
            text, release_audit.artifact_scanner({"<PROVIDER_1>", "<FACILITY_1>"})
        )
        == set()
    )


@pytest.mark.parametrize(
    "text",
    [
        "Seen today by Dr Zelquin Vantreske at the clinic. Sodium 138 mmol/L.",
        "Contact the patient, Jordan Whitfield, about results.",
        # A real name beside a token: the token is discounted, the name is not.
        "Seen today by <PERSON_1> Zelquin Vantreske. Sodium 138 mmol/L.",
        "Reviewed by Dr <PROVIDER_1> and Dr Ysolde Crandmere.",
    ],
)
def test_real_names_still_block_the_exact_text_audit(text):
    assert "PERSON" in _categories(text, release_audit.artifact_scanner())


def test_a_forged_token_shape_carrying_a_name_still_blocks():
    assert _categories(
        "Seen by <PERSON_Zelquin> today.", release_audit.artifact_scanner()
    )


def test_a_token_the_run_did_not_write_is_read_like_any_other_text():
    # With the run's ledger, only tokens it wrote are discounted.
    text = "Seen today by Dr <PROVIDER_1> at the <FACILITY_1>."
    assert (
        _categories(
            text, release_audit.artifact_scanner({"<PROVIDER_1>", "<FACILITY_1>"})
        )
        == set()
    )
    from services.multiformat import generated_token_provenance

    start = text.index("<PROVIDER_1>")
    partial = generated_token_provenance(text, {"<FACILITY_1>"})
    assert partial.covering(start, start + len("<PROVIDER_1>")) is None


@pytest.mark.parametrize(
    "altered",
    [
        "<<PERSON_1>>",
        "prefix<PERSON_1>",
        "<PERSON_1>suffix",
        "<OUTER_<PERSON_1>>",
    ],
)
def test_altered_or_nested_current_job_tokens_receive_no_provenance(altered):
    from services.multiformat import generated_token_provenance

    provenance = generated_token_provenance(altered, {"<PERSON_1>"})
    assert not provenance.regions


@pytest.mark.parametrize(
    "text",
    [
        "History of Parkinson disease and Hodgkin lymphoma; Babinski sign negative.",
        "Started Warfarin and Lasix; Gram stain positive; Apgar 9.",
    ],
)
def test_clinical_eponyms_are_judged_the_same_either_way(text):
    # Eponyms and name-like terms are the detectors' business, not the
    # audit's: the exact-text reading neither suppresses nor adds anything.
    assert _categories(text, release_audit.artifact_scanner()) == _categories(
        text, residual_phi_categories
    )


def test_a_scanner_without_the_marker_still_receives_masked_text():
    seen = []

    def plain(text):
        seen.append(text)
        return []

    release_audit._Audit(plain).scan_text("Seen by Dr <PROVIDER_1> today.")
    assert seen and "<PROVIDER_1>" not in seen[0]

    marked = release_audit.artifact_scanner()
    received = []

    def exact(text):
        received.append(text)
        return marked(text)

    setattr(exact, release_audit.READS_EXACT_TEXT, True)
    release_audit._Audit(exact).scan_text("Seen by Dr <PROVIDER_1> today.")
    assert received == ["Seen by Dr <PROVIDER_1> today."]


def test_a_name_split_across_the_audit_window_edge_is_read_whole(tmp_path):
    filler = "Plan: continue amlodipine, repeat renal function in three months. "
    lead = 4096 + 256 - 5
    text = (
        (filler * (lead // len(filler) + 1))[:lead]
        + " Zelquin Vantreske "
        + filler * 60
    )
    path = tmp_path / "note.txt"
    path.write_text(text, encoding="utf-8")

    report = release_audit.audit_text_file(
        path, text_scanner=release_audit.artifact_scanner(), window=4096, overlap=256
    )

    assert not report.passed
    assert "PERSON" in {finding.category for finding in report.findings}


@pytest.mark.parametrize(
    "token", ["<PERSON_999>", "PERSON_A1B2C3D4", "<REDACTED_DATE>"]
)
def test_a_token_only_artifact_requires_current_run_provenance(token):
    assert _categories(token, release_audit.artifact_scanner())
    assert not _categories(token, release_audit.artifact_scanner({token}))
