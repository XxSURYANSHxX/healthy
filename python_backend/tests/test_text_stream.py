"""The bounded-memory text engine: windows, overlap ownership, consistency, verification.

A stand-in anonymizer finds exactly the planted shapes, and only when a whole
value is inside the text it is given - so a value split by a window boundary
is found only if the overlap design gives some window all of it. Every value
is invented.
"""

import io
import os
import re
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stream_fixtures as fixtures  # noqa: E402
from services import request_deadline, text_stream  # noqa: E402
from services.stream_units import UnitBudget  # noqa: E402

WINDOW = 4 * 1024
BUDGET = UnitBudget(text_window_chars=WINDOW, text_overlap_chars=256)
FILLER = "Plan: continue amlodipine, repeat renal function in three months. "


def _run(
    text: str,
    anonymize=fixtures.fake_window_anonymizer,
    residual=fixtures.fake_residual,
    window=WINDOW,
):
    content = text.encode("utf-8")
    sink = io.BytesIO()

    class _Keep(io.BytesIO):
        def close(self):
            sink.write(self.getvalue())
            super().close()

    outcome = text_stream.anonymize_text_stream(
        lambda: io.BytesIO(content),
        len(content),
        _Keep,
        lambda: io.BytesIO(sink.getvalue()),
        budget=BUDGET,
        window_chars=window,
        anonymize=anonymize,
        residual=residual,
    )
    return outcome, sink.getvalue().decode("utf-8")


def _straddle(value: str, offset: int) -> str:
    """Filler up to a window boundary, with ``value`` crossing it ``offset`` characters in."""
    lead = WINDOW - offset - 1
    head = (FILLER * (lead // len(FILLER) + 1))[:lead]
    return head + " " + value + " " + FILLER * 8


@pytest.mark.parametrize(
    "value",
    [
        "Zelquin Vantreske",
        "Dr Zelquin Vantreske",
        "zv.contact@example.invalid",
        "(555) 014-2277",
        "14 Harrow Lane",
        "2019-03-14",
        "MRN-4821337",
        "Ysolde Crandmère",
    ],
)
@pytest.mark.parametrize("offset", [1, 3, 7])
def test_a_value_crossing_a_window_boundary_is_replaced_once(value, offset):
    text = _straddle(value, min(offset, len(value) - 1))

    outcome, output = _run(text)

    assert outcome.windows > 1
    assert value not in output
    assert output.count("<") == 1
    expected = re.sub(re.escape(value), "", text)
    assert re.sub(r"<[A-Z_]+(?:_\d+)?>", "", output) == expected
    assert outcome.handler_result["residual_phi_categories"] == {}


def test_no_character_is_lost_or_duplicated_across_many_windows():
    text = "".join(
        f"{index:06d} {FILLER}" + ("\r\n" if index % 5 == 0 else "")
        for index in range(1_500)
    )
    outcome, output = _run(text)

    assert outcome.windows > 10
    assert output == text


def test_long_unbroken_text_and_a_final_partial_window_are_handled():
    text = "x" * (WINDOW * 3 + 17) + " Zelquin Vantreske"
    outcome, output = _run(text)

    assert output == "x" * (WINDOW * 3 + 17) + " <PERSON_1>"
    assert outcome.windows >= 3


def test_the_same_value_keeps_one_surrogate_however_far_apart():
    text = (
        "Zelquin Vantreske seen. "
        + FILLER * 400
        + "Zelquin Vantreske again. "
        + (FILLER * 400)
        + "Marbrint Morrowgale once."
    )
    _outcome, output = _run(text)

    assert output.count("<PERSON_1>") == 2
    assert "<PERSON_2>" in output


def test_a_value_the_anonymizer_missed_is_caught_by_the_independent_verification():
    def forgetful(text, profile, salt, allocator):
        result, applied = fixtures.fake_window_anonymizer(
            text, profile, salt, allocator
        )
        return {**result, "anonymized_text": text}, []

    outcome, output = _run(_straddle("Zelquin Vantreske", 5), anonymize=forgetful)

    assert "Zelquin Vantreske" in output
    assert outcome.handler_result["residual_phi_categories"].get("PERSON")


_WORDS = set(re.findall(r"[A-Za-z]+", FILLER))


def _fragment_reader(text, provenance=None):
    """Reads any letter run that is not a whole word as a name.

    What the proper-noun rule and the model did with "ssessment" and
    "potassi" at the edge of a raw-offset window on the 25 KB boundary note.
    """
    return [
        {
            "blocking": True,
            "category": "PERSON",
            "detector": "stand_in",
            "evidence_type": "heuristic",
        }
        for word in re.findall(r"[A-Za-z]+", text)
        if word not in _WORDS and not word.isupper()
    ]


def test_verification_windows_never_hand_the_detectors_half_a_word():
    outcome, output = _run(FILLER * 400, residual=_fragment_reader)

    assert outcome.windows > 1
    assert output == FILLER * 400
    assert outcome.handler_result["residual_phi_categories"] == {}


@pytest.mark.parametrize("shift", [-9, -5, -2, 0, 2, 5])
def test_a_name_on_a_verification_window_edge_is_still_read_whole(shift):
    def forgetful(text, profile, salt, allocator):
        result, _applied = fixtures.fake_window_anonymizer(
            text, profile, salt, allocator
        )
        return {**result, "anonymized_text": text}, []

    # The verification window ends at window + overlap; put the name across it.
    edge = WINDOW + BUDGET.text_overlap_chars
    lead = edge + shift - 1
    text = (
        (FILLER * (lead // len(FILLER) + 1))[:lead]
        + " Zelquin Vantreske "
        + FILLER * 120
    )

    outcome, output = _run(text, anonymize=forgetful)

    assert "Zelquin Vantreske" in output
    assert outcome.handler_result["residual_phi_categories"].get("PERSON")


@pytest.mark.parametrize("final", [False, True])
def test_window_bounds_fall_on_boundaries_and_keep_the_overlap(final):
    buffer = FILLER * 50
    end, advance = text_stream.window_bounds(buffer, 1000, 100, final)

    assert buffer[end - 1] in text_stream._BOUNDARY
    assert buffer[advance - 1] in text_stream._BOUNDARY
    assert end - advance >= 100 and 900 <= end <= 1000


def test_window_bounds_fall_back_to_raw_offsets_inside_an_unbroken_token():
    assert text_stream.window_bounds("x" * 5000, 1000, 100, False) == (1000, 900)
    assert text_stream.window_bounds("short", 1000, 100, True) == (5, 5)


def test_the_streamed_release_audit_reads_whole_words_too(tmp_path):
    from services import release_audit

    path = tmp_path / "note.txt"
    path.write_text(FILLER * 400, encoding="utf-8")

    report = release_audit.audit_text_file(
        path,
        text_scanner=lambda text: [f["category"] for f in _fragment_reader(text)],
        window=4096,
        overlap=256,
    )

    assert report.passed, report.findings


def test_a_review_reason_says_where_it_was_raised_and_never_what():
    def uncertain(text, profile, salt, allocator):
        result, applied = fixtures.fake_window_anonymizer(
            text, profile, salt, allocator
        )
        reasons = ["ambiguous_phone_number_requires_review"] if "01103" in text else []
        return {**result, "review_required_reasons": reasons}, applied

    text = FILLER * 90 + "Springfield, MA 01103. " + FILLER * 90
    outcome, _output = _run(text, anonymize=uncertain)

    locations = outcome.handler_result["review_locations"]
    assert outcome.handler_result["review_required_reasons"] == [
        "ambiguous_phone_number_requires_review"
    ]
    assert locations and all(
        set(entry) == {"reason", "pass", "window", "char_start", "char_end"}
        for entry in locations
    )
    at = text.index("01103")
    assert any(entry["char_start"] <= at < entry["char_end"] for entry in locations)
    assert "01103" not in repr(locations)


def test_a_residual_finding_is_located_in_the_verification_pass():
    def forgetful(text, profile, salt, allocator):
        result, _applied = fixtures.fake_window_anonymizer(
            text, profile, salt, allocator
        )
        return {**result, "anonymized_text": text}, []

    text = _straddle("Zelquin Vantreske", 5)
    outcome, _output = _run(text, anonymize=forgetful)

    located = [
        entry
        for entry in outcome.handler_result["review_locations"]
        if entry["pass"] == "verify"
    ]
    at = text.index("Zelquin")
    assert any(
        entry["reason"] == "residual_person"
        and entry["char_start"] <= at < entry["char_end"]
        for entry in located
    )


def test_every_output_byte_is_verified():
    text = FILLER * 1_000
    outcome, _output = _run(text)
    phases = outcome.ledger.summary()["phases"]

    assert phases["verify_text"]["bytes"] == len(text.encode("utf-8"))
    assert phases["anonymize"]["bytes"] == len(text.encode("utf-8"))


def test_a_short_note_is_one_window_as_before():
    outcome, output = _run("Seen by Zelquin Vantreske today.", window=256 * 1024)

    assert outcome.windows == 1
    assert output == "Seen by <PERSON_1> today."
    assert "verify_text" not in outcome.ledger.summary()["phases"]


def test_invalid_utf8_and_nul_bytes_are_refused():
    for content in (b"caf\xe9 note", b"note\x00here"):
        with pytest.raises(text_stream.TextStreamError):
            text_stream.anonymize_text_stream(
                lambda content=content: io.BytesIO(content),
                len(content),
                io.BytesIO,
                io.BytesIO,
                budget=BUDGET,
                anonymize=fixtures.fake_window_anonymizer,
            )


def test_cancellation_between_windows_stops_the_note():
    cancel = threading.Event()
    calls = []

    def anonymizer(text, profile, salt, allocator):
        calls.append(1)
        cancel.set()
        return fixtures.fake_window_anonymizer(text, profile, salt, allocator)

    token = request_deadline.bind_job(cancel)
    request_deadline.start(600)
    try:
        with pytest.raises(request_deadline.RequestCancelled):
            _run(FILLER * 500, anonymize=anonymizer)
    finally:
        request_deadline.clear()
        request_deadline.unbind_job(token)
    assert len(calls) == 1


def test_a_deadline_that_expires_refuses_the_note():
    request_deadline.start(0.000001)
    try:
        with pytest.raises(request_deadline.RequestDeadlineExceeded):
            _run(FILLER * 500)
    finally:
        request_deadline.clear()


def test_a_streamed_note_reports_the_decisions_of_every_window():
    """Occurrence decisions are summed across windows, value-free, and say so."""

    def deciding(text, profile, salt, allocator):
        result, applied = fixtures.fake_window_anonymizer(
            text, profile, salt, allocator
        )
        summary = {
            "policy": "occurrence-decisions-v1",
            "decisions": 1,
            "overflow": 0,
            "by_action": {"hold": 1},
            "by_reason": {"hold:unresolved_short_number": 1},
        }
        return {**result, "occurrence_decisions": summary}, applied

    outcome, _output = _run(FILLER * 500, anonymize=deciding)

    decisions = outcome.handler_result["occurrence_decisions"]
    assert outcome.windows > 1
    assert (
        decisions["windows"] == outcome.windows
        and decisions["counted_per_window"] is True
    )
    assert decisions["by_reason"] == {"hold:unresolved_short_number": outcome.windows}


# -- windows never open or close inside a word ---------------------------------------------------


def _record_windows(text):
    seen = []

    def recording(window_text, *args, **kwargs):
        seen.append(window_text)
        return fixtures.fake_window_anonymizer(window_text, *args, **kwargs)

    outcome, _output = _run(text, anonymize=recording)
    return outcome, seen


def test_no_window_opens_or_closes_inside_a_word():
    # Words of varied length, so that a raw offset would land inside one at either
    # edge of nearly every window. A fragment at an edge ("Nutriti", "rotein") is
    # read by the detectors as a name.
    words = [
        "Nutrition",
        "protein",
        "physiotherapy",
        "assessment",
        "potassium",
        "sagittal",
    ]
    text = " ".join(words[index % len(words)] for index in range(4_000))
    outcome, windows = _record_windows(text)

    assert outcome.windows > 3
    cursor = 0
    for window in windows:
        found = text.find(window, cursor)
        assert found >= 0
        before = text[found - 1] if found else " "
        after = text[found + len(window)] if found + len(window) < len(text) else " "
        assert before in text_stream._BOUNDARY, "a window opened inside a word"
        assert (
            window[-1] in text_stream._BOUNDARY or after in text_stream._BOUNDARY
        ), "a window closed inside a word"
        cursor = found + 1


def test_every_character_is_still_owned_once_with_word_aligned_edges():
    words = ["Nutrition", "protein", "physiotherapy", "assessment"]
    text = " ".join(words[index % len(words)] for index in range(3_000))
    outcome, output = _run(text)
    assert outcome.windows > 3
    assert output == text
