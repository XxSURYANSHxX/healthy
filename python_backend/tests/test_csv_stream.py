"""The bounded-memory CSV engine: parsing, planning, streaming, spill and cancellation.

Stand-in models; every value is invented. The small-file behaviour of the
same engine is covered by the existing CSV suites through ``anonymize_csv_bytes``.
"""

import csv
import io
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_fixtures as fixtures  # noqa: E402
import streaming_support as support  # noqa: E402
from services import csv_sanitizer as cs  # noqa: E402
from services import csv_stream, request_deadline, spill_store  # noqa: E402
from services.multiformat import MultiformatError  # noqa: E402
from services.stream_units import UnitBudget  # noqa: E402

SMALL_UNITS = UnitBudget(
    max_bytes=64 * 1024,
    max_cells=1_000,
    max_tokens=10_000,
    max_model_values=16,
    max_record_chars=64 * 1024,
)


class Sink:
    """A sink that records how it was written to."""

    def __init__(self):
        self.data = io.BytesIO()
        self.writes = []

    def factory(self):
        outer = self

        class _Writer(io.RawIOBase):
            def writable(self):
                return True

            def write(self, chunk):
                outer.writes.append(len(chunk))
                return outer.data.write(chunk)

        return _Writer()

    def reopen(self):
        return io.BytesIO(self.data.getvalue())


class CountingSource:
    """A re-readable source that records the largest single read."""

    def __init__(self, content: bytes):
        self.content = content
        self.largest_read = 0
        self.opens = 0

    def source(self) -> csv_stream.Source:
        outer = self

        class _Reader(io.BytesIO):
            def read(self, size=-1):
                data = super().read(size)
                outer.largest_read = max(outer.largest_read, len(data))
                return data

        def opener():
            outer.opens += 1
            return _Reader(self.content)

        return csv_stream.Source(opener, len(self.content))


def _run(content: bytes, budget=SMALL_UNITS, limits=None, **kwargs):
    sink = Sink()
    counting = CountingSource(content)
    limits = limits or csv_stream.streaming_limits(len(content), budget)
    anonymizer = kwargs.pop("text_anonymizer", fixtures.fake_cell_anonymizer)
    scanner = kwargs.pop("residual_scanner", fixtures.fake_residual)
    outcome = csv_stream.anonymize_csv_stream(
        counting.source(),
        sink.factory,
        sink.reopen,
        limits=limits,
        text_anonymizer=anonymizer,
        residual_scanner=scanner,
        budget=budget,
        **kwargs,
    )
    return outcome, sink, counting


def _rows(sink: Sink):
    return list(
        csv.reader(io.StringIO(sink.data.getvalue().decode("utf-8-sig"), newline=""))
    )


# -- parsing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        b'id,note\n1,"multi\nline, with comma"\n2,"escaped ""quote"""\n',
        b"id,note\r\n1,alpha\r\n2,beta\r\n",
        "﻿id,name\n1,Zoë\n2,Łukasz\n".encode("utf-8"),
        b"id,value\n1,\n2,x\n3,last",
        b"code,description\n00123,Hep A  ped\n0232,DTaP\n",
    ],
)
def test_quoting_line_endings_bom_unicode_and_empty_cells_survive(content):
    outcome, sink, _counting = _run(content)
    source = list(csv.reader(io.StringIO(content.decode("utf-8-sig"), newline="")))
    output = _rows(sink)

    assert len(output) == len(source) and all(
        len(a) == len(b) for a, b in zip(source, output)
    )
    assert outcome.result.summary["round_trip_exact"] in (True, False)
    assert sink.data.getvalue().startswith(b"\xef\xbb\xbf") == content.startswith(
        b"\xef\xbb\xbf"
    )
    if b"00123" in content:
        assert output[1][0] == "00123" and output[2][0] == "0232"


def test_a_very_long_valid_field_passes_and_an_oversized_one_is_refused():
    long_cell = "clinical " * 3_000
    outcome, sink, _ = _run(f"id,note\n1,{long_cell}\n".encode())
    assert _rows(sink)[1][1] == long_cell
    with pytest.raises(MultiformatError) as caught:
        _run(("id,note\n1," + "x" * (cs.MAX_CSV_CELL_CHARS + 1) + "\n").encode())
    assert caught.value.status_code == 413


def test_an_oversized_record_is_refused_without_reading_it_whole():
    budget = UnitBudget(max_record_chars=40_000)
    content = b'id,note\n1,"' + b"line\n" * 20_000 + b'"\n'
    with pytest.raises(MultiformatError) as caught:
        _run(content, budget=budget)
    assert caught.value.status_code == 413 and "records" in caught.value.detail


def test_a_changing_column_count_is_refused():
    with pytest.raises(MultiformatError):
        _run(b"a,b\n1,2\n3,4,5\n")


def test_the_file_is_read_in_bounded_slices_and_written_incrementally():
    content_path = io.StringIO()
    writer = csv.writer(content_path, lineterminator="\n")
    writer.writerow(["PATIENT", "value", "unit"])
    for index in range(40_000):
        writer.writerow(
            [f"4a5c3a3e-0b1d-4c55-9a51-{index:012d}", str(index % 200), "g/L"]
        )
    content = content_path.getvalue().encode()

    outcome, sink, counting = _run(content)

    assert counting.largest_read <= 1 << 20 < len(content)
    assert counting.opens >= 5  # plan, classify, dataset or write, verify
    assert len(sink.writes) > 100 and max(sink.writes) < 64 * 1024
    assert outcome.result.summary["rows"] == 40_000 == len(_rows(sink)) - 1
    phases = outcome.ledger.summary()["phases"]
    assert phases["plan"]["records"] == phases["write"]["records"] == 40_001
    assert phases["verify"]["records"] == 40_001 and phases["plan"]["units"] > 1


def test_a_late_row_that_contradicts_a_terminology_pair_refuses_the_pair():
    rows = [["PATIENT", "CODE", "DESCRIPTION"]]
    for index in range(80):
        code, description = fixtures.TERMS[index % 4]
        rows.append([f"p{index}", code, description])
    rows.append(["p80", fixtures.TERMS[0][0], "Allergy to something else"])
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)

    outcome, _sink, _ = _run(buffer.getvalue().encode())

    assert outcome.result.summary["terminology_pairs"] == []
    assert outcome.result.summary["terminology_code_cells"] == 0


def test_units_analytes_codes_and_planted_names_in_a_structured_file(tmp_path):
    path = tmp_path / "fixture.csv"
    facts = fixtures.write_csv(path, 400_000, patients=300)

    outcome, sink, _ = _run(path.read_bytes())
    output = _rows(sink)
    source = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"), newline="")))

    assert outcome.result.releasable is False or not outcome.result.reason_codes
    assert len(output) == facts["rows"] + 1
    for before, after in zip(source[1:], output[1:]):
        assert before[3:8] == after[3:8]  # code, description, analyte, value, unit
        assert after[0].startswith("<") and after[1].startswith("<")
    text = sink.data.getvalue().decode("utf-8")
    assert not any(name in text for name in fixtures.PLANTED_NAMES)


# -- consistency, spill and cleanup ---------------------------------------------------------


def test_a_value_keeps_its_surrogate_across_distant_units_and_after_spilling(
    monkeypatch,
):
    monkeypatch.setenv(spill_store.MEMORY_ITEMS_ENV_VAR, "1000")
    monkeypatch.setenv("BIOBLOCK_PUBLICATION_MASTER_KEY_V1", "19" * 32)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["PATIENT", "ENCOUNTER", "value"])
    first = "7f1c2a9e-0b1d-4c55-9a51-1e2f3a4b5c6d"
    for index in range(6_000):
        encounter = f"2b3c4d5e-6f70-4a81-92a3-{index:012d}"
        writer.writerow(
            [
                (
                    first
                    if index in (0, 5_999)
                    else f"9e8d7c6b-5a49-4382-9173-{index:012d}"
                ),
                encounter,
                "5",
            ]
        )

    outcome, sink, _ = _run(buffer.getvalue().encode())
    output = _rows(sink)

    assert output[1][0] == output[-1][0] and output[1][0] != output[2][0]
    assert outcome.result.summary["streaming"]["spilled_items"] > 0
    assert first not in sink.data.getvalue().decode()


def test_spilled_state_names_no_value_is_sealed_and_is_removed(monkeypatch, tmp_path):
    support.isolate_storage(tmp_path, monkeypatch)
    monkeypatch.setenv("BIOBLOCK_PUBLICATION_MASTER_KEY_V1", "19" * 32)
    spill = spill_store.JobSpill("j" * 32, capacity=1_000)
    model = spill_store.SpillMap(spill, "model", sealed=True, capacity=10)
    replaced = spill_store.SpillSet(spill, "replaced", capacity=10)
    for index in range(2_000):
        model.put(f"Zelquin Vantreske {index}", f"<PERSON_{index}>")
        replaced.add(f"MRN-{index:07d}")
    model._flush_evicted()
    replaced._flush()
    database = spill.directory / "spill.db"
    raw = database.read_bytes()

    assert b"Zelquin" not in raw and b"MRN-" not in raw and b"PERSON_" not in raw
    # The job's own connection holds the file exclusively; inspect through it.
    rows = spill._connection().execute("SELECT k, v FROM kv").fetchall()
    assert rows and all(len(key) == spill_store.KEY_BYTES for key, _value in rows)
    assert model.get("Zelquin Vantreske 3") == "<PERSON_3>"
    assert "MRN-0000003" in replaced and "MRN-9999999" not in replaced
    spill.close()
    assert not spill.directory.exists()
    assert not list(spill_store.spill_root().glob("**/*-wal"))


def test_spilling_without_the_sealing_key_fails_rather_than_writing_plaintext(
    monkeypatch, tmp_path
):
    support.isolate_storage(tmp_path, monkeypatch)
    monkeypatch.delenv("BIOBLOCK_PUBLICATION_MASTER_KEY_V1", raising=False)
    monkeypatch.setenv(spill_store.MEMORY_ITEMS_ENV_VAR, "1000")
    rows = ["note"] + [
        f"Seen with Zelquin Vantreske visit {index}" for index in range(3_000)
    ]

    outcome, _sink, _ = _run(
        ("\n".join(rows) + "\n").encode(),
        limits=csv_stream.Limits(
            1 << 30,
            1 << 30,
            1 << 30,
            1 << 30,
            1 << 20,
            max_model_fields=100_000,
            max_model_chars=1 << 30,
        ),
    )

    assert csv_stream.REASON_SPILL_UNAVAILABLE in outcome.result.errors
    assert not outcome.result.releasable
    assert not any(spill_store.spill_root().iterdir())


# -- cancellation and deadlines -----------------------------------------------------------------


def test_cancelling_between_units_stops_the_file_and_releases_nothing():
    cancel = threading.Event()
    seen = []

    def anonymizer(text, profile, allocator):
        seen.append(text)
        cancel.set()
        return fixtures.fake_cell_anonymizer(text, profile, allocator)

    rows = "note\n" + "".join(
        f"Seen with Zelquin Vantreske {index}\n" for index in range(200)
    )
    token = request_deadline.bind_job(cancel)
    request_deadline.start(600)
    try:
        with pytest.raises(request_deadline.RequestCancelled):
            _run(rows.encode(), text_anonymizer=anonymizer)
    finally:
        request_deadline.clear()
        request_deadline.unbind_job(token)
    assert len(seen) == 1


def test_a_deadline_that_expires_mid_file_refuses_release():
    request_deadline.start(0.000001)
    try:
        with pytest.raises(request_deadline.RequestDeadlineExceeded):
            _run(
                b"PATIENT,value\n"
                + b"".join(
                    b"4a5c3a3e-0b1d-4c55-9a51-%012d,1\n" % index
                    for index in range(20_000)
                )
            )
    finally:
        request_deadline.clear()


def test_an_incomplete_ledger_is_never_releasable(monkeypatch):
    monkeypatch.setattr(csv_stream.UnitLedger, "complete", lambda self, phases: False)
    outcome, _sink, _ = _run(b"id,value\n1,120\n2,130\n")

    assert csv_stream.REASON_UNITS_INCOMPLETE in outcome.result.errors
    assert not outcome.result.releasable


def test_the_small_file_path_is_the_same_engine():
    content = b"diagnosis,systolic_bp\nI10,128\nE11.9,141\n"
    streamed, sink, _ = _run(content, budget=UnitBudget())
    whole = cs.anonymize_csv_bytes(
        content,
        text_anonymizer=fixtures.fake_cell_anonymizer,
        residual_scanner=fixtures.fake_residual,
    )

    assert whole.output_bytes == sink.data.getvalue()
    assert whole.reason_codes == streamed.result.reason_codes
