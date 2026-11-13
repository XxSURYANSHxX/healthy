"""Dataset-level privacy: k = 5, l = 2, t = 0.2 over quasi-identifiers.

Seeded generators stand in for a property-testing library: every generated
table that is released must verify from its own cells, keep its shape and
every non-quasi-identifier cell, and generalize only to values that contain
the original. Adversarial tables try to single out a row, disclose a
diagnosis for a whole group, or skew a group's diagnoses. Every value is
invented.
"""

import json
import os
import random
import re
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import csv_sanitizer  # noqa: E402
from services import dataset_privacy as privacy  # noqa: E402

HEADERS = ["age", "admission date", "sex", "zip", "diagnosis", "systolic bp"]
AGE, YEAR, SEX, ZIP, DIAGNOSIS, BP = range(6)
DIAGNOSES = ("J45.909", "E11.9", "I10", "N18.3", "F32.A", "K21.9")


def _roles(headers):
    return privacy.classify_columns(
        [csv_sanitizer.header_tokens(header) for header in headers],
        [csv_sanitizer.classify_header(header) for header in headers],
    )


def _run(rows, headers=HEADERS):
    return privacy.anonymize_dataset(
        rows, _roles(headers), is_null=csv_sanitizer._is_null
    )


def _check(rows, headers=HEADERS):
    return privacy.verify_dataset(rows, _roles(headers), is_null=csv_sanitizer._is_null)


def _generated(rng, size=None):
    size = size or rng.randint(5, 240)
    skew = rng.random()
    zones = rng.randint(1, 40)
    rows = []
    for _ in range(size):
        rows.append(
            [
                str(rng.randint(18, 89)) if rng.random() > 0.05 else "",
                (
                    str(rng.randint(2015, 2024))
                    if rng.random() > 0.1
                    else "<REDACTED_DATE>"
                ),
                rng.choice(("F", "M", "f", "")),
                f"<LOCATION_{rng.randint(1, zones)}>",
                DIAGNOSES[0] if rng.random() < skew else rng.choice(DIAGNOSES),
                str(rng.randint(95, 180)),
            ]
        )
    return rows


# -- column roles -----------------------------------------------------------------------------


def test_headers_name_quasi_identifiers_sensitive_columns_and_neither():
    headers = [
        "age",
        "date of birth",
        "sex",
        "gender identity",
        "zip",
        "city",
        "state",
        "race",
        "ethnicity",
        "occupation",
        "hospital",
        "diagnosis",
        "icd10 code",
        "systolic bp",
        "patient name",
        "mrn",
        "note",
    ]

    roles = _roles(headers)

    named = {headers[column]: kind for column, kind in roles.quasi_identifiers.items()}
    assert named == {
        "age": "age",
        "date of birth": "year",
        "sex": "sex",
        "gender identity": "sex",
        "zip": "zip",
        "city": "location",
        "state": "location",
        "race": "race_ethnicity",
        "ethnicity": "race_ethnicity",
        "occupation": "occupation",
        "hospital": "institution",
    }
    assert [headers[column] for column in roles.sensitive] == [
        "diagnosis",
        "icd10 code",
    ]
    assert roles.ambiguous == ()


def test_a_header_naming_two_roles_is_ambiguous_and_held():
    headers = ["sex", "hospital diagnosis", "age"]
    rows = [["F", "I10", str(40 + n)] for n in range(10)]

    outcome = _run(rows, headers)

    assert not outcome.releasable
    assert outcome.reason_codes == [privacy.REASON_AMBIGUOUS]
    # Held for its roles, the table is not rewritten - except that an age is never left exact.
    assert [(row[0], row[1]) for row in outcome.rows] == [
        (row[0], row[1]) for row in rows
    ]
    assert [row[2] for row in outcome.rows] == [f"40{EN_DASH}49"] * 10


# -- holds ------------------------------------------------------------------------------------


def test_a_table_without_quasi_identifiers_is_left_alone_at_any_size():
    rows = [["I10", "128"], ["E11.9", "141"]]

    outcome = _run(rows, ["diagnosis", "systolic bp"])

    assert outcome.releasable and not outcome.applicable
    assert outcome.rows == rows and outcome.generalized == {}


def test_fewer_than_five_rows_with_quasi_identifiers_are_held():
    rows = _generated(random.Random(1), size=4)

    outcome = _run(rows)

    assert outcome.reason_codes == [privacy.REASON_TOO_FEW_ROWS]


@pytest.mark.parametrize("value", ["thirty-four", "-4", "212", "34 years"])
def test_an_unreadable_age_is_ambiguous_and_held(value):
    rows = _generated(random.Random(2), size=20)
    rows[7][AGE] = value

    outcome = _run(rows)

    assert outcome.reason_codes == [privacy.REASON_AMBIGUOUS]


@pytest.mark.parametrize("values", [["I10"], [""]])
def test_a_sensitive_column_that_can_never_reach_l_is_impossible_and_held(values):
    rows = _generated(random.Random(3), size=30)
    for row in rows:
        row[DIAGNOSIS] = values[0]

    outcome = _run(rows)

    assert outcome.reason_codes == [privacy.REASON_IMPOSSIBLE]


# -- properties over generated tables ----------------------------------------------------------


def _band(age_text):
    """Independent of the code under test: the decade an age falls in."""
    age = float(age_text)
    if age >= 90:
        return "90+"
    decade = int(age // 10) * 10
    return f"{decade}–{decade + 9}"


def _contains(original, output, column):
    if output == privacy.SUPPRESSED:
        return True
    if column == AGE:
        # An age is written as its one canonical band or not at all.
        return output == _band(original)
    if column == YEAR:
        low, _, high = output.partition("-")
        return float(low) <= float(original) <= float(high or low)
    return privacy._fold(output) == privacy._fold(original)


def test_every_released_generated_table_verifies_from_its_own_cells():
    rng = random.Random(2024)
    released = 0
    for _ in range(150):
        rows = _generated(rng)
        outcome = _run(rows)
        if not outcome.releasable:
            assert outcome.reason_codes == [privacy.REASON_IMPOSSIBLE]
            assert len({row[DIAGNOSIS] for row in rows if row[DIAGNOSIS]}) < 2
            continue
        released += 1
        check = _check(outcome.rows)
        assert check.satisfied
        assert check.k >= 5 and check.l >= 2 and check.t <= 0.2 + privacy.EPSILON
        assert len(outcome.rows) == len(rows)
        for original, output in zip(rows, outcome.rows):
            assert len(output) == len(original)
            assert (
                output[DIAGNOSIS] == original[DIAGNOSIS] and output[BP] == original[BP]
            )
        for (row, column), original in outcome.generalized.items():
            assert column in (AGE, YEAR, SEX, ZIP)
            assert rows[row][column] == original
            if original and not original.startswith("<"):
                assert _contains(original, outcome.rows[row][column], column)
    assert released > 100


# -- age bands ---------------------------------------------------------------------------------


EN_DASH = "–"
BOUNDARY_AGES = ["0", "9", "10", "19", "20", "79", "80", "89", "90", "94", "102", "130"]
EXPECTED_BANDS = {
    "0": f"0{EN_DASH}9",
    "9": f"0{EN_DASH}9",
    "10": f"10{EN_DASH}19",
    "19": f"10{EN_DASH}19",
    "20": f"20{EN_DASH}29",
    "79": f"70{EN_DASH}79",
    "80": f"80{EN_DASH}89",
    "89": f"80{EN_DASH}89",
    "90": "90+",
    "94": "90+",
    "102": "90+",
    "130": "90+",
}


def _age_rows(ages, copies=5):
    return [
        [age, "2020", "F", "<LOCATION_1>", "I10" if n % 2 else "E11.9", "120"]
        for age in ages
        for n in range(copies)
    ]


def test_every_valid_age_is_written_as_its_one_decade_band():
    rows = _age_rows(BOUNDARY_AGES, copies=6)

    outcome = _run(rows)

    assert outcome.releasable
    for original, output in zip(rows, outcome.rows):
        assert output[AGE] == EXPECTED_BANDS[original[AGE]]
    # Ages 9/10, 19/20, 79/80 and 89/90 fall on different sides of a boundary.
    assert {row[AGE] for row in outcome.rows} == set(EXPECTED_BANDS.values())
    assert _check(outcome.rows).satisfied


def test_a_decimal_age_is_banded_by_its_whole_years():
    rows = _age_rows(["34.5", "39.9", "30"])

    assert {row[AGE] for row in _run(rows).rows} == {f"30{EN_DASH}39"}


def test_ages_in_a_band_that_cannot_reach_k_are_suppressed_not_widened():
    # Three adjacent decades with four people each: no band reaches k = 5, and
    # merging them would write a range that is not a decade band.
    rows = _age_rows(["21", "31", "41"], copies=4)
    rows += _age_rows(["55"], copies=8)

    outcome = _run(rows)

    written = {row[AGE] for row in outcome.rows}
    assert written <= {privacy.SUPPRESSED, f"50{EN_DASH}59"}
    assert all(
        row[AGE] == privacy.SUPPRESSED
        for row, source in zip(outcome.rows, rows)
        if source[AGE] in {"21", "31", "41"}
    )
    assert _check(outcome.rows).satisfied


def test_blank_and_generated_ages_are_kept_and_never_banded():
    rows = (
        _age_rows(["45"], copies=8)
        + _age_rows([""], copies=6)
        + _age_rows(["<REDACTED_AGE>"], copies=6)
    )

    outcome = _run(rows)

    assert outcome.releasable
    assert {row[AGE] for row in outcome.rows} == {
        f"40{EN_DASH}49",
        "",
        "<REDACTED_AGE>",
    }


@pytest.mark.parametrize(
    "value", ["34 years", "34y", "about 40", "+34", "1e2", "131", "-1"]
)
def test_an_age_that_is_not_a_plain_number_is_held_not_cast(value):
    rows = _age_rows(["45"], copies=20)
    rows[3][AGE] = value

    assert _run(rows).reason_codes == [privacy.REASON_AMBIGUOUS]


def test_only_the_age_column_is_written_as_a_band():
    rows = _age_rows(["45", "52"], copies=10)
    rows = [row[:SEX] + ["F", "<LOCATION_1>", row[DIAGNOSIS], "45"] for row in rows]

    outcome = _run(rows)

    assert all(row[BP] == "45" for row in outcome.rows)  # a value equal to an age
    assert all(row[DIAGNOSIS] in DIAGNOSES for row in outcome.rows)
    assert {row[AGE] for row in outcome.rows} == {f"40{EN_DASH}49", f"50{EN_DASH}59"}


def test_the_canonical_form_is_stable_and_not_rebanded():
    for band in (f"30{EN_DASH}39", "90+"):
        assert privacy.age_band(band) == band
    assert privacy.age_band(f"35{EN_DASH}44") is None  # not a decade band
    assert privacy.age_band("30-39") is None  # a hyphen is not the form
    assert privacy.age_band("34") == f"30{EN_DASH}39"


@pytest.mark.parametrize(
    "reason", ["roles", "too_few_rows", "unreadable_age", "impossible"]
)
def test_a_held_table_still_writes_every_valid_age_as_a_band(reason):
    rows = _age_rows(["34", "47", "52", "91"], copies=5)
    headers = HEADERS
    if reason == "roles":
        headers = ["age", "hospital diagnosis", "sex"]
        rows = [[row[AGE], "I10", "F"] for row in rows]
    elif reason == "too_few_rows":
        rows = rows[:3]
    elif reason == "unreadable_age":
        rows[1][AGE] = "34 years"
    else:
        rows = [row[:DIAGNOSIS] + ["I10"] + row[DIAGNOSIS + 1 :] for row in rows]

    outcome = _run(rows, headers)

    assert not outcome.releasable
    for original, written in zip(rows, outcome.rows):
        if original[AGE] == "34 years":
            assert (
                written[AGE] == "34 years"
            ), "an age that is not a number is never cast"
        else:
            assert written[AGE] == _band(original[AGE]), "no exact age remains"
    assert all(
        outcome.rows[index][AGE] != rows[index][AGE]
        for (index, column) in outcome.generalized
        if column == AGE
    )


def test_a_table_held_for_an_unresolved_column_is_still_generalized_over_the_known_ones():
    rng = random.Random(77)
    rows = [
        [
            str(rng.randint(30, 79)),
            "2020",
            rng.choice("FM"),
            f"<LOCATION_{rng.randint(1, 3)}>",
            "I10" if n % 2 else "E11.9",
            "120",
            f"u{n}",
        ]
        for n in range(90)
    ]
    headers = [*HEADERS, "note"]
    roles = _roles(headers)
    held = privacy.ColumnRoles(roles.quasi_identifiers, roles.sensitive, (6,))

    outcome = privacy.anonymize_dataset(rows, held, is_null=csv_sanitizer._is_null)

    assert outcome.reason_codes == [privacy.REASON_AMBIGUOUS], "still held"
    assert (
        _check(outcome.rows, headers).k >= 5
    ), "but no group of the known columns is below k"
    assert all(
        row[AGE] in {"*", *(f"{d}0{EN_DASH}{d}9" for d in range(3, 8))}
        for row in outcome.rows
    )
    assert [row[6] for row in outcome.rows] == [
        row[6] for row in rows
    ], "the unresolved column is not edited"


def test_generalization_is_deterministic():
    rows = _generated(random.Random(99), size=180)

    assert _run(rows).rows == _run([list(row) for row in rows]).rows


def test_the_summary_reports_counts_and_achieved_values_only():
    rows = _generated(random.Random(5), size=60)

    summary = _run(rows).summary

    serialized = json.dumps(summary)
    assert not re.search(r"LOCATION_|J45|E11|<", serialized)
    assert summary["k_achieved"] >= 5
    assert summary["dataset_privacy_policy"] == privacy.POLICY_VERSION


# -- adversarial tables -----------------------------------------------------------------------


def _even_diagnoses(rows):
    for index, row in enumerate(rows):
        row[DIAGNOSIS] = DIAGNOSES[index % 2]
    return rows


def test_a_unique_outlier_is_never_left_alone():
    rows = _even_diagnoses(
        [[str(40 + n % 10), "2021", "F", "<LOCATION_1>", "", "120"] for n in range(29)]
    )
    rows.append(["17", "2021", "F", "<LOCATION_1>", "E11.9", "120"])

    outcome = _run(rows)

    assert outcome.releasable
    teen = outcome.rows[-1]
    peers = [row for row in outcome.rows if row[:4] == teen[:4]]
    assert len(peers) >= 5
    assert all(row[AGE] != "17" for row in outcome.rows)


def test_a_group_sharing_one_diagnosis_is_merged_until_it_is_diverse():
    rows = [["30", "2020", "M", "<LOCATION_2>", "B20", "118"] for _ in range(5)]
    rows += _even_diagnoses(
        [[str(50 + n), "2020", "F", "<LOCATION_3>", "", "130"] for n in range(25)]
    )

    outcome = _run(rows)

    assert outcome.releasable
    group = [row for row in outcome.rows if row[:4] == outcome.rows[0][:4]]
    assert len({row[DIAGNOSIS] for row in group}) >= 2


def test_a_group_with_skewed_diagnoses_stays_within_t():
    rows = [
        [str(20 + n % 10), "2019", "F", "<LOCATION_4>", "E11.9", "125"]
        for n in range(10)
    ]
    rows += [
        [
            str(60 + n % 10),
            "2019",
            "F",
            "<LOCATION_4>",
            ("E11.9", "I10", "I10")[n % 3],
            "140",
        ]
        for n in range(30)
    ]

    outcome = _run(rows)

    assert outcome.releasable
    diagnoses = [row[DIAGNOSIS] for row in rows]
    group = [row[DIAGNOSIS] for row in outcome.rows if row[:4] == outcome.rows[0][:4]]
    assert privacy.distribution_distance(diagnoses, group) <= 0.2 + privacy.EPSILON


def test_every_row_in_its_own_area_is_still_grouped_by_five():
    rng = random.Random(8)
    rows = [
        [
            str(rng.randint(20, 80)),
            "2022",
            rng.choice("FM"),
            f"<LOCATION_{n}>",
            DIAGNOSES[n % 3],
            "120",
        ]
        for n in range(100)
    ]

    outcome = _run(rows)

    assert outcome.releasable and _check(outcome.rows).k >= 5


def test_a_table_that_already_meets_the_policy_changes_only_its_ages_to_bands():
    rows = _even_diagnoses(
        [["45", "2021", "F", "<LOCATION_1>", "", "120"] for _ in range(12)]
    )

    outcome = _run(rows)

    assert outcome.releasable
    assert set(outcome.generalized) == {(index, AGE) for index in range(12)}
    assert {row[AGE] for row in outcome.rows} == {f"40{EN_DASH}49"}
    assert [row[:AGE] + row[AGE + 1 :] for row in outcome.rows] == [
        row[:AGE] + row[AGE + 1 :] for row in rows
    ]
    assert outcome.summary["k_achieved"] == 12


# -- verification and distances ------------------------------------------------------------------


def test_verification_catches_a_class_below_k_l_or_t():
    base = _even_diagnoses(
        [["45", "2021", "F", "<LOCATION_1>", "", "120"] for _ in range(10)]
    )
    small = [list(row) for row in base]
    for row in small[:4]:
        row[AGE] = "46"
    uniform = [list(row) for row in base]
    for row in uniform[:5]:
        row[AGE], row[DIAGNOSIS] = "46", "J45.909"
    # A class of five that is 80% J45.909 in a table that is 45% J45.909.
    skewed = [list(row) for row in base] + [list(row) for row in base]
    for index, row in enumerate(skewed):
        row[AGE] = "46" if index < 5 else "45"
        row[DIAGNOSIS] = "E11.9" if index == 4 or index >= 10 else "J45.909"

    assert _check(base).satisfied
    assert _check(small).k == 4 and not _check(small).satisfied
    assert _check(uniform).l == 1 and not _check(uniform).satisfied
    assert _check(skewed).t > 0.2 and not _check(skewed).satisfied


def test_distances_match_their_definitions():
    assert privacy.distribution_distance(["A", "B"] * 5, ["A"] * 5) == pytest.approx(
        0.5
    )
    assert privacy.distribution_distance(["A", "B", "C"] * 4, ["A", "B", "C"]) == 0.0
    # Ordered: values 1, 2, 3 uniform; a sample of only 1s is 0.5 away.
    assert privacy.distribution_distance(["1", "2", "3"], ["1", "1"]) == pytest.approx(
        0.5
    )
    assert privacy.distribution_distance(["1", "2", "3"], ["3"]) == pytest.approx(0.5)
    assert privacy.distribution_distance(["1", "2", "3"], ["2"]) == pytest.approx(1 / 3)
