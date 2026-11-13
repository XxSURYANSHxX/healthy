"""The one clinical-terminology policy for CSV cells.

A table often carries a clinical code and its display name side by side:
SNOMED, CVX, LOINC or ICD codes with the term each one stands for. General NER
reads those terms without context - "DTaP" as a place, "Influenza" as a person
- and a nine-digit SNOMED code has the shape of an identifier. Both are
clinical content, not PHI. The CSV sanitizer's planning and its own final
verification, and the separate artifact audit, all decide this here and
nowhere else.

A pair is recognised only on evidence taken together, never from a header
alone and never by learning values from the upload:

* a code-like header, and a code-shaped value in every row of that column;
* few distinct codes relative to rows, as a vocabulary has and a column of
  record numbers does not;
* each code with exactly one display, and each display with exactly one code,
  across the whole table;
* every display passing a guard that refuses contact details, digit runs,
  honorifics, capitalised names mid-phrase and narrative sentences.

Codes of a confirmed pair are kept exactly. A display is kept exactly only
when every word of it is a curated terminology word and the residual
detectors' findings on it are the known misreading - a single model's
PERSON, ORGANIZATION or LOCATION label. Two detectors agreeing, or any
deterministic finding, refuses the display. Any value that does not fit
refuses the whole pair, and refused content takes the normal PHI path.

Adding a vocabulary word
------------------------
1. Reproduce the false positive with the pinned models on the real term.
2. Add adversarial cases beside it: a name, a name plus the new word, contact
   details, an address.
3. Show none of them survives or releases.
4. Add the word to ``TERMINOLOGY_WORDS`` and bump ``POLICY_VERSION``.
5. Re-run the real-model CSV validation.

Measurement units
-----------------
A column whose header names a unit ("unit", "units", "uom", "unit of
measure", "result unit") may keep a cell verbatim only when the whole cell is
one curated unit of ``MEASUREMENT_UNITS`` - "g/L", "mmol/L", "mmHg". A
general-purpose model reads a unit as an organization. Anything else in the
column, including a unit followed by a name, takes the normal PHI path.
A column whose header names an analyte or test ("analyte", "test", "lab
test", "component") may likewise keep a cell whose every word is a curated
laboratory analyte word of ``ANALYTE_WORDS`` - "haemoglobin", "serum
creatinine", "white blood cell count". The vocabulary holds no word that is
also a common given name. A name, or an analyte with a name, takes the
normal PHI path.

Version 2 added these two rules; the pair rules are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

#: Changes whenever a rule or the vocabulary changes. The sanitizer declares it
#: in its provenance; the audit accepts a declaration only from this version.
POLICY_VERSION = "csv-terminology-v2"

_CODE_HEADER_WORDS = frozenset(
    {
        "code",
        "codes",
        "snomed",
        "sctid",
        "cvx",
        "loinc",
        "icd",
        "icd9",
        "icd10",
        "cpt",
        "hcpcs",
        "rxnorm",
        "ndc",
        "concept",
    }
)
_DISPLAY_HEADER_WORDS = frozenset({"description", "display", "term", "label"})
_TERMINOLOGY_CODE = re.compile(
    r"\d{1,18}"  # SNOMED CT, CVX, RxNorm
    r"|\d{1,7}-\d"  # LOINC
    r"|[A-Z]\d{2}(?:\.[0-9A-Z]{1,4})?"  # ICD-10
    r"|\d{3}(?:\.\d{1,2})?"  # ICD-9
)
#: Fewer rows than this carry too little evidence to call a column a vocabulary.
MIN_TERMINOLOGY_ROWS = 20
#: A vocabulary repeats; at most one distinct code per this many rows.
TERMINOLOGY_ROWS_PER_CODE = 4
_HONORIFIC_WORD = re.compile(
    r"\b(?:dr|mr|mrs|ms|miss|mx|prof|professor|sir|dame)\b\.?", re.IGNORECASE
)
_MID_PHRASE_CAPITAL = re.compile(r"(?<=\S\s)[A-Z][a-z]{2,}")
_SENTENCE_BREAK = re.compile(r"[.!?;:]\s+\S")

#: The words terminology display names are built from: vaccine, allergy and
#: immunization terms and their standard abbreviations, and the connecting
#: words between them. Curated here, never learned from an upload. A model
#: reading "Allergy to bee venom" as a person and one reading "Alice" as a
#: person give the same evidence, so the words are what tell them apart.
_VOCABULARY_SOURCE = """
    allergy allergic allergies intolerance sensitivity reaction to of and or with without for
    in the an free product products animal dander dust mite mites house pollen tree trees
    grass weed mould mold venom bee wasp insect sting latex shellfish fish nut nuts
    peanut peanuts egg eggs dairy milk wheat gluten soy soya sesame drug penicillin
    sulfonamide aspirin
    vaccine vaccines vaccination immunization toxoid toxoids antigen conjugate polysaccharide
    live attenuated inactivated recombinant injectable intranasal oral seasonal preservative
    adult adults adolescent pediatric ped adol child infant dose doses valent monovalent
    bivalent trivalent quadrivalent hep hepatitis influenza flu pneumococcal meningococcal
    rotavirus varicella zoster measles mumps rubella polio diphtheria tetanus pertussis
    acellular haemophilus hemophilus typhoid rabies cholera yellow fever japanese
    encephalitis anthrax smallpox covid sars coronavirus rsv hpv hib dtap tdap td dt ipv opv
    mmr mmrv pcv ppsv mcv mcv4 menacwy menb bcg prp omp hbv hav
"""
_VOCABULARY_ENTRY = re.compile(r"[a-z][a-z0-9]{1,31}")


def _validated_vocabulary(source: str) -> FrozenSet[str]:
    """The vocabulary, or an import-time failure if any entry is malformed or repeated."""
    words = source.split()
    malformed = [word for word in words if not _VOCABULARY_ENTRY.fullmatch(word)]
    repeated = sorted({word for word in words if words.count(word) > 1})
    if not words or malformed or repeated:
        raise ValueError("invalid_terminology_vocabulary")
    return frozenset(words)


TERMINOLOGY_WORDS: FrozenSet[str] = _validated_vocabulary(_VOCABULARY_SOURCE)

#: Header words that make a column a column of measurement units, and the only
#: other words such a header may carry.
_UNIT_HEADER_WORDS = frozenset({"unit", "units", "uom"})
_UNIT_HEADER_CONTEXT = frozenset(
    {
        "of",
        "measure",
        "measurement",
        "result",
        "dose",
        "lab",
        "test",
        "value",
        "reference",
    }
)
#: Clinical and laboratory units, written with "u" for micro. Compared after
#: stripping and case folding, so "mmol/L" and "MMOL/L" are one entry. No
#: single letter: an initial must never pass as a unit.
_UNIT_SOURCE = """
    g/l g/dl mg/dl mg/l ug/l ng/ml ng/l pg/ml ug/ml mg/ml g/24h mg/24h mg/mmol
    mmol/l umol/l nmol/l pmol/l mol/l meq/l mosm/kg mosm/l
    u/l iu/l miu/l mu/l ku/l iu/ml miu/ml u/ml
    % mmhg cmh2o kpa bpm /min beats/min breaths/min
    degc degf °c °f
    kg mg ug mcg ng pg ml ul dl fl cm mm km in lb lbs oz
    kg/m2 kg/m² mg/kg ug/kg ml/min ml/h ml/kg ml/min/1.73m2 ml/min/1.73m²
    10^9/l 10*9/l x10^9/l x10*9/l 10^12/l 10*12/l x10^12/l x10*12/l 10^3/ul 10^6/ul
    cells/ul copies/ml
    sec min hr hrs hours days weeks months years
    {score} score ratio index
"""
_UNIT_ENTRY = re.compile(r"[a-z0-9%/^*.{}°²]{2,24}|%")


def _validated_units(source: str) -> FrozenSet[str]:
    units = source.split()
    malformed = [unit for unit in units if not _UNIT_ENTRY.fullmatch(unit)]
    repeated = sorted({unit for unit in units if units.count(unit) > 1})
    if not units or malformed or repeated:
        raise ValueError("invalid_measurement_unit_vocabulary")
    return frozenset(units)


MEASUREMENT_UNITS: FrozenSet[str] = _validated_units(_UNIT_SOURCE)


_ANALYTE_HEADER_WORDS = frozenset(
    {"analyte", "analytes", "test", "tests", "component", "observation"}
)
_ANALYTE_HEADER_CONTEXT = frozenset({"lab", "laboratory", "name", "type", "result"})
#: Laboratory analytes and the words their names are built from. Curated,
#: never learned from an upload; no entry is also a common given name.
_ANALYTE_SOURCE = """
    haemoglobin hemoglobin hb hgb haematocrit hematocrit hct glucose sodium potassium
    chloride bicarbonate urea bun creatinine egfr hba1c a1c glycated wbc rbc platelet
    platelets plt cholesterol ldl hdl triglycerides triglyceride alt ast alp ggt bilirubin
    albumin protein crp esr inr aptt troponin bnp lactate ph pco2 po2 calcium magnesium
    phosphate ferritin iron tsh ft4 ft3 t3 t4 b12 folate neutrophils lymphocytes monocytes
    eosinophils basophils mcv mch mchc rdw reticulocytes fibrinogen amylase lipase urate
    uric acid ammonia cortisol insulin psa hcg osmolality
    white red blood cell cells count total free direct indirect serum plasma urine fasting
    random corpuscular volume mean arterial venous capillary level concentration
"""
_ANALYTE_ENTRY = re.compile(r"[a-z][a-z0-9]{1,23}")


def _validated_analytes(source: str) -> FrozenSet[str]:
    words = source.split()
    malformed = [word for word in words if not _ANALYTE_ENTRY.fullmatch(word)]
    repeated = sorted({word for word in words if words.count(word) > 1})
    if not words or malformed or repeated:
        raise ValueError("invalid_analyte_vocabulary")
    return frozenset(words)


ANALYTE_WORDS: FrozenSet[str] = _validated_analytes(_ANALYTE_SOURCE)
_ANALYTE_SPLIT = re.compile(r"[\s,()/-]+")
MAX_ANALYTE_WORDS = 6


def is_analyte_header(tokens) -> bool:
    """Whether a header's words name a column of analytes or tests, and nothing else."""
    words = frozenset(tokens)
    return bool(words & _ANALYTE_HEADER_WORDS) and words <= (
        _ANALYTE_HEADER_WORDS | _ANALYTE_HEADER_CONTEXT
    )


def is_analyte_name(value: str) -> bool:
    """Whether every word of a whole cell is a curated analyte word, and there are few."""
    words = [word for word in _ANALYTE_SPLIT.split(value.strip().casefold()) if word]
    return 0 < len(words) <= MAX_ANALYTE_WORDS and all(
        word in ANALYTE_WORDS for word in words
    )


#: Word endings that name a substance or an enzyme, not a person: an analyte
#: column no list covers still reads as chemistry ("Testosterone",
#: "Transaminase", "Fructosamine"). Deliberately strict: no "-in" or "-ine",
#: which are also given names, and a minimum length so short surnames with a
#: chemical-looking ending are not read as substances.
_CHEMICAL_SUFFIXES = (
    "ase",
    "sterone",
    "olone",
    "dione",
    "isone",
    "globin",
    "emia",
    "uria",
    "ium",
    "ose",
    "amine",
    "ogen",
    "trophin",
    "tropin",
    "peptide",
    "lipid",
    "protein",
    "phate",
    "bonate",
    "oxide",
    "chloride",
    "ol",
    "ate",
)
_MIN_CHEMICAL_WORD = 7
_ANALYTE_CONNECTORS = frozenset(
    {
        "of",
        "in",
        "and",
        "or",
        "the",
        "to",
        "per",
        "ratio",
        "index",
        "binding",
        "globulin",
        "hormone",
        "vitamin",
        "free",
        "total",
        "activity",
        "antibody",
        "antigen",
        "receptor",
    }
)
_ANALYTE_CELL_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'+]*|[A-Z0-9]{1,6}")
_TITLE_OR_AFFIX = re.compile(
    r"(?i)^(?:dr|mr|mrs|ms|miss|mx|prof|professor|doctor|nurse|"
    r"sister|sir|dame|rev|jr|sr|md|phd)\.?$"
)


def is_structural_analyte(value: str) -> bool:
    """A whole analyte-column cell that reads as chemistry by its own shape.

    Only in a column the header names as analytes or tests. Every word must
    be a curated analyte word, an acronym, lower case, a connector, or carry
    a substance ending; a title, a digit run, an address sign or a hyphenated
    capitalised compound (a surname's shape) refuses it, and anything refused
    is left to the detectors and the edit-justification gate.
    """
    stripped = value.strip()
    if not stripped or "@" in stripped or re.search(r"\d{3,}", stripped):
        return False
    if re.search(r"\b[A-Z][a-z'’]+-[A-Z][a-z'’]+\b", stripped):
        return False
    words = [word for word in _ANALYTE_SPLIT.split(stripped) if word]
    if not 0 < len(words) <= MAX_ANALYTE_WORDS:
        return False
    substance = False
    for word in words:
        if not _ANALYTE_CELL_WORD.fullmatch(word) or _TITLE_OR_AFFIX.match(word):
            return False
        folded = word.casefold()
        if folded in ANALYTE_WORDS or folded in _ANALYTE_CONNECTORS:
            substance = substance or folded in ANALYTE_WORDS
            continue
        if word.isupper() and len(word) <= 6:
            continue
        if len(word) >= _MIN_CHEMICAL_WORD and folded.endswith(_CHEMICAL_SUFFIXES):
            substance = True
            continue
        if word.islower():
            continue
        return False
    return substance


def is_analyte_cell(value: str) -> bool:
    """An analyte-column cell kept verbatim: curated, or chemistry by its shape."""
    return is_analyte_name(value) or is_structural_analyte(value)


def is_unit_header(tokens) -> bool:
    """Whether a header's words name a column of units, and nothing else."""
    words = frozenset(tokens)
    return bool(words & _UNIT_HEADER_WORDS) and words <= (
        _UNIT_HEADER_WORDS | _UNIT_HEADER_CONTEXT
    )


def is_measurement_unit(value: str) -> bool:
    """Whether a whole cell is one curated unit, and nothing more."""
    key = value.strip().replace("µ", "u").replace("μ", "u").casefold()
    return key in MEASUREMENT_UNITS


#: Categories a single general-purpose model is known to assign to terminology.
_TERMINOLOGY_MISREAD = frozenset(
    {"PERSON", "ORGANIZATION", "LOCATION", "GPE", "FACILITY", "NRP"}
)


@dataclass(frozen=True)
class TerminologyPair:
    code_column: int
    display_column: int
    codes: FrozenSet[str]
    displays: FrozenSet[str]
    #: Digest of the sorted code/display mapping. Value-free, and changes if
    #: any code or display in the table changes.
    mapping_sha256: str = ""


def terminology_display_guard(value: str) -> bool:
    """Whether a display value could be a terminology term and carries no PHI shape."""
    from services import csv_sanitizer as cs
    from services.residual_identifier_scan import scan as structured_scan

    text = value.strip()
    if not text or len(text) > 80 or len(text.split()) > 10:
        return False
    if (
        cs._longest_digit_run(text) >= 4
        or "@" in text
        or "://" in text
        or ("www." in text.lower())
    ):
        return False
    if _HONORIFIC_WORD.search(text) or _MID_PHRASE_CAPITAL.search(text):
        return False
    if _SENTENCE_BREAK.search(text) or text.endswith("."):
        return False
    return not (cs._whole_cell_structured(text) or structured_scan(text))


def _mapping_digest(pairs: Dict[str, str]) -> str:
    encoded = json.dumps(sorted(pairs.items()), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


#: Distinct codes one candidate column may hold before it can no longer be a
#: vocabulary. Above it the pair fails; the cardinality rule would refuse it on
#: any table short enough to keep it, so this bounds memory, not policy.
MAX_TERMINOLOGY_DISTINCT_CODES = 50_000


class PairEvidence:
    """The evidence for every candidate code/display pair, gathered row by row.

    One pass over the data rows - a list or a stream - decides the same pairs
    ``terminology_pairs`` always did: a code in every row, the cardinality
    limit, a one-to-one mapping and the display guard on every display. State
    is per candidate column and per distinct code, never per row, and a
    candidate is dropped the moment its evidence fails, so a later row that
    contradicts an earlier one always wins.
    """

    def __init__(
        self,
        header: Sequence[str],
        column_classes: Sequence[Any],
        guard_generated: bool = False,
    ):
        from services import csv_sanitizer as cs

        self.header = list(header)
        self.data_rows = 0
        self._guard_generated = guard_generated
        self.code_columns = [
            column
            for column, cell in enumerate(header)
            if column_classes[column].policy == cs.POLICY_TEXT
            and set(cs.header_tokens(cell)) & _CODE_HEADER_WORDS
        ]
        self.display_columns = [
            column
            for column, cell in enumerate(header)
            if column_classes[column].policy == cs.POLICY_TEXT
            and set(cs.header_tokens(cell)) & _DISPLAY_HEADER_WORDS
        ]
        self._code_ok = {column: True for column in self.code_columns}
        self._codes: Dict[int, set] = {column: set() for column in self.code_columns}
        #: (code column, display column) -> code -> display, and display -> code;
        #: None once the pair has failed.
        self._forward: Dict[tuple, Optional[Dict[str, str]]] = {}
        self._backward: Dict[tuple, Optional[Dict[str, str]]] = {}
        for code_column in self.code_columns:
            for display_column in self.display_columns:
                if display_column != code_column:
                    self._forward[(code_column, display_column)] = {}
                    self._backward[(code_column, display_column)] = {}
        self._guard: Dict[str, bool] = {}

    def add(self, row: Sequence[str]) -> None:
        """One data row; blank rows are not data and are skipped."""
        if not any(cell.strip() for cell in row):
            return
        self.data_rows += 1
        for code_column in self.code_columns:
            if not self._code_ok[code_column]:
                continue
            code = row[code_column]
            if not (code and _TERMINOLOGY_CODE.fullmatch(code)):
                self._fail_code_column(code_column)
                continue
            codes = self._codes[code_column]
            codes.add(code)
            if len(codes) > MAX_TERMINOLOGY_DISTINCT_CODES:
                self._fail_code_column(code_column)
                continue
            for display_column in self.display_columns:
                key = (code_column, display_column)
                forward = self._forward.get(key)
                if forward is None:
                    continue
                backward = self._backward[key]
                display = row[display_column]
                if forward.setdefault(code, display) != display or (
                    backward.setdefault(display, code) != code
                ):
                    self._forward[key] = self._backward[key] = None

    def _fail_code_column(self, code_column: int) -> None:
        self._code_ok[code_column] = False
        self._codes[code_column] = set()
        for key in self._forward:
            if key[0] == code_column:
                self._forward[key] = self._backward[key] = None

    def _display_ok(self, display: str) -> bool:
        from services import csv_sanitizer as cs

        if display not in self._guard:
            self._guard[display] = bool(
                (self._guard_generated and cs._GENERATED_ONLY.match(display.strip()))
                or terminology_display_guard(display)
            )
        return self._guard[display]

    def candidate(
        self, code_column: int, display_column: int
    ) -> Optional["TerminologyPair"]:
        """The pair these two columns form, on all the evidence, or None."""
        if not self._code_ok.get(code_column):
            return None
        codes = self._codes[code_column]
        if self.data_rows < MIN_TERMINOLOGY_ROWS or (
            len(codes) * TERMINOLOGY_ROWS_PER_CODE > self.data_rows
        ):
            return None
        forward = self._forward.get((code_column, display_column))
        backward = self._backward.get((code_column, display_column))
        if forward is None or backward is None:
            return None
        if not all(self._display_ok(display) for display in backward):
            return None
        return TerminologyPair(
            code_column,
            display_column,
            frozenset(codes),
            frozenset(backward),
            _mapping_digest(dict(forward)),
        )

    def pairs(self) -> List["TerminologyPair"]:
        """The accepted pairs, first code column first, each column used once."""
        pairs: List[TerminologyPair] = []
        used = set()
        for code_column in self.code_columns:
            for display_column in self.display_columns:
                if display_column == code_column or display_column in used:
                    continue
                pair = self.candidate(code_column, display_column)
                if pair is not None:
                    pairs.append(pair)
                    used.update({code_column, display_column})
                    break
        return pairs


def terminology_pairs(
    rows: Sequence[Sequence[str]], has_header: bool, column_classes: Sequence[Any]
) -> List[TerminologyPair]:
    """Code and display columns that behave as a clinical terminology, on evidence."""
    if not has_header:
        return []
    evidence = PairEvidence(rows[0], column_classes)
    for row in rows[1:]:
        evidence.add(row)
    return evidence.pairs()


def terminology_words_only(value: str) -> bool:
    """Whether every word of a display is a terminology word or a single letter."""
    words = re.findall(r"[^\W\d_]+", value.casefold())
    return bool(words) and all(
        len(word) == 1 or word in TERMINOLOGY_WORDS for word in words
    )


def terminology_display_accepted(
    value: str, findings: Sequence[Dict[str, Any]]
) -> bool:
    """The one rule for keeping a terminology display, in every path that reads one.

    The display is made entirely of terminology words, and what the detectors
    find on it alone is nothing, or only a single model's misreading.
    """
    return terminology_words_only(value) and terminology_findings_explained(findings)


def terminology_findings_explained(findings: Sequence[Dict[str, Any]]) -> bool:
    """Whether a display's blocking findings are only the known single-model misreading."""
    from services.detection_evidence import EVIDENCE_DETERMINISTIC

    blocking = [finding for finding in findings if finding.get("blocking")]
    if not blocking:
        return True
    if any(
        finding.get("evidence_type") == EVIDENCE_DETERMINISTIC for finding in blocking
    ):
        return False
    if any(
        str(finding.get("category") or "").upper() not in _TERMINOLOGY_MISREAD
        for finding in blocking
    ):
        return False
    return len({finding.get("detector") for finding in blocking}) <= 1


# -- provenance: what the sanitizer declares and the audit checks ------------------------------


def pair_provenance(pairs: Sequence[TerminologyPair]) -> Dict[str, Any]:
    """The value-free declaration of the pairs a sanitizer run relied on."""
    return {
        "policy_version": POLICY_VERSION,
        "pairs": [
            {
                "code_column": f"column_{pair.code_column}",
                "display_column": f"column_{pair.display_column}",
                "codes": len(pair.codes),
                "mapping_sha256": pair.mapping_sha256,
            }
            for pair in pairs
        ],
    }


class OutputPairs:
    """The accepted pairs as the written file carries them, gathered row by row."""

    def __init__(self, pairs: Sequence[TerminologyPair]):
        self.pairs = list(pairs)
        self._mappings: List[Dict[str, str]] = [{} for _ in self.pairs]

    def add(self, row: Sequence[str]) -> None:
        if not any(cell.strip() for cell in row):
            return
        for pair, mapping in zip(self.pairs, self._mappings):
            mapping[row[pair.code_column]] = row[pair.display_column]

    def declared(self) -> List[TerminologyPair]:
        return [
            TerminologyPair(
                pair.code_column,
                pair.display_column,
                frozenset(mapping),
                frozenset(mapping.values()),
                _mapping_digest(mapping),
            )
            for pair, mapping in zip(self.pairs, self._mappings)
        ]


def output_pairs(
    rows: Sequence[Sequence[str]], pairs: Sequence[TerminologyPair]
) -> List[TerminologyPair]:
    """The accepted pairs as the written file carries them, with its mapping digest."""
    gathered = OutputPairs(pairs)
    for row in rows[1:]:
        gathered.add(row)
    return gathered.declared()


def parse_declared(declared: Optional[Dict[str, Any]], width: int):
    """The declared (code column, display column, entry) triples, or (None, reason)."""
    if not isinstance(declared, dict):
        return None, "terminology_provenance_missing"
    if declared.get("policy_version") != POLICY_VERSION:
        return None, "terminology_policy_version_mismatch"
    entries = declared.get("pairs")
    if not isinstance(entries, list):
        return None, "terminology_provenance_malformed"
    parsed = []
    for entry in entries:
        try:
            code_column = int(str(entry["code_column"]).removeprefix("column_"))
            display_column = int(str(entry["display_column"]).removeprefix("column_"))
        except (KeyError, TypeError, ValueError):
            return None, "terminology_provenance_malformed"
        if not (0 <= code_column < width and 0 <= display_column < width) or (
            code_column == display_column
        ):
            return None, "terminology_provenance_mismatch"
        parsed.append((code_column, display_column, entry))
    return parsed, None


def check_declared(
    evidence: "PairEvidence", parsed
) -> Tuple[Optional[List[TerminologyPair]], Optional[str]]:
    """Each declared pair re-proven on the evidence of the artifact's own cells."""
    pairs: List[TerminologyPair] = []
    for code_column, display_column, entry in parsed:
        if (
            code_column not in evidence.code_columns
            or display_column not in evidence.display_columns
        ):
            return None, "terminology_provenance_mismatch"
        pair = evidence.candidate(code_column, display_column)
        if (
            pair is None
            or entry.get("codes") != len(pair.codes)
            or entry.get("mapping_sha256") != pair.mapping_sha256
        ):
            return None, "terminology_provenance_mismatch"
        pairs.append(pair)
    return pairs, None


def verified_declared_pairs(
    rows: Sequence[Sequence[str]],
    has_header: bool,
    column_classes: Sequence[Any],
    declared: Optional[Dict[str, Any]],
):
    """The declared pairs, re-proven on an artifact; or (None, reason).

    Nothing declared is taken on trust. Each pair must still show every piece
    of evidence on the artifact's own cells - its header words, a code in
    every row, the cardinality limit, a one-to-one mapping, the display guard
    on every display the sanitizer did not replace - and the mapping digest
    must equal the one declared, so a changed code or display is caught.
    """
    parsed, reason = parse_declared(declared, len(rows[0]) if rows else 0)
    if parsed is None:
        return None, reason
    if not parsed:
        return [], None
    if not has_header:
        return None, "terminology_provenance_mismatch"
    evidence = PairEvidence(rows[0], column_classes, guard_generated=True)
    for row in rows[1:]:
        evidence.add(row)
    return check_declared(evidence, parsed)
