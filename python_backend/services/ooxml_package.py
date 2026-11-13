"""What is actually inside an OOXML package, decided by an allowlist.

A spreadsheet is a ZIP of parts joined by relationships, and the grid of
visible cells is one of them. The rest can hold a macro project, an embedded
OLE object, an ActiveX control, a data connection to a database, a link to
another workbook, a custom XML store, or a digital signature over any of it.

The handler that came before this listed the parts it knew were trouble and
let everything else through. That is the wrong way round. A denylist is a
statement about what has been thought of, and an OOXML package is an open
format: the part that matters is the one nobody listed. So this inverts it.
Every part is matched against an allowlist of what a sanitized spreadsheet is
permitted to contain, every relationship against an allowlist of what one is
permitted to reference, and anything unrecognised is a refusal rather than a
shrug.

Three separate questions get three separate answers, because collapsing them
loses the reason:

* is the container safe to open at all - entry count, decompressed size,
  compression ratio, member paths, encryption. That is ``admission``'s
  question and is asked first, before any part is read;
* is every part one we recognise, and does the package declare it as the type
  it is;
* does every relationship point somewhere we allow, and does it stay inside
  the package.

Nothing here rewrites anything. It reports what is in the file so a caller can
decide, and it is used twice: once on the upload, and once on the package the
pipeline produced, where the same allowlist becomes the proof that nothing
unverified was carried across.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Dict, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

from services import admission

#: An XML part larger than this is not read. The container limits already cap
#: the package as a whole; this bounds one part, so a single enormous
#: sharedStrings cannot be expanded in memory during inspection.
MAX_XML_PART_BYTES = 64 * 1024 * 1024
#: Nesting deeper than this is not a document, it is an attack on the parser.
MAX_XML_DEPTH = 64

REASON_CONTAINER = "ooxml_container_unsafe"
REASON_MALFORMED = "ooxml_package_malformed"
REASON_ENCRYPTED = "ooxml_package_encrypted"
REASON_NOT_A_WORKBOOK = "ooxml_not_a_spreadsheet_package"
REASON_UNKNOWN_PART = "ooxml_unknown_part_present"
REASON_ACTIVE_CONTENT = "ooxml_active_content_present"
REASON_MACRO_PROJECT = "ooxml_macro_project_present"
REASON_SIGNATURE = "ooxml_signature_present"
REASON_EXTERNAL_LINK = "ooxml_external_workbook_link_present"
REASON_DATA_CONNECTION = "ooxml_data_connection_present"
REASON_EMBEDDED_OBJECT = "ooxml_embedded_object_present"
REASON_CUSTOM_XML = "ooxml_custom_xml_present"
REASON_UNKNOWN_RELATIONSHIP = "ooxml_unknown_relationship_present"
REASON_CONTENT_TYPE_MISMATCH = "ooxml_content_type_not_declared"

#: Encrypted OOXML is an OLE compound file, not a ZIP. Its signature is the
#: one thing that identifies it before anything tries to unzip it.
_OLE_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

_RELATIONSHIP_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"

#: Parts a sanitized spreadsheet may contain. Anchored patterns, because
#: "starts with xl/worksheets/" would also admit xl/worksheets/../evil.
_ALLOWED_PARTS: Tuple[re.Pattern, ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"^\[Content_Types\]\.xml$",
        r"^_rels/\.rels$",
        r"^docProps/(?:core|app|custom)\.xml$",
        r"^xl/workbook\.xml$",
        r"^xl/_rels/workbook\.xml\.rels$",
        r"^xl/worksheets/sheet\d+\.xml$",
        r"^xl/worksheets/_rels/sheet\d+\.xml\.rels$",
        r"^xl/sharedStrings\.xml$",
        r"^xl/styles\.xml$",
        r"^xl/theme/theme\d+\.xml$",
        r"^xl/calcChain\.xml$",
        # Two spellings for each, because Excel and the writer used here do
        # not agree: Excel produces xl/comments1.xml and vmlDrawing1.vml,
        # openpyxl produces xl/comments/comment1.xml and
        # commentsDrawing1.vml. Both are the same part.
        r"^xl/comments\d*\.xml$",
        r"^xl/comments/comment\d*\.xml$",
        r"^xl/drawings/(?:vmlDrawing|commentsDrawing)\d*\.vml$",
        r"^xl/drawings/_rels/(?:vmlDrawing|commentsDrawing)\d*\.vml\.rels$",
    )
)

#: Parts that are refused by name, each with the reason that names it. Checked
#: before the allowlist so the refusal says what was found rather than only
#: that something was unrecognised.
_NAMED_REFUSALS: Tuple[Tuple[re.Pattern, str], ...] = (
    (re.compile(r"^xl/vbaProject\.bin$", re.IGNORECASE), REASON_MACRO_PROJECT),
    (re.compile(r"^xl/vbaProjectSignature\.bin$", re.IGNORECASE), REASON_MACRO_PROJECT),
    (
        re.compile(r"^xl/(?:macrosheets|dialogsheets)/", re.IGNORECASE),
        REASON_MACRO_PROJECT,
    ),
    (re.compile(r"^xl/(?:activeX|ctrlProps)/", re.IGNORECASE), REASON_ACTIVE_CONTENT),
    (re.compile(r"^xl/embeddings/", re.IGNORECASE), REASON_EMBEDDED_OBJECT),
    (re.compile(r"^xl/externalLinks/", re.IGNORECASE), REASON_EXTERNAL_LINK),
    (
        re.compile(r"^xl/(?:connections\.xml|queryTables/)", re.IGNORECASE),
        REASON_DATA_CONNECTION,
    ),
    (re.compile(r"^customXml/", re.IGNORECASE), REASON_CUSTOM_XML),
    (re.compile(r"^_xmlsignatures/", re.IGNORECASE), REASON_SIGNATURE),
)

#: Relationship types a sanitized spreadsheet may carry.
_ALLOWED_RELATIONSHIPS = frozenset(
    {
        f"{_OFFICE_REL}/officeDocument",
        f"{_OFFICE_REL}/worksheet",
        f"{_OFFICE_REL}/sharedStrings",
        f"{_OFFICE_REL}/styles",
        f"{_OFFICE_REL}/theme",
        f"{_OFFICE_REL}/comments",
        f"{_OFFICE_REL}/vmlDrawing",
        f"{_OFFICE_REL}/calcChain",
        f"{_OFFICE_REL}/extended-properties",
        f"{_OFFICE_REL}/custom-properties",
        f"{_PACKAGE_REL}/metadata/core-properties",
        # A hyperlink is allowed to exist on the way in - the sanitizer removes
        # it - so that an ordinary workbook is not refused for having one. It is
        # not allowed on the way out, and the output check says so.
        f"{_OFFICE_REL}/hyperlink",
    }
)

#: Relationship types refused by name, so the reason is specific.
_REFUSED_RELATIONSHIPS: Dict[str, str] = {
    f"{_OFFICE_REL}/vbaProject": REASON_MACRO_PROJECT,
    f"{_OFFICE_REL}/externalLink": REASON_EXTERNAL_LINK,
    f"{_OFFICE_REL}/externalLinkPath": REASON_EXTERNAL_LINK,
    f"{_OFFICE_REL}/oleObject": REASON_EMBEDDED_OBJECT,
    f"{_OFFICE_REL}/package": REASON_EMBEDDED_OBJECT,
    f"{_OFFICE_REL}/control": REASON_ACTIVE_CONTENT,
    f"{_OFFICE_REL}/customXml": REASON_CUSTOM_XML,
    f"{_OFFICE_REL}/connections": REASON_DATA_CONNECTION,
    f"{_OFFICE_REL}/queryTable": REASON_DATA_CONNECTION,
    f"{_PACKAGE_REL}/digital-signature/signature": REASON_SIGNATURE,
    f"{_PACKAGE_REL}/digital-signature/origin": REASON_SIGNATURE,
}

#: Relationship types allowed only in a package on the way in.
_INPUT_ONLY_RELATIONSHIPS = frozenset({f"{_OFFICE_REL}/hyperlink"})


_SPREADSHEET = "application/vnd.openxmlformats-officedocument.spreadsheetml"
_OFFICEDOC = "application/vnd.openxmlformats-officedocument"
_PACKAGE = "application/vnd.openxmlformats-package"

#: The content type each allowlisted part must be declared as. Checking that
#: a part is "declared" at all is not enough: every ``.xml`` entry inherits
#: the package's Default for that extension, so an injected part arrives
#: already declared as plain XML. What matters is that the package says a
#: part is the thing it is, because a part two readers disagree about is
#: where one that gets executed hides.
_EXPECTED_TYPES: Tuple[Tuple[re.Pattern, Tuple[str, ...]], ...] = (
    (
        re.compile(r"^xl/workbook\.xml$"),
        (
            f"{_SPREADSHEET}.sheet.main+xml",
            f"{_SPREADSHEET}.template.main+xml",
            # A macro-enabled workbook declares itself so. It is recognised here
            # and refused by the caller that asked for a macro-free package, which
            # keeps "what this is" separate from "what we will accept".
            "application/vnd.ms-excel.sheet.macroEnabled.main+xml",
        ),
    ),
    (re.compile(r"^xl/worksheets/sheet\d+\.xml$"), (f"{_SPREADSHEET}.worksheet+xml",)),
    (re.compile(r"^xl/sharedStrings\.xml$"), (f"{_SPREADSHEET}.sharedStrings+xml",)),
    (re.compile(r"^xl/styles\.xml$"), (f"{_SPREADSHEET}.styles+xml",)),
    (re.compile(r"^xl/calcChain\.xml$"), (f"{_SPREADSHEET}.calcChain+xml",)),
    (
        re.compile(r"^xl/(?:comments\d*|comments/comment\d*)\.xml$"),
        (f"{_SPREADSHEET}.comments+xml",),
    ),
    (re.compile(r"^xl/theme/theme\d+\.xml$"), (f"{_OFFICEDOC}.theme+xml",)),
    (re.compile(r"^docProps/core\.xml$"), (f"{_PACKAGE}.core-properties+xml",)),
    (re.compile(r"^docProps/app\.xml$"), (f"{_OFFICEDOC}.extended-properties+xml",)),
    (re.compile(r"^docProps/custom\.xml$"), (f"{_OFFICEDOC}.custom-properties+xml",)),
    (re.compile(r"\.rels$"), (f"{_PACKAGE}.relationships+xml",)),
    (
        re.compile(r"^xl/drawings/(?:vmlDrawing|commentsDrawing)\d*\.vml$"),
        (f"{_OFFICEDOC}.vmlDrawing",),
    ),
)

#: The workbook content type that means macros are in the package.
MACRO_ENABLED_MAIN_TYPE = "application/vnd.ms-excel.sheet.macroEnabled.main+xml"


def expected_content_types(name: str) -> Tuple[str, ...]:
    """What the package must declare this part as, or () if unconstrained."""
    for pattern, types in _EXPECTED_TYPES:
        if pattern.search(name):
            return types
    return ()


@dataclass(frozen=True)
class PackagePart:
    """One entry of the archive, and what the package says it is."""

    name: str
    declared_type: str
    size: int
    compressed_size: int


@dataclass(frozen=True)
class Relationship:
    """One edge of the package graph."""

    source: str
    identifier: str
    type: str
    target: str
    external: bool


@dataclass
class PackageInspection:
    """Everything the package contains, and every reason to refuse it."""

    parts: Tuple[PackagePart, ...] = ()
    relationships: Tuple[Relationship, ...] = ()
    problems: Tuple[str, ...] = ()
    unknown_parts: Tuple[str, ...] = ()
    unknown_relationship_types: Tuple[str, ...] = ()
    external_targets: int = 0
    hyperlinks: int = 0
    undeclared_parts: Tuple[str, ...] = ()
    #: Macro payload found and set aside, when the caller said it intends to
    #: drop it. Empty when the caller did not, because then it is a refusal.
    macro_parts: Tuple[str, ...] = ()
    notes: Dict[str, int] = field(default_factory=dict)

    @property
    def acceptable(self) -> bool:
        return not self.problems

    def safe_summary(self) -> Dict[str, object]:
        """Counts and part names only - never a cell, a string or a target."""
        return {
            "parts": len(self.parts),
            "relationships": len(self.relationships),
            "unknown_parts": list(self.unknown_parts),
            "unknown_relationship_types": list(self.unknown_relationship_types),
            "external_targets": self.external_targets,
            "hyperlinks": self.hyperlinks,
            "macro_parts": list(self.macro_parts),
            "problems": list(self.problems),
        }


def _parse_xml(data: bytes) -> Optional[ElementTree.Element]:
    """Parse one part, refusing anything that is an attack on the parser."""
    if len(data) > MAX_XML_PART_BYTES:
        return None
    lowered = data[:4096].lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        # No spreadsheet part needs a doctype or an entity declaration, and
        # both are how a parser is turned into a file reader or a bomb.
        return None
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError:
        return None
    # Depth is checked after parsing rather than during, because ElementTree
    # gives no hook; the size cap above is what bounds the cost of getting
    # here at all.
    stack = [(root, 0)]
    while stack:
        element, depth = stack.pop()
        if depth > MAX_XML_DEPTH:
            return None
        for child in element:
            stack.append((child, depth + 1))
    return root


def _declared_types(archive: zipfile.ZipFile) -> Dict[str, str]:
    """Part name -> content type, from [Content_Types].xml.

    A package that declares a part as one thing and stores another is not a
    package a reader will agree about, and disagreement between readers is
    where a part that one of them executes hides.
    """
    try:
        data = archive.read("[Content_Types].xml")
    except KeyError:
        return {}
    root = _parse_xml(data)
    if root is None:
        return {}
    defaults: Dict[str, str] = {}
    overrides: Dict[str, str] = {}
    for element in root:
        tag = element.tag.split("}")[-1]
        if tag == "Default":
            extension = (element.get("Extension") or "").lower()
            if extension:
                defaults[extension] = element.get("ContentType") or ""
        elif tag == "Override":
            name = (element.get("PartName") or "").lstrip("/")
            if name:
                overrides[name] = element.get("ContentType") or ""

    declared: Dict[str, str] = {}
    for name in archive.namelist():
        if name.endswith("/"):
            continue
        if name in overrides:
            declared[name] = overrides[name]
            continue
        extension = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        declared[name] = defaults.get(extension, "")
    return declared


def _relationships(archive: zipfile.ZipFile) -> List[Relationship]:
    """Every relationship in every .rels part of the package."""
    found: List[Relationship] = []
    for name in archive.namelist():
        if not name.endswith(".rels"):
            continue
        root = _parse_xml(archive.read(name))
        if root is None:
            continue
        for element in root:
            if element.tag.split("}")[-1] != "Relationship":
                continue
            mode = (element.get("TargetMode") or "Internal").lower()
            found.append(
                Relationship(
                    source=name,
                    identifier=element.get("Id") or "",
                    type=element.get("Type") or "",
                    target=element.get("Target") or "",
                    external=mode == "external",
                )
            )
    return found


#: Parts that are a macro payload and nothing else. They are removable in the
#: strict sense that matters: they are reachable only from the workbook's own
#: vbaProject relationship, no sheet's content depends on them, and dropping
#: them leaves a document that says exactly what it said before. A macrosheet
#: or an ActiveX control is not on this list, because a sheet refers to it and
#: removing it changes what the document is.
_REMOVABLE_MACRO_PARTS: Tuple[re.Pattern, ...] = (
    re.compile(r"^xl/vbaProject\.bin$", re.IGNORECASE),
    re.compile(r"^xl/vbaProjectSignature\.bin$", re.IGNORECASE),
)
_REMOVABLE_MACRO_RELATIONSHIPS = frozenset({f"{_OFFICE_REL}/vbaProject"})


def is_macro_enabled(content: bytes) -> bool:
    """Whether the package declares itself macro-enabled or carries a project.

    Asked of the bytes rather than the filename, because a file named .xlsx
    that contains a macro project is still a file containing a macro project.
    """
    raw = bytes(content or b"")
    try:
        archive = zipfile.ZipFile(BytesIO(raw))
    except (zipfile.BadZipFile, ValueError):
        return False
    try:
        names = archive.namelist()
        if any(
            pattern.match(name) for name in names for pattern in _REMOVABLE_MACRO_PARTS
        ):
            return True
        declared = _declared_types(archive)
        return declared.get("xl/workbook.xml", "") == MACRO_ENABLED_MAIN_TYPE
    finally:
        archive.close()


def inspect(
    content: bytes,
    *,
    output: bool = False,
    allow_macro_project: bool = False,
) -> PackageInspection:
    """Enumerate a package and report every reason it cannot be accepted.

    ``output`` tightens the rules for a package this pipeline produced: a
    hyperlink may arrive in an upload and be removed, but one in the artifact
    means the removal did not happen.

    ``allow_macro_project`` says the caller intends to drop the macro payload
    and rebuild without it, so a vbaProject is counted rather than refused.
    It does not extend to a macrosheet, an ActiveX control or anything else a
    sheet refers to: those cannot be dropped without changing the document,
    and they still refuse the package.
    """
    raw = bytes(content or b"")
    if raw.startswith(_OLE_SIGNATURE):
        return PackageInspection(problems=(REASON_ENCRYPTED,))

    container = admission.container_problems(raw)
    if container:
        return PackageInspection(problems=(REASON_CONTAINER,) + tuple(container))

    try:
        archive = zipfile.ZipFile(BytesIO(raw))
    except (zipfile.BadZipFile, ValueError):
        return PackageInspection(problems=(REASON_MALFORMED,))

    problems: List[str] = []
    unknown: List[str] = []
    undeclared: List[str] = []
    unknown_types: List[str] = []
    macro_parts: List[str] = []
    macro_relationships = 0
    try:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        if "xl/workbook.xml" not in names:
            archive.close()
            return PackageInspection(problems=(REASON_NOT_A_WORKBOOK,))

        declared = _declared_types(archive)
        parts: List[PackagePart] = []
        for info in archive.infolist():
            if info.filename.endswith("/"):
                continue
            name = info.filename
            parts.append(
                PackagePart(
                    name=name,
                    declared_type=declared.get(name, ""),
                    size=info.file_size,
                    compressed_size=info.compress_size,
                )
            )
            if allow_macro_project and any(
                pattern.match(name) for pattern in _REMOVABLE_MACRO_PARTS
            ):
                macro_parts.append(name)
                continue
            refused = next(
                (reason for pattern, reason in _NAMED_REFUSALS if pattern.match(name)),
                None,
            )
            if refused:
                if refused not in problems:
                    problems.append(refused)
                continue
            if not any(pattern.match(name) for pattern in _ALLOWED_PARTS):
                unknown.append(name)
                continue
            expected = expected_content_types(name)
            if expected and declared.get(name, "") not in expected:
                # The package stores this part and calls it something else -
                # or, for an injected ``.xml``, calls it nothing in
                # particular and lets it inherit the Default. Readers
                # disagree about such a part, and disagreement is where one
                # that is executed hides.
                undeclared.append(name)

        relationships = _relationships(archive)
        external = 0
        hyperlinks = 0
        for relationship in relationships:
            removable_macro = relationship.type in _REMOVABLE_MACRO_RELATIONSHIPS
            if allow_macro_project and removable_macro:
                macro_relationships += 1
                continue
            refused = _REFUSED_RELATIONSHIPS.get(relationship.type)
            if refused:
                if refused not in problems:
                    problems.append(refused)
                continue
            if relationship.type not in _ALLOWED_RELATIONSHIPS:
                unknown_types.append(relationship.type)
                continue
            if output and relationship.type in _INPUT_ONLY_RELATIONSHIPS:
                unknown_types.append(relationship.type)
                continue
            if relationship.type == f"{_OFFICE_REL}/hyperlink":
                hyperlinks += 1
            if relationship.external:
                external += 1
                if (
                    REASON_EXTERNAL_LINK not in problems
                    and relationship.type != f"{_OFFICE_REL}/hyperlink"
                ):
                    problems.append(REASON_EXTERNAL_LINK)
            elif admission.unsafe_member_path(relationship.target.lstrip("/")):
                if REASON_MALFORMED not in problems:
                    problems.append(REASON_MALFORMED)
    finally:
        archive.close()

    if unknown and REASON_UNKNOWN_PART not in problems:
        problems.append(REASON_UNKNOWN_PART)
    if undeclared and REASON_CONTENT_TYPE_MISMATCH not in problems:
        problems.append(REASON_CONTENT_TYPE_MISMATCH)
    if unknown_types and REASON_UNKNOWN_RELATIONSHIP not in problems:
        problems.append(REASON_UNKNOWN_RELATIONSHIP)

    return PackageInspection(
        parts=tuple(parts),
        relationships=tuple(relationships),
        problems=tuple(problems),
        unknown_parts=tuple(sorted(set(unknown))),
        unknown_relationship_types=tuple(sorted(set(unknown_types))),
        external_targets=external,
        hyperlinks=hyperlinks,
        undeclared_parts=tuple(sorted(set(undeclared))),
        macro_parts=tuple(sorted(set(macro_parts))),
        notes={"macro_relationships": macro_relationships},
    )


def readable_text(content: bytes, parts: Optional[Sequence[str]] = None) -> str:
    """Everything a package stores as text, for searching afterwards.

    The last check on an artifact is not "does it extract cleanly" but "is the
    value anywhere in the file at all". A cell can be in sharedStrings, in a
    worksheet as an inline string, in a comment, in a defined name, in a
    document property or in a relationship target, and a check that reads only
    the grid is a check that passes by not looking.
    """
    pieces: List[str] = []
    try:
        archive = zipfile.ZipFile(BytesIO(bytes(content or b"")))
    except (zipfile.BadZipFile, ValueError):
        return ""
    try:
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            if parts is not None and name not in parts:
                continue
            try:
                data = archive.read(name)
            except Exception:
                continue
            if len(data) > MAX_XML_PART_BYTES:
                continue
            pieces.append(name)
            pieces.append(data.decode("utf-8", "ignore"))
    finally:
        archive.close()
    return "\n".join(pieces)
