"""A spreadsheet is a ZIP of parts, and the grid is only one of them.

The rest can hold a macro project, an embedded OLE object, an ActiveX
control, a connection to a database, a link to another workbook, a custom XML
store, or a signature over any of it. The handler that came before listed the
parts it knew were trouble and let everything else through - a statement about
what had been thought of, not about what is in the file.

These tests are written against the inversion: an allowlist, where the part
nobody listed is refused rather than admitted. Each one names a specific way a
package can carry something, and asserts both that it is caught and that the
reason says which thing it was - a refusal that says only "unsupported" cannot
be acted on.
"""

from __future__ import annotations

import io
import zipfile

import openpyxl
import pytest
from openpyxl.comments import Comment
from services import ooxml_package as oo


def _workbook(with_comment=False, with_hyperlink=False, hidden=False) -> bytes:
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Results"
    sheet["A1"] = "Analyte"
    sheet["B1"] = "haemoglobin"
    sheet["A2"] = "Value"
    sheet["B2"] = 96
    if with_comment:
        sheet["B1"].comment = Comment("checked by the lab", "Nadia Oyelowo")
    if with_hyperlink:
        sheet["A1"].hyperlink = "https://example.invalid/record"
    if hidden:
        book.create_sheet("Working")
        book["Working"].sheet_state = "hidden"
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def _with_parts(package: bytes, extra: dict) -> bytes:
    """The same package with additional entries, leaving the rest alone."""
    source = zipfile.ZipFile(io.BytesIO(package))
    out = io.BytesIO()
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target:
            for name in source.namelist():
                target.writestr(name, source.read(name))
            for name, content in extra.items():
                target.writestr(name, content)
    finally:
        source.close()
    return out.getvalue()


def _with_relationship(
    package: bytes, rel_type: str, target: str, external: bool = False
) -> bytes:
    """The same package with one extra relationship on the workbook."""
    source = zipfile.ZipFile(io.BytesIO(package))
    out = io.BytesIO()
    mode = ' TargetMode="External"' if external else ""
    addition = (
        f'<Relationship Id="rIdTest" Type="{rel_type}" ' f'Target="{target}"{mode}/>'
    ).encode("utf-8")
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as write:
            for name in source.namelist():
                data = source.read(name)
                if name == "xl/_rels/workbook.xml.rels":
                    data = data.replace(
                        b"</Relationships>", addition + b"</Relationships>"
                    )
                write.writestr(name, data)
    finally:
        source.close()
    return out.getvalue()


# -- an ordinary workbook is accepted ------------------------------------------------------


def test_a_plain_workbook_is_accepted():
    inspection = oo.inspect(_workbook())
    assert inspection.acceptable, inspection.problems
    assert inspection.unknown_parts == ()


def test_a_workbook_with_comments_and_hidden_sheets_is_accepted():
    """Both are ordinary. Refusing them would make the profile useless."""
    inspection = oo.inspect(_workbook(with_comment=True, hidden=True))
    assert inspection.acceptable, inspection.problems


def test_a_hyperlink_is_allowed_in_but_not_out():
    """It arrives in ordinary workbooks and the sanitizer removes it. One in
    the artifact means the removal did not happen."""
    package = _workbook(with_hyperlink=True)
    assert oo.inspect(package).acceptable
    assert not oo.inspect(package, output=True).acceptable


def test_every_part_and_relationship_is_enumerated():
    inspection = oo.inspect(_workbook(with_comment=True))
    names = {part.name for part in inspection.parts}
    assert "xl/workbook.xml" in names
    assert "[Content_Types].xml" in names
    assert inspection.relationships, "no relationships were read"
    assert any(r.type.endswith("/officeDocument") for r in inspection.relationships)


# -- active and executable content ---------------------------------------------------------


@pytest.mark.parametrize(
    "part,reason",
    [
        ("xl/vbaProject.bin", oo.REASON_MACRO_PROJECT),
        ("xl/vbaProjectSignature.bin", oo.REASON_MACRO_PROJECT),
        ("xl/macrosheets/sheet1.xml", oo.REASON_MACRO_PROJECT),
        ("xl/dialogsheets/sheet1.xml", oo.REASON_MACRO_PROJECT),
        ("xl/activeX/activeX1.xml", oo.REASON_ACTIVE_CONTENT),
        ("xl/ctrlProps/ctrlProp1.xml", oo.REASON_ACTIVE_CONTENT),
        ("xl/embeddings/oleObject1.bin", oo.REASON_EMBEDDED_OBJECT),
        ("xl/externalLinks/externalLink1.xml", oo.REASON_EXTERNAL_LINK),
        ("xl/connections.xml", oo.REASON_DATA_CONNECTION),
        ("xl/queryTables/queryTable1.xml", oo.REASON_DATA_CONNECTION),
        ("customXml/item1.xml", oo.REASON_CUSTOM_XML),
        ("_xmlsignatures/sig1.xml", oo.REASON_SIGNATURE),
    ],
)
def test_a_named_dangerous_part_is_refused_by_name(part, reason):
    inspection = oo.inspect(_with_parts(_workbook(), {part: b"<x/>"}))
    assert reason in inspection.problems, inspection.problems


def test_a_part_nobody_listed_is_still_refused():
    """The whole point of an allowlist: the part that matters is the one no
    denylist anticipated."""
    inspection = oo.inspect(
        _with_parts(_workbook(), {"xl/somethingEntirelyNew.xml": b"<x/>"})
    )
    assert oo.REASON_UNKNOWN_PART in inspection.problems
    assert "xl/somethingEntirelyNew.xml" in inspection.unknown_parts


@pytest.mark.parametrize(
    "part",
    [
        "xl/media/image1.png",
        "xl/charts/chart1.xml",
        "xl/pivotTables/pivotTable1.xml",
        "xl/slicers/slicer1.xml",
        "xl/threadedComments/threadedComment1.xml",
        "xl/persons/person.xml",
        "xl/model/item.data",
        "xl/printerSettings/printerSettings1.bin",
    ],
)
def test_parts_outside_the_profile_are_refused(part):
    inspection = oo.inspect(_with_parts(_workbook(), {part: b"data"}))
    assert not inspection.acceptable
    assert part in inspection.unknown_parts


# -- relationships -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel,reason",
    [
        (f"{oo._OFFICE_REL}/vbaProject", oo.REASON_MACRO_PROJECT),
        (f"{oo._OFFICE_REL}/externalLink", oo.REASON_EXTERNAL_LINK),
        (f"{oo._OFFICE_REL}/oleObject", oo.REASON_EMBEDDED_OBJECT),
        (f"{oo._OFFICE_REL}/package", oo.REASON_EMBEDDED_OBJECT),
        (f"{oo._OFFICE_REL}/control", oo.REASON_ACTIVE_CONTENT),
        (f"{oo._OFFICE_REL}/customXml", oo.REASON_CUSTOM_XML),
        (f"{oo._OFFICE_REL}/connections", oo.REASON_DATA_CONNECTION),
        (f"{oo._PACKAGE_REL}/digital-signature/signature", oo.REASON_SIGNATURE),
    ],
)
def test_a_dangerous_relationship_is_refused_by_type(rel, reason):
    package = _with_relationship(_workbook(), rel, "somewhere.bin")
    assert reason in oo.inspect(package).problems


def test_a_relationship_type_nobody_listed_is_refused():
    package = _with_relationship(_workbook(), "http://example.invalid/rel/new", "x.xml")
    inspection = oo.inspect(package)
    assert oo.REASON_UNKNOWN_RELATIONSHIP in inspection.problems
    assert "http://example.invalid/rel/new" in inspection.unknown_relationship_types


def test_an_external_target_is_refused():
    package = _with_relationship(
        _workbook(),
        f"{oo._OFFICE_REL}/worksheet",
        "file:///c:/records/patients.xlsx",
        external=True,
    )
    assert oo.REASON_EXTERNAL_LINK in oo.inspect(package).problems


def test_a_relationship_pointing_outside_the_package_is_malformed():
    package = _with_relationship(
        _workbook(),
        f"{oo._OFFICE_REL}/worksheet",
        "../../../etc/passwd",
    )
    assert oo.REASON_MALFORMED in oo.inspect(package).problems


# -- the manifest has to agree with the archive ---------------------------------------------


def test_a_part_declared_as_something_else_is_refused():
    """Every .xml entry inherits the package Default, so an injected part
    arrives already "declared". What matters is that the package says a part
    is the thing it is - a part two readers disagree about is where one that
    gets executed hides."""
    package = _with_parts(_workbook(), {"xl/comments1.xml": b"<comments/>"})
    inspection = oo.inspect(package)
    assert oo.REASON_CONTENT_TYPE_MISMATCH in inspection.problems
    assert "xl/comments1.xml" in inspection.undeclared_parts


def test_the_expected_type_of_a_worksheet_is_specific():
    assert oo.expected_content_types("xl/worksheets/sheet1.xml") == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
    )
    assert oo.expected_content_types("xl/somethingNew.xml") == ()


def test_a_macro_enabled_workbook_declares_itself():
    """Recognising what a package is stays separate from deciding to accept
    it, so the caller that wants a macro-free artifact can say so itself."""
    assert oo.MACRO_ENABLED_MAIN_TYPE in oo.expected_content_types("xl/workbook.xml")


# -- the container, before any part is read -------------------------------------------------


def test_an_encrypted_package_is_recognised_without_unzipping():
    """Encrypted OOXML is an OLE compound file, not a ZIP at all."""
    inspection = oo.inspect(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
    assert inspection.problems == (oo.REASON_ENCRYPTED,)


def test_something_that_is_not_a_zip_is_refused():
    assert oo.REASON_CONTAINER in oo.inspect(b"this is not a package").problems


def test_a_zip_that_is_not_a_spreadsheet_is_refused():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("hello.txt", "not a workbook")
    assert oo.REASON_NOT_A_WORKBOOK in oo.inspect(out.getvalue()).problems


def test_a_compression_bomb_is_refused_before_any_part_is_read():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("xl/workbook.xml", b"\x00" * (300 * 1024 * 1024))
    inspection = oo.inspect(out.getvalue())
    assert oo.REASON_CONTAINER in inspection.problems
    assert inspection.parts == (), "parts were enumerated despite the container failing"


def test_a_member_path_that_escapes_is_refused():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("xl/workbook.xml", b"<workbook/>")
        archive.writestr("../../escape.xml", b"<x/>")
    assert oo.REASON_CONTAINER in oo.inspect(out.getvalue()).problems


# -- the XML parser is not a way in ---------------------------------------------------------


def test_a_doctype_is_refused_rather_than_parsed():
    """No spreadsheet part needs one, and both a doctype and an entity
    declaration are how a parser becomes a file reader or a bomb."""
    bomb = b'<?xml version="1.0"?><!DOCTYPE t [<!ENTITY a "x">]><t>&a;</t>'
    assert oo._parse_xml(bomb) is None


def test_an_oversized_part_is_not_parsed():
    assert oo._parse_xml(b"<t/>" + b" " * (oo.MAX_XML_PART_BYTES + 1)) is None


def test_deep_nesting_is_refused():
    deep = b"<a>" * (oo.MAX_XML_DEPTH + 10) + b"</a>" * (oo.MAX_XML_DEPTH + 10)
    assert oo._parse_xml(deep) is None


def test_ordinary_xml_still_parses():
    assert oo._parse_xml(b"<a><b/></a>") is not None


# -- reading a package back -----------------------------------------------------------------


def test_readable_text_reaches_beyond_the_grid():
    """A value can sit in a property or a relationship target, and a check
    that reads only the cells passes by not looking."""
    package = _workbook(with_comment=True)
    blob = oo.readable_text(package)
    assert "haemoglobin" in blob
    assert "checked by the lab" in blob
    assert "[Content_Types].xml" in blob


def test_readable_text_of_rubbish_is_empty():
    assert oo.readable_text(b"not a zip") == ""


def test_the_summary_carries_no_cell_or_target():
    package = _workbook(with_comment=True, with_hyperlink=True)
    summary = repr(oo.inspect(package).safe_summary())
    assert "haemoglobin" not in summary
    assert "example.invalid" not in summary
    assert "Nadia" not in summary
