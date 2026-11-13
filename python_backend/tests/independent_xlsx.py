"""Small read-only OOXML inspector independent of the openpyxl sanitizer."""

from __future__ import annotations

import io
import posixpath
import zipfile
from dataclasses import dataclass
from typing import Dict, Tuple
from xml.etree import ElementTree as ET

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class SheetView:
    state: str
    cells: Dict[str, str]
    formulas: Dict[str, str]
    cached_values: Dict[str, str]
    comments: Dict[str, str]
    comment_authors: Dict[str, str]
    headers: Tuple[str, ...]
    footers: Tuple[str, ...]
    hyperlinks: Dict[str, str]
    part: str


@dataclass(frozen=True)
class PackageView:
    sheets: Dict[str, SheetView]
    parts: Dict[str, bytes]
    core_properties: Dict[str, str]
    custom_properties: Dict[str, str]
    defined_names: Dict[str, str]
    shared_strings: Tuple[str, ...]
    unused_shared_strings: Tuple[int, ...]
    retained_drawing_text: Tuple[str, ...]
    external_relationships: Tuple[str, ...]


def _text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.iter(f"{{{MAIN}}}t"))


def inspect(content: bytes) -> PackageView:
    """Parse workbook relationships, shared strings and actual serialized cells."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith("/") or ".." in name.split("/") for name in names
        ):
            raise ValueError("unsafe OOXML member name")
        parts = {name: archive.read(name) for name in names}
    workbook = ET.fromstring(parts["xl/workbook.xml"])
    rels = ET.fromstring(parts["xl/_rels/workbook.xml.rels"])
    targets = {
        item.attrib["Id"]: item.attrib["Target"]
        for item in rels.findall(f"{{{PKG_REL}}}Relationship")
    }
    shared = []
    if "xl/sharedStrings.xml" in parts:
        root = ET.fromstring(parts["xl/sharedStrings.xml"])
        shared = [_text(item) for item in root.findall(f"{{{MAIN}}}si")]
    used_shared = set()
    defined_names = {
        item.attrib.get("name", ""): item.text or ""
        for item in workbook.findall(f".//{{{MAIN}}}definedName")
    }
    core_properties = {}
    if "docProps/core.xml" in parts:
        root = ET.fromstring(parts["docProps/core.xml"])
        core_properties = {
            item.tag.rsplit("}", 1)[-1]: item.text or "" for item in root
        }
    custom_properties = {}
    if "docProps/custom.xml" in parts:
        root = ET.fromstring(parts["docProps/custom.xml"])
        custom_properties = {
            item.attrib.get("name", ""): "".join(item.itertext()) for item in root
        }
    drawing_text = []
    external_relationships = []
    for name, data in parts.items():
        if name.endswith(".rels"):
            root = ET.fromstring(data)
            external_relationships.extend(
                item.attrib.get("Target", "")
                for item in root.findall(f"{{{PKG_REL}}}Relationship")
                if item.attrib.get("TargetMode") == "External"
            )
        if name.endswith(".vml"):
            root = ET.fromstring(data)
            drawing_text.extend(
                (item.text or "").strip()
                for item in root.iter()
                if item.text
                and item.text.strip()
                and item.tag.rsplit("}", 1)[-1] not in {"AutoFill", "Row", "Column"}
            )
    sheets = {}
    for item in workbook.findall(f".//{{{MAIN}}}sheet"):
        title = item.attrib["name"]
        target = targets[item.attrib[f"{{{REL}}}id"]]
        part = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath(posixpath.join("xl", target))
        )
        root = ET.fromstring(parts[part])
        cells: Dict[str, str] = {}
        formulas: Dict[str, str] = {}
        cached_values: Dict[str, str] = {}
        for cell in root.findall(f".//{{{MAIN}}}c"):
            address = cell.attrib["r"]
            value = cell.find(f"{{{MAIN}}}v")
            inline = cell.find(f"{{{MAIN}}}is")
            formula = cell.find(f"{{{MAIN}}}f")
            if formula is not None:
                formulas[address] = formula.text or ""
                if value is not None and value.text:
                    cached_values[address] = value.text
            if inline is not None:
                cells[address] = _text(inline)
            elif value is not None:
                raw = value.text or ""
                if cell.attrib.get("t") == "s":
                    index = int(raw)
                    used_shared.add(index)
                    cells[address] = shared[index]
                else:
                    cells[address] = raw
            elif formula is not None:
                cells[address] = ""
        header_footer = root.find(f"{{{MAIN}}}headerFooter")
        headers = (
            tuple(
                child.text or ""
                for child in header_footer
                if child.tag.rsplit("}", 1)[-1].lower().endswith("header")
            )
            if header_footer is not None
            else ()
        )
        footers = (
            tuple(
                child.text or ""
                for child in header_footer
                if child.tag.rsplit("}", 1)[-1].lower().endswith("footer")
            )
            if header_footer is not None
            else ()
        )
        rel_name = posixpath.join(
            posixpath.dirname(part), "_rels", posixpath.basename(part) + ".rels"
        )
        sheet_rels = {}
        if rel_name in parts:
            rel_root = ET.fromstring(parts[rel_name])
            sheet_rels = {
                child.attrib["Id"]: child
                for child in rel_root.findall(f"{{{PKG_REL}}}Relationship")
            }
        hyperlinks = {}
        for link in root.findall(f".//{{{MAIN}}}hyperlink"):
            relation = sheet_rels.get(link.attrib.get(f"{{{REL}}}id", ""))
            hyperlinks[link.attrib.get("ref", "")] = (
                relation.attrib.get("Target", "")
                if relation is not None
                else link.attrib.get("location", "")
            )
        comments = {}
        authors = {}
        for relation in sheet_rels.values():
            if not relation.attrib.get("Type", "").endswith("/comments"):
                continue
            target = relation.attrib["Target"]
            comment_part = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath(posixpath.join(posixpath.dirname(part), target))
            )
            if comment_part not in parts:
                continue
            comment_root = ET.fromstring(parts[comment_part])
            names = [
                item.text or "" for item in comment_root.findall(f".//{{{MAIN}}}author")
            ]
            for comment in comment_root.findall(f".//{{{MAIN}}}comment"):
                address = comment.attrib["ref"]
                comments[address] = _text(comment)
                author_id = int(comment.attrib.get("authorId", "0"))
                authors[address] = names[author_id] if author_id < len(names) else ""
        sheets[title] = SheetView(
            item.attrib.get("state", "visible"),
            cells,
            formulas,
            cached_values,
            comments,
            authors,
            headers,
            footers,
            hyperlinks,
            part,
        )
    return PackageView(
        sheets,
        parts,
        core_properties,
        custom_properties,
        defined_names,
        tuple(shared),
        tuple(index for index in range(len(shared)) if index not in used_shared),
        tuple(drawing_text),
        tuple(external_relationships),
    )


def coordinate(column: int, row: int) -> str:
    """One-based column and row to A1 coordinate."""
    label = ""
    while column:
        column, digit = divmod(column - 1, 26)
        label = chr(65 + digit) + label
    return f"{label}{row}"
