from collections import Counter


def _surface_value(view, item):
    surface = item["surface"]
    sheet = view.sheets.get(item.get("sheet", ""))
    cell = item.get("cell", "")
    if surface == "comment_text":
        return sheet.comments.get(cell) if sheet else None
    if surface == "comment_author":
        return sheet.comment_authors.get(cell) if sheet else None
    if surface == "formula_literal":
        return sheet.formulas.get(cell) if sheet else None
    if surface == "hyperlink":
        return sheet.hyperlinks.get(cell) if sheet else None
    if surface == "header":
        return " ".join(sheet.headers) if sheet else None
    if surface == "footer":
        return " ".join(sheet.footers) if sheet else None
    if surface in {"hidden_cell", "very_hidden_cell"}:
        return sheet.cells.get(cell) if sheet else None
    if surface == "core_property":
        return view.core_properties.get(item["field"])
    if surface == "custom_property":
        return view.custom_properties.get(item["field"])
    if surface == "defined_name":
        name = item["field"]
        return (
            name + " " + view.defined_names[name]
            if name in view.defined_names
            else None
        )
    if surface == "chart_title":
        return (
            " ".join(
                content.decode("utf-8", "ignore")
                for name, content in view.parts.items()
                if name.startswith("xl/charts/") and name.endswith(".xml")
            )
            or None
        )
    raise ValueError("unknown surface in versioned gold")


def _output_sheet_name(source_view, output_view, source_name):
    """Follow a renamed sheet by its package ordinal and visibility state."""
    if source_name in output_view.sheets:
        return source_name
    if source_name not in source_view.sheets:
        return None
    names_before = list(source_view.sheets)
    names_after = list(output_view.sheets)
    ordinal = names_before.index(source_name)
    if ordinal >= len(names_after):
        return None
    candidate = names_after[ordinal]
    if source_view.sheets[source_name].state != output_view.sheets[candidate].state:
        return None
    return candidate


def _surface_evidence(source_view, output_view, protocol):
    counts = Counter()
    unexpected = []
    planted_survivors = []
    for ordinal, item in enumerate(protocol["occurrences"]):
        source = _surface_value(source_view, item)
        if not source or item["value"] not in source:
            unexpected.append(
                {
                    "ordinal": ordinal,
                    "surface": item["surface"],
                    "status": "source_not_measurable",
                }
            )
            counts["not_measurable"] += 1
            continue
        output_item = item
        if item.get("sheet"):
            mapped_name = _output_sheet_name(source_view, output_view, item["sheet"])
            output_item = {**item, "sheet": mapped_name or ""}
        output = _surface_value(output_view, output_item)
        if output and item["value"] in output:
            status = "preserved"
            planted_survivors.append({"ordinal": ordinal, "surface": item["surface"]})
        elif not output:
            status = (
                "removed_unsupported"
                if item["expected"] == "removed_unsupported"
                else "dropped"
            )
        else:
            status = "transformed"
        counts[status] += 1
        acceptable = status == item["expected"] or (
            status == "dropped" and item["expected"] == "removed_by_policy"
        )
        if not acceptable:
            unexpected.append(
                {"ordinal": ordinal, "surface": item["surface"], "status": status}
            )
    for item in protocol["non_text_surfaces"]:
        surface = item["surface"]
        if surface == "full_face_image":
            source_present = any(
                name.startswith("xl/media/") for name in source_view.parts
            )
            output_present = any(
                name.startswith("xl/media/") for name in output_view.parts
            )
        elif surface == "chart":
            source_present = any(
                name.startswith("xl/charts/") for name in source_view.parts
            )
            output_present = any(
                name.startswith("xl/charts/") for name in output_view.parts
            )
        elif surface == "external_workbook_link":
            source_present = bool(source_view.external_relationships)
            output_present = bool(output_view.external_relationships)
        elif surface == "text_box":
            source_present = bool(source_view.retained_drawing_text)
            output_present = bool(output_view.retained_drawing_text)
        elif surface == "shared_strings":
            source_present = bool(source_view.shared_strings)
            output_present = bool(output_view.shared_strings)
        elif surface == "cached_formula_value":
            source_present = any(
                sheet.cached_values for sheet in source_view.sheets.values()
            )
            output_present = any(
                sheet.cached_values for sheet in output_view.sheets.values()
            )
        else:
            raise ValueError("unknown non-text surface")
        status = (
            "not_measurable"
            if not source_present
            else ("preserved" if output_present else "removed_unsupported")
        )
        counts[status] += 1
        if status != item["expected"]:
            unexpected.append({"surface": surface, "status": status})
    # A sanitized field could still leave the same value in an unused ZIP
    # part. The targeted parser proves the occurrence; this check proves the
    # package has no second unaccounted copy.
    package_survivors = []
    for ordinal, item in enumerate(protocol["occurrences"]):
        parts = [
            name
            for name, content in output_view.parts.items()
            if item["value"].encode() in content
        ]
        if parts and not any(
            found.get("ordinal") == ordinal for found in planted_survivors
        ):
            package_survivors.append({"ordinal": ordinal, "parts": parts})
    return dict(counts), unexpected, planted_survivors, package_survivors
