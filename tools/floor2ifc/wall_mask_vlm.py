#!/usr/bin/env python3
"""Extract a first-pass wall mask from one SVG using VLM-labeled candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

from svg_inventory import GEOMETRY_TAGS, build_inventory, element_bounds, local_name, parse_float
from vlm_client import REPO_ROOT, VlmClient, extract_json_object


DEFAULT_OUT_DIR = Path(__file__).parent / "out/wall-mask-vlm"
OVERLAY_GROUP_ID = "floor2ifc-wall-candidate-overlay"
MASK_GROUP_ID = "floor2ifc-wall-mask"
SVG_NS = "http://www.w3.org/2000/svg"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def color_token(value: Any, missing: str) -> str:
    text = norm(value)
    if not text:
        return missing
    if text in {"#000", "#000000", "black", "rgb(0,0,0)", "rgb(0%,0%,0%)"}:
        return "black"
    if text in {"none", "transparent"}:
        return "none"
    return text


def is_default_black(element: dict[str, Any]) -> bool:
    style = element.get("style", {})
    return not style.get("fill") and not style.get("stroke") and element["tag"] in {"polygon", "rect", "path"}


def has_colored_fill(element: dict[str, Any]) -> bool:
    fill = color_token(element.get("style", {}).get("fill"), "default")
    return fill not in {"default", "black", "none"}


def is_wallish_candidate(element: dict[str, Any]) -> bool:
    tag = element["tag"]
    style = element.get("style", {})
    fill = color_token(style.get("fill"), "default")
    stroke = color_token(style.get("stroke"), "none")
    stroke_width = parse_float(style.get("stroke-width"), 0)
    area = float(element.get("area") or 0)
    length = float(element.get("length_estimate") or 0)
    aspect = float(element.get("aspect_ratio") or 0)
    width = float(element.get("width") or 0)
    height = float(element.get("height") or 0)
    thickness = min(width, height)

    if area < 16 or length < 8:
        return False
    if tag in {"circle", "ellipse"}:
        return False
    if is_default_black(element) and area >= 180:
        return True
    if fill == "black" and area >= 180:
        return True
    if has_colored_fill(element):
        return False
    if stroke == "black" and stroke_width <= 2 and (length >= 24 or area >= 400):
        return True
    if fill in {"black", "default"} and length >= 24 and (aspect >= 2.5 or thickness <= 28):
        return True
    if tag in {"polygon", "rect", "path"} and area >= 1000 and (stroke == "black" or fill == "black"):
        return True
    return False


def select_candidates(inventory: dict[str, Any], max_candidates: int) -> list[dict[str, Any]]:
    elements = [item for item in inventory["elements"] if is_wallish_candidate(item)]

    def score(element: dict[str, Any]) -> tuple[float, float, float]:
        style = element.get("style", {})
        default_black = 3.0 if is_default_black(element) else 0.0
        fill_black = 2.0 if color_token(style.get("fill"), "default") == "black" else 0.0
        stroke = 1.0 if color_token(style.get("stroke"), "none") == "black" else 0.0
        aspect = float(element.get("aspect_ratio") or 0)
        area = float(element.get("area") or 0)
        return (default_black + fill_black + stroke, area, aspect)

    return sorted(elements, key=score, reverse=True)[:max_candidates]


def visual_kind(element: dict[str, Any]) -> str:
    style = element.get("style", {})
    fill = color_token(style.get("fill"), "default")
    stroke = color_token(style.get("stroke"), "none")
    if is_default_black(element):
        return "default_fill"
    if fill == "black":
        return "black_fill"
    if stroke == "black":
        return "black_stroke"
    if has_colored_fill(element):
        return "colored_fill"
    return "other"


def candidate_group_id(element: dict[str, Any]) -> str:
    return f"{element['tag']}_{visual_kind(element)}"


def short_group_id(base_id: str, depth: int, candidate_ids: list[str], index: int) -> str:
    digest = hashlib.sha1(",".join(candidate_ids).encode("utf-8")).hexdigest()[:8]
    return f"{base_id}__d{depth}_{index:02d}_{digest}"


def geometry_bucket(element: dict[str, Any], depth: int) -> str:
    style = element.get("style", {})
    classes = "_".join(element.get("class") or ["none"])
    area = float(element.get("area") or 0)
    aspect = float(element.get("aspect_ratio") or 0)
    width = float(element.get("width") or 0)
    height = float(element.get("height") or 0)
    thickness = min(width, height)
    bbox = element.get("bbox", {})
    center_x = (float(bbox.get("min_x", 0)) + float(bbox.get("max_x", 0))) / 2
    center_y = (float(bbox.get("min_y", 0)) + float(bbox.get("max_y", 0))) / 2

    if depth <= 0:
        return candidate_group_id(element)
    if depth == 1:
        if area >= 50000:
            area_bucket = "huge"
        elif area >= 5000:
            area_bucket = "large"
        elif area >= 500:
            area_bucket = "medium"
        else:
            area_bucket = "small"
        return f"{candidate_group_id(element)}__area_{area_bucket}"
    if depth == 2:
        if aspect >= 12 or thickness <= 14:
            shape = "thin"
        elif aspect >= 3:
            shape = "long"
        else:
            shape = "block"
        return f"{candidate_group_id(element)}__shape_{shape}"
    if depth == 3:
        return f"{candidate_group_id(element)}__class_{norm(classes) or 'none'}"
    if depth == 4:
        fill = color_token(style.get("fill"), "default")
        stroke = color_token(style.get("stroke"), "none")
        return f"{candidate_group_id(element)}__paint_{fill}_{stroke}"
    return f"{candidate_group_id(element)}__pos_{int(center_x // 500)}_{int(center_y // 500)}"


def group_candidates(candidates: list[dict[str, Any]], max_group_size: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate_group_id(candidate), []).append(candidate)

    groups: list[dict[str, Any]] = []
    for base_id, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        for index in range(0, len(items), max_group_size):
            chunk = items[index : index + max_group_size]
            suffix = "" if len(items) <= max_group_size else f"_{index // max_group_size + 1:02d}"
            group_id = f"{base_id}{suffix}"
            groups.append(
                {
                    "group_id": group_id,
                    "base_group_id": base_id,
                    "depth": 0,
                    "candidate_count": len(chunk),
                    "candidate_ids": [item["id"] for item in chunk],
                    "candidates": chunk,
                }
            )
    return groups


def split_group(group: dict[str, Any], max_group_size: int, max_depth: int) -> list[dict[str, Any]]:
    next_depth = int(group.get("depth", 0)) + 1
    if next_depth > max_depth or len(group["candidates"]) <= 1:
        return []

    buckets: dict[str, list[dict[str, Any]]] = {}
    split_depth = next_depth
    for depth in range(next_depth, max_depth + 1):
        trial: dict[str, list[dict[str, Any]]] = {}
        for candidate in group["candidates"]:
            trial.setdefault(geometry_bucket(candidate, depth), []).append(candidate)
        if len(trial) > 1:
            buckets = trial
            split_depth = depth
            break

    if not buckets:
        midpoint = max(1, len(group["candidates"]) // 2)
        buckets = {
            "manual_a": group["candidates"][:midpoint],
            "manual_b": group["candidates"][midpoint:],
        }
        split_depth = next_depth

    split_groups: list[dict[str, Any]] = []
    child_index = 1
    for bucket_id, items in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        for index in range(0, len(items), max_group_size):
            chunk = items[index : index + max_group_size]
            candidate_ids = [item["id"] for item in chunk]
            group_id = short_group_id(group["base_group_id"], split_depth, candidate_ids, child_index)
            child_index += 1
            split_groups.append(
                {
                    "group_id": group_id,
                    "base_group_id": group["base_group_id"],
                    "parent_group_id": group["group_id"],
                    "split_key": bucket_id,
                    "depth": split_depth,
                    "candidate_count": len(chunk),
                    "candidate_ids": candidate_ids,
                    "candidates": chunk,
                }
            )
    return split_groups


def candidate_summary(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in candidates:
        style = item.get("style", {})
        bbox = item.get("bbox", {})
        rows.append(
            {
                "id": item["id"],
                "tag": item["tag"],
                "cls": item.get("class") or ["(none)"],
                "fill": style.get("fill", ""),
                "stroke": style.get("stroke", ""),
                "sw": style.get("stroke-width", ""),
                "bbox": [
                    round(float(bbox.get("min_x", 0)), 1),
                    round(float(bbox.get("min_y", 0)), 1),
                    round(float(bbox.get("max_x", 0)), 1),
                    round(float(bbox.get("max_y", 0)), 1),
                ],
                "a": item.get("area"),
                "ar": item.get("aspect_ratio"),
            }
        )
    return rows


def compact_style_summary(inventory: dict[str, Any], limit: int = 24) -> list[dict[str, Any]]:
    return inventory.get("style_summary", [])[:limit]


def build_group_prompt(inventory: dict[str, Any], group: dict[str, Any]) -> str:
    summary = {
        "source_svg": inventory.get("source_svg"),
        "viewBox": inventory.get("viewBox"),
        "group_id": group["group_id"],
        "base_group_id": group["base_group_id"],
        "candidate_count": group["candidate_count"],
        "style_summary_top": compact_style_summary(inventory),
        "candidates": candidate_summary(group["candidates"]),
    }
    return f"""You are judging whether one candidate mask group is pure wall in a floor-plan SVG.

You will see TWO images:
1. Original clean floor-plan image.
2. Candidate mask image for exactly one candidate group. Magenta shapes are the candidates in this group. There may be no visible ids.

Task:
- Compare image 1 and image 2.
- Decide whether ALL magenta mask shapes are structural walls, NO magenta mask shapes are structural walls, or the group is MIXED.
- Structural walls include exterior perimeter walls, interior partition walls, and wall outlines.
- Do not select furniture, fixtures, room fills, labels, dimension arrows, doors, windows, stairs, page borders, logos, or decorative details.
- Do not worry about walls that are not shown in this candidate mask; other groups will handle them.

Output rules:
- Return ONLY one minified JSON object.
- No markdown, no code fence, no checklist, no explanation, no prose.

JSON shape:
{{
  "group_id": "{group['group_id']}",
  "verdict": "all_wall|none_wall|mixed",
  "reason": "short visual reason",
  "confidence": 0.0
}}

Verdict rules:
- all_wall: every visible magenta mask shape in image 2 is wall.
- none_wall: no visible magenta mask shape in image 2 is wall.
- mixed: some magenta shapes are wall and some are not, or you cannot decide for the whole group.

Group/candidate summary:
{json.dumps(summary, ensure_ascii=False, separators=(",", ":"))[:12000]}
"""


def svg_close_insert(svg_text: str, insertion: str) -> str:
    if "</svg>" not in svg_text:
        raise ValueError("Could not find closing </svg>.")
    return svg_text.replace("</svg>", insertion + "\n</svg>", 1)


def candidate_overlay_element(elem: ET.Element) -> str:
    clone = deepcopy(elem)
    tag = local_name(clone.tag)
    clone.attrib.pop("class", None)
    if tag in {"line", "polyline"}:
        clone.attrib["fill"] = "none"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = "0.95"
        clone.attrib["stroke-width"] = mask_stroke_width(clone, fallback=5)
    elif tag == "path" and norm(clone.attrib.get("fill")) in {"", "none"} and clone.attrib.get("stroke"):
        clone.attrib["fill"] = "none"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = "0.95"
        clone.attrib["stroke-width"] = mask_stroke_width(clone, fallback=5)
    else:
        clone.attrib["fill"] = "#ff2d55"
        clone.attrib["fill-opacity"] = "0.42"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = "0.95"
        clone.attrib["stroke-width"] = "2"
    clone.attrib["vector-effect"] = "non-scaling-stroke"
    text = ET.tostring(clone, encoding="unicode")
    return re.sub(r"\sxmlns=\"[^\"]+\"", "", text, count=1)


def render_candidate_overlay(source_svg: Path, candidates: list[dict[str, Any]], out_svg: Path) -> None:
    original = source_svg.read_text(encoding="utf-8", errors="ignore")
    _root, by_id = index_source_elements(source_svg)
    parts = [
        f'\n<g id="{OVERLAY_GROUP_ID}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">',
    ]
    for item in candidates:
        elem = by_id.get(item["id"])
        if elem is None:
            continue
        cx, cy = item["center"]
        parts.append(candidate_overlay_element(elem))
        parts.append(
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="13" fill="#ff2d55" opacity="0.95"/>'
            f'<text x="{cx + 16:.3f}" y="{cy - 12:.3f}" fill="#ff2d55" '
            'paint-order="stroke" stroke="white" stroke-width="5" vector-effect="non-scaling-stroke">'
            f'{escape(item["id"])}</text>'
        )
    parts.append("</g>")
    out_svg.write_text(svg_close_insert(original, "\n".join(parts)), encoding="utf-8")


def render_group_mask(source_svg: Path, group: dict[str, Any], out_svg: Path, show_ids: bool) -> None:
    root, by_id = index_source_elements(source_svg)
    view_box = root.attrib.get("viewBox", "0 0 1000 1000")
    parts = [
        f'<svg xmlns="{SVG_NS}" viewBox="{escape(view_box)}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<g id="{OVERLAY_GROUP_ID}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">',
    ]
    for item in group["candidates"]:
        elem = by_id.get(item["id"])
        if elem is None:
            continue
        parts.append(candidate_overlay_element(elem))
        if show_ids:
            cx, cy = item["center"]
            parts.append(
                f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="13" fill="#ff2d55" opacity="0.95"/>'
                f'<text x="{cx + 16:.3f}" y="{cy - 12:.3f}" fill="#ff2d55" '
                'paint-order="stroke" stroke="white" stroke-width="5" vector-effect="non-scaling-stroke">'
                f'{escape(item["id"])}</text>'
            )
    parts.extend(["</g>", "</svg>"])
    out_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")


def index_source_elements(svg_path: Path) -> tuple[ET.Element, dict[str, ET.Element]]:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    by_id: dict[str, ET.Element] = {}
    index = 1
    for elem in root.iter():
        if local_name(elem.tag) not in GEOMETRY_TAGS:
            continue
        if element_bounds(elem) is None:
            continue
        by_id[f"e{index:05d}"] = elem
        index += 1
    return root, by_id


def mask_stroke_width(elem: ET.Element, fallback: float = 8) -> str:
    width = max(parse_float(elem.attrib.get("stroke-width"), 0), fallback)
    return f"{width:g}"


def mask_element(elem: ET.Element) -> str:
    clone = deepcopy(elem)
    tag = local_name(clone.tag)
    clone.attrib.pop("class", None)
    clone.attrib["fill"] = "#000"
    clone.attrib["fill-opacity"] = "1"
    clone.attrib["stroke"] = "#000"
    clone.attrib["stroke-opacity"] = "1"
    if tag in {"line", "polyline"}:
        clone.attrib["fill"] = "none"
        clone.attrib["stroke-width"] = mask_stroke_width(clone)
    elif tag == "path" and norm(clone.attrib.get("fill")) in {"", "none"}:
        clone.attrib["fill"] = "none"
        clone.attrib["stroke-width"] = mask_stroke_width(clone)
    else:
        clone.attrib["stroke-width"] = "0"
    text = ET.tostring(clone, encoding="unicode")
    return re.sub(r"\sxmlns=\"[^\"]+\"", "", text, count=1)


def render_mask_svgs(
    source_svg: Path,
    selected_ids: list[str],
    overlay_svg: Path,
    mask_svg: Path,
) -> None:
    root, by_id = index_source_elements(source_svg)
    view_box = root.attrib.get("viewBox", "0 0 1000 1000")
    mask_parts = [
        f'<svg xmlns="{SVG_NS}" viewBox="{escape(view_box)}">',
        f'<g id="{MASK_GROUP_ID}">',
    ]
    overlay_parts = [f'\n<g id="{MASK_GROUP_ID}" opacity="0.65">']
    for item_id in selected_ids:
        elem = by_id.get(item_id)
        if elem is None:
            continue
        part = mask_element(elem)
        mask_parts.append(part)
        overlay_parts.append(part.replace('fill="#000"', 'fill="#ff2d55"').replace('stroke="#000"', 'stroke="#ff2d55"'))
    mask_parts.extend(["</g>", "</svg>"])
    overlay_parts.append("</g>")

    mask_svg.write_text("\n".join(mask_parts) + "\n", encoding="utf-8")
    original = source_svg.read_text(encoding="utf-8", errors="ignore")
    overlay_svg.write_text(svg_close_insert(original, "\n".join(overlay_parts)), encoding="utf-8")


def parse_selector_width_range(selector: dict[str, Any]) -> tuple[float, float] | None:
    value = selector.get("stroke_width_px")
    if not isinstance(value, list) or len(value) != 2:
        return None
    return parse_float(value[0]), parse_float(value[1])


def list_constraint(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {norm(item) for item in value if norm(item)}


def selector_matches(selector: dict[str, Any], element: dict[str, Any]) -> bool:
    tag_values = list_constraint(selector.get("tags"))
    if tag_values and norm(element.get("tag")) not in tag_values:
        return False

    class_values = list_constraint(selector.get("classes"))
    element_classes = {norm(item) for item in element.get("class", [])}
    if class_values and "*" not in class_values:
        if "(none)" in class_values and not element_classes:
            pass
        elif not (class_values & element_classes):
            return False

    style = element.get("style", {})
    fill_values = {color_token(item, "default") for item in selector.get("fill", []) if norm(item)}
    if fill_values and color_token(style.get("fill"), "default") not in fill_values:
        return False
    stroke_values = {color_token(item, "none") for item in selector.get("stroke", []) if norm(item)}
    if stroke_values and color_token(style.get("stroke"), "none") not in stroke_values:
        return False
    width_range = parse_selector_width_range(selector)
    if width_range:
        stroke_width = parse_float(style.get("stroke-width"), 0)
        if not (width_range[0] <= stroke_width <= width_range[1]):
            return False
    return True


def selected_ids_from_review(
    review: dict[str, Any],
    candidates: list[dict[str, Any]],
    apply_selectors: bool,
) -> list[str]:
    candidate_ids = {item["id"] for item in candidates}
    selected: set[str] = {
        str(item)
        for item in review.get("wall_ids", [])
        if isinstance(item, str) and item in candidate_ids
    }
    if apply_selectors:
        for selector in review.get("wall_selectors", []):
            if not isinstance(selector, dict):
                continue
            if float(selector.get("confidence") or 0) < 0.6:
                continue
            for candidate in candidates:
                if selector_matches(selector, candidate):
                    selected.add(candidate["id"])
    return sorted(selected)


def parse_vlm_json(raw: str, raw_path: Path) -> dict[str, Any]:
    try:
        return extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as error:
        preview = raw[:1000].replace("\n", "\\n")
        raise ValueError(
            f"Could not parse VLM JSON: {error}. Raw response saved to {raw_path}. "
            f"Preview: {preview}"
        ) from error


def repair_vlm_json(client: VlmClient, raw: str, dump_request: bool, max_tokens: int) -> str:
    prompt = f"""Convert the following model response into ONLY one valid minified JSON object with keys group_id, verdict, reason, confidence.

Rules:
- No markdown.
- No explanation.
- verdict must be one of all_wall, none_wall, mixed.
- If the response is incomplete or unusable, return {{"group_id":"","verdict":"mixed","reason":"unusable_partial_response","confidence":0}}.

Response:
{raw[:12000]}
"""
    return client.chat(prompt, [], dump_request=dump_request)


def run(svg_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(svg_path)
    candidates = select_candidates(inventory, args.max_candidates)
    initial_groups = group_candidates(candidates, args.max_group_size)

    inventory_path = out_dir / "inventory.json"
    candidates_path = out_dir / "candidates.json"
    candidate_overlay_path = out_dir / "candidate_overlay.svg"
    groups_dir = out_dir / "groups"
    groups_path = out_dir / "candidate_groups.json"
    review_path = out_dir / "vlm_wall_labels.json"
    mask_path = out_dir / "wall_mask.svg"
    overlay_path = out_dir / "wall_mask_overlay.svg"
    manifest_path = out_dir / "result.json"

    write_json(inventory_path, inventory)
    write_json(candidates_path, {"source_svg": str(svg_path), "candidates": candidates})
    write_json(
        groups_path,
        {
            "source_svg": str(svg_path),
            "group_count": len(initial_groups),
            "groups": [
                {
                    "group_id": group["group_id"],
                    "base_group_id": group["base_group_id"],
                    "depth": group["depth"],
                    "candidate_count": group["candidate_count"],
                    "candidate_ids": group["candidate_ids"],
                }
                for group in initial_groups
            ],
        },
    )
    render_candidate_overlay(svg_path, candidates, candidate_overlay_path)

    groups_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.prepare_only or args.mock_response else VlmClient(args.env_file_root)
    group_reviews: list[dict[str, Any]] = []
    selected_id_set: set[str] = set()
    queue = list(initial_groups)
    processed_count = 0

    mock_review = None
    if args.mock_response:
        mock_review = extract_json_object(args.mock_response.read_text(encoding="utf-8"))

    while queue:
        group = queue.pop(0)
        processed_count += 1
        group_dir = groups_dir / group["group_id"]
        group_dir.mkdir(parents=True, exist_ok=True)
        group_mask_path = group_dir / "candidate_mask.svg"
        group_prompt_path = group_dir / "wall_label_prompt.txt"
        group_raw_path = group_dir / "vlm_raw.txt"
        group_review_path = group_dir / "vlm_wall_labels.json"
        group_prompt = build_group_prompt(inventory, group)

        render_group_mask(svg_path, group, group_mask_path, args.show_ids)
        group_prompt_path.write_text(group_prompt, encoding="utf-8")

        review: dict[str, Any] = {
            "group_id": group["group_id"],
            "verdict": "mixed" if args.prepare_only else "none_wall",
            "reason": "prepare_only" if args.prepare_only else "",
            "confidence": 0,
        }
        if mock_review is not None:
            review = {**review, **mock_review, "group_id": group["group_id"]}
            group_raw_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
        elif not args.prepare_only and client is not None:
            raw = client.chat_with_svgs(
                group_prompt,
                [svg_path, group_mask_path],
                image_size=args.image_size,
                dump_request=args.dump_request,
                max_tokens=args.max_output_tokens,
            )
            group_raw_path.write_text(raw, encoding="utf-8")
            try:
                review = parse_vlm_json(raw, group_raw_path)
            except ValueError:
                repaired = repair_vlm_json(client, raw, args.dump_request, args.max_output_tokens)
                repair_path = group_dir / "vlm_repair_raw.txt"
                repair_path.write_text(repaired, encoding="utf-8")
                review = parse_vlm_json(repaired, repair_path)

        verdict = str(review.get("verdict") or "").strip().lower()
        if verdict not in {"all_wall", "none_wall", "mixed"}:
            verdict = "mixed"

        selected_ids: list[str] = []
        split_group_ids: list[str] = []
        if verdict == "all_wall":
            selected_ids = list(group["candidate_ids"])
            selected_id_set.update(selected_ids)
        elif verdict == "mixed":
            children = split_group(group, args.max_group_size, args.max_split_depth)
            if children:
                queue.extend(children)
                split_group_ids = [child["group_id"] for child in children]
            elif args.accept_single_mixed and len(group["candidate_ids"]) == 1:
                selected_ids = list(group["candidate_ids"])
                selected_id_set.update(selected_ids)

        write_json(group_review_path, review)
        group_reviews.append(
            {
                "group_id": group["group_id"],
                "base_group_id": group["base_group_id"],
                "parent_group_id": group.get("parent_group_id"),
                "split_key": group.get("split_key"),
                "depth": group["depth"],
                "candidate_count": group["candidate_count"],
                "verdict": verdict,
                "selected_wall_count": len(selected_ids),
                "selected_wall_ids": selected_ids,
                "split_group_ids": split_group_ids,
                "candidate_mask": str(group_mask_path),
                "prompt": str(group_prompt_path),
                "vlm_raw": str(group_raw_path) if group_raw_path.exists() else None,
                "vlm_wall_labels": str(group_review_path),
                "confidence": review.get("confidence"),
            }
        )

    selected_ids = sorted(selected_id_set)
    write_json(
        review_path,
        {
            "source_svg": str(svg_path),
            "initial_group_count": len(initial_groups),
            "processed_group_count": processed_count,
            "selected_wall_count": len(selected_ids),
            "selected_wall_ids": selected_ids,
            "groups": group_reviews,
        },
    )
    render_mask_svgs(svg_path, selected_ids, overlay_path, mask_path)

    manifest = {
        "source_svg": str(svg_path),
        "status": "prepared" if args.prepare_only and not args.mock_response else "completed",
        "candidate_count": len(candidates),
        "initial_group_count": len(initial_groups),
        "processed_group_count": processed_count,
        "selected_wall_count": len(selected_ids),
        "selected_wall_ids": selected_ids,
        "inventory": str(inventory_path),
        "candidates": str(candidates_path),
        "candidate_groups": str(groups_path),
        "candidate_overlay_debug": str(candidate_overlay_path),
        "groups_dir": str(groups_dir),
        "vlm_wall_labels": str(review_path),
        "wall_mask_svg": str(mask_path),
        "wall_mask_overlay_svg": str(overlay_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path, help="Input floor-plan SVG.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-file-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--image-size", type=int, default=1800)
    parser.add_argument("--max-candidates", type=int, default=220)
    parser.add_argument("--max-group-size", type=int, default=45)
    parser.add_argument("--max-split-depth", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--show-ids", action="store_true", help="Draw candidate ids on group masks for debugging.")
    parser.add_argument("--accept-single-mixed", action="store_true", help="Treat a single-candidate mixed leaf as wall.")
    parser.add_argument("--prepare-only", action="store_true", help="Write inventory/overlay/prompt without calling VLM.")
    parser.add_argument("--mock-response", type=Path, default=None, help="Use a saved VLM JSON response instead of calling VLM.")
    parser.add_argument("--dump-request", action="store_true")
    args = parser.parse_args()

    if not args.svg.exists():
        print(f"SVG not found: {args.svg}", file=sys.stderr)
        return 2
    if args.mock_response and not args.mock_response.exists():
        print(f"Mock response not found: {args.mock_response}", file=sys.stderr)
        return 2

    out_dir = args.out_dir / args.svg.stem
    try:
        manifest = run(args.svg, out_dir, args)
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"status: {manifest['status']}")
    print(f"candidates: {manifest['candidate_count']}")
    print(f"initial groups: {manifest['initial_group_count']}")
    print(f"processed groups: {manifest['processed_group_count']}")
    print(f"selected walls: {manifest['selected_wall_count']}")
    print(f"candidate overlay debug: {manifest['candidate_overlay_debug']}")
    print(f"groups dir: {manifest['groups_dir']}")
    print(f"wall mask: {manifest['wall_mask_svg']}")
    print(f"wall mask overlay: {manifest['wall_mask_overlay_svg']}")
    print(f"manifest: {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
