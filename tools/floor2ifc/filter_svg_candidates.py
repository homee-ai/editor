#!/usr/bin/env python3
"""Filter obvious non-wall SVG elements and render a red kept-candidate overlay.

This is a tuning/debug tool for the png2svg -> wall_mask_vlm pipeline. It reads
one colour-cutout SVG produced by png2svg.py, applies cheap rule-based filters,
and writes the remaining elements as a red mask over the original SVG.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

from svg_inventory import GEOMETRY_TAGS, build_inventory, element_bounds, local_name, parse_float


DEFAULT_OUT_DIR = Path(__file__).parent / "out/svg-candidate-filter"
SVG_NS = "http://www.w3.org/2000/svg"
OVERLAY_GROUP_ID = "floor2ifc-filter-kept-overlay"
MASK_GROUP_ID = "floor2ifc-filter-kept-mask"


def norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def parse_rgb(value: Any) -> tuple[int, int, int] | None:
    text = norm(value)
    if not text or text in {"none", "transparent", "default", "(default)", "(none)"}:
        return None
    short_hex = re.fullmatch(r"#([0-9a-f]{3})", text)
    if short_hex:
        digits = short_hex.group(1)
        return tuple(int(ch * 2, 16) for ch in digits)  # type: ignore[return-value]
    long_hex = re.fullmatch(r"#([0-9a-f]{6})", text)
    if long_hex:
        digits = long_hex.group(1)
        return tuple(int(digits[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
    rgb = re.fullmatch(r"rgba?\(([^)]+)\)", text)
    if rgb:
        parts = [part.strip() for part in rgb.group(1).split(",")[:3]]
        if len(parts) == 3:
            values: list[int] = []
            for part in parts:
                if part.endswith("%"):
                    values.append(round(float(part[:-1]) * 2.55))
                else:
                    values.append(round(float(part)))
            return tuple(max(0, min(255, item)) for item in values)  # type: ignore[return-value]
    if text == "black":
        return (0, 0, 0)
    if text == "white":
        return (255, 255, 255)
    return None


def luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def color_token(value: Any, missing: str) -> str:
    text = norm(value)
    if not text:
        return missing
    if text in {"#000", "#000000", "black", "rgb(0,0,0)", "rgb(0%,0%,0%)"}:
        return "black"
    if text in {"none", "transparent"}:
        return "none"
    return text


def paint_luminance(value: Any, default: float) -> float:
    rgb = parse_rgb(value)
    return luminance(rgb) if rgb is not None else default


def is_default_black(element: dict[str, Any]) -> bool:
    style = element.get("style", {})
    return not style.get("fill") and not style.get("stroke") and element["tag"] in {"polygon", "rect", "path"}


def viewbox_numbers(inventory: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = inventory.get("viewBox")
    if not value:
        return None
    numbers = [float(item) for item in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))]
    if len(numbers) != 4:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def page_area(inventory: dict[str, Any]) -> float | None:
    viewbox = viewbox_numbers(inventory)
    if viewbox is None:
        return None
    return max(viewbox[2] * viewbox[3], 1e-6)


def is_hidden(element: dict[str, Any]) -> bool:
    style = element.get("style", {})
    return norm(style.get("display")) == "none" or norm(style.get("visibility")) == "hidden"


def effective_stroke_width(element: dict[str, Any]) -> float:
    style = element.get("style", {})
    raw = parse_float(style.get("stroke-width"), 0)
    return float(element.get("effective_stroke_width") or raw)


def reject_reason(element: dict[str, Any], inventory: dict[str, Any], args: argparse.Namespace) -> str | None:
    if is_hidden(element):
        return "hidden"

    tag = element["tag"]
    area = float(element.get("area") or 0)
    width = float(element.get("width") or 0)
    height = float(element.get("height") or 0)
    aspect = float(element.get("aspect_ratio") or 0)
    thickness = min(width, height)
    length = float(element.get("length_estimate") or 0)
    style = element.get("style", {})
    fill = color_token(style.get("fill"), "default")
    stroke = color_token(style.get("stroke"), "none")
    stroke_width = effective_stroke_width(element)

    if args.reject_round and tag in {"circle", "ellipse"}:
        return "round_shape"
    if area < args.min_area:
        return "small_area"
    canvas_area = page_area(inventory)
    if canvas_area is not None and area / canvas_area >= args.max_area_ratio:
        return "page_sized"

    fill_luma = paint_luminance(style.get("fill"), 0 if is_default_black(element) else 255)
    stroke_luma = paint_luminance(style.get("stroke"), 255)
    has_real_stroke = stroke not in {"none", "default"} or stroke_width > 0
    darkest_paint = min(fill_luma, stroke_luma if has_real_stroke else 255)

    if args.exclude_light_fill >= 0 and fill not in {"none", "default"}:
        rgb = parse_rgb(style.get("fill"))
        if rgb is not None and min(rgb) >= args.exclude_light_fill and not has_real_stroke:
            return "light_fill"
    if args.max_paint_luminance >= 0 and darkest_paint > args.max_paint_luminance:
        return "too_light"

    fill_rgb = parse_rgb(style.get("fill"))
    fill_is_solid = fill_rgb is not None and fill not in {"none", "default"} and not has_real_stroke
    if fill_is_solid and area >= args.light_area_min_area:
        if fill_luma >= args.light_area_luminance and (
            aspect <= args.light_area_max_aspect or thickness >= args.light_area_min_thickness
        ):
            return "light_area_fill"
        if fill_luma >= args.very_light_luminance and thickness >= args.very_light_min_thickness:
            return "very_light_fill"
        if (
            args.medium_compact_luminance >= 0
            and fill_luma >= args.medium_compact_luminance
            and area >= args.medium_compact_min_area
            and aspect <= args.medium_compact_max_aspect
        ):
            return "medium_compact_fill"
        if (
            args.nonlinear_fill_luminance >= 0
            and fill_luma >= args.nonlinear_fill_luminance
            and area >= args.nonlinear_fill_min_area
            and aspect <= args.nonlinear_fill_max_aspect
            and thickness >= args.nonlinear_fill_min_thickness
        ):
            return "nonlinear_mid_fill"

    if args.reject_compact_small and area < args.compact_area and aspect < args.min_compact_aspect:
        return "compact_small"
    if args.min_length > 0 and length < args.min_length and aspect < args.min_length_aspect:
        return "short_non_linear"
    if args.max_thin_stroke_width >= 0 and stroke_width > 0 and stroke_width <= args.max_thin_stroke_width:
        if area < args.thin_stroke_min_area and length < args.thin_stroke_min_length:
            return "tiny_thin_stroke"
    if args.max_thickness > 0 and thickness > args.max_thickness and aspect < args.min_aspect_for_thick:
        return "thick_blob"

    return None


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


def svg_close_insert(svg_text: str, insertion: str) -> str:
    if "</svg>" not in svg_text:
        raise ValueError("Could not find closing </svg>.")
    return svg_text.replace("</svg>", insertion + "\n</svg>", 1)


def ensure_stroke_width(elem: ET.Element, fallback: float) -> None:
    if parse_float(elem.attrib.get("stroke-width"), 0) <= 0:
        elem.attrib["stroke-width"] = f"{fallback:g}"


def is_stroke_only(elem: ET.Element, item: dict[str, Any]) -> bool:
    if local_name(elem.tag) in {"line", "polyline"}:
        return True
    fill = color_token(item.get("style", {}).get("fill"), "default")
    stroke = color_token(item.get("style", {}).get("stroke"), "none")
    return local_name(elem.tag) == "path" and fill == "none" and stroke != "none"


def copy_resolved_stroke_width(clone: ET.Element, item: dict[str, Any]) -> None:
    if parse_float(clone.attrib.get("stroke-width"), 0) > 0:
        return
    stroke_width = item.get("style", {}).get("stroke-width")
    if stroke_width:
        clone.attrib["stroke-width"] = str(stroke_width)


def red_overlay_element(elem: ET.Element, item: dict[str, Any], opacity: float) -> str:
    clone = deepcopy(elem)
    clone.attrib.pop("class", None)
    if is_stroke_only(elem, item):
        clone.attrib["fill"] = "none"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = f"{opacity:g}"
        copy_resolved_stroke_width(clone, item)
        ensure_stroke_width(clone, fallback=5)
    else:
        clone.attrib["fill"] = "#ff2d55"
        clone.attrib["fill-opacity"] = f"{opacity:g}"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = "0.95"
        ensure_stroke_width(clone, fallback=1)
    text = ET.tostring(clone, encoding="unicode")
    return re.sub(r"\sxmlns=\"[^\"]+\"", "", text, count=1)


def black_mask_element(elem: ET.Element, item: dict[str, Any]) -> str:
    clone = deepcopy(elem)
    clone.attrib.pop("class", None)
    clone.attrib["fill"] = "#000"
    clone.attrib["fill-opacity"] = "1"
    clone.attrib["stroke"] = "#000"
    clone.attrib["stroke-opacity"] = "1"
    if is_stroke_only(elem, item):
        clone.attrib["fill"] = "none"
        copy_resolved_stroke_width(clone, item)
        ensure_stroke_width(clone, fallback=5)
    else:
        clone.attrib["stroke-width"] = "0"
    text = ET.tostring(clone, encoding="unicode")
    return re.sub(r"\sxmlns=\"[^\"]+\"", "", text, count=1)


def render_outputs(
    svg_path: Path,
    inventory: dict[str, Any],
    kept: list[dict[str, Any]],
    out_dir: Path,
    opacity: float,
    show_ids: bool,
) -> tuple[Path, Path]:
    root, by_id = index_source_elements(svg_path)
    view_box = root.attrib.get("viewBox", "0 0 1000 1000")

    overlay_parts = [
        f'\n<g id="{OVERLAY_GROUP_ID}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">',
    ]
    mask_parts = [
        f'<svg xmlns="{SVG_NS}" viewBox="{escape(view_box)}">',
        f'<g id="{MASK_GROUP_ID}">',
    ]
    for item in kept:
        elem = by_id.get(item["id"])
        if elem is None:
            continue
        overlay_parts.append(red_overlay_element(elem, item, opacity))
        mask_parts.append(black_mask_element(elem, item))
        if show_ids:
            cx, cy = item["center"]
            overlay_parts.append(
                f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="13" fill="#ff2d55" opacity="0.95"/>'
                f'<text x="{cx + 16:.3f}" y="{cy - 12:.3f}" fill="#ff2d55" '
                'paint-order="stroke" stroke="white" stroke-width="5" vector-effect="non-scaling-stroke">'
                f'{escape(item["id"])}</text>'
            )
    overlay_parts.append("</g>")
    mask_parts.extend(["</g>", "</svg>"])

    overlay_path = out_dir / "filtered_overlay.svg"
    mask_path = out_dir / "kept_mask.svg"
    original = svg_path.read_text(encoding="utf-8", errors="ignore")
    overlay_path.write_text(svg_close_insert(original, "\n".join(overlay_parts)), encoding="utf-8")
    mask_path.write_text("\n".join(mask_parts) + "\n", encoding="utf-8")
    return overlay_path, mask_path


def run(svg_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(svg_path)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for element in inventory["elements"]:
        reason = reject_reason(element, inventory, args)
        if reason is None:
            kept.append(element)
        else:
            rejected.append({"id": element["id"], "reason": reason})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    overlay_path, mask_path = render_outputs(svg_path, inventory, kept, out_dir, args.overlay_opacity, args.show_ids)
    report = {
        "source_svg": str(svg_path),
        "element_count": inventory["element_count"],
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "rejected_by_reason": dict(sorted(reason_counts.items())),
        "parameters": {
            key: value
            for key, value in vars(args).items()
            if key not in {"svg", "out_dir"}
        },
        "kept_ids": [item["id"] for item in kept],
        "rejected": rejected,
        "filtered_overlay_svg": str(overlay_path),
        "kept_mask_svg": str(mask_path),
    }
    report_path = out_dir / "filter_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**report, "filter_report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path, help="SVG produced by png2svg.py.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-area", type=float, default=180, help="Reject elements with bbox area below this.")
    parser.add_argument(
        "--exclude-light-fill",
        type=int,
        default=245,
        help="Reject unstroked fills whose RGB channels are all at least this value. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-paint-luminance",
        type=float,
        default=248,
        help="Reject elements whose darkest visible paint is lighter than this. Use -1 to disable.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.82,
        help="Reject elements occupying at least this fraction of the SVG viewBox.",
    )
    parser.add_argument(
        "--light-area-luminance",
        type=float,
        default=190,
        help="Reject large unstroked fills at or above this luminance when they look like area fills.",
    )
    parser.add_argument("--light-area-min-area", type=float, default=4500)
    parser.add_argument("--light-area-max-aspect", type=float, default=2.8)
    parser.add_argument("--light-area-min-thickness", type=float, default=75)
    parser.add_argument(
        "--very-light-luminance",
        type=float,
        default=225,
        help="Reject very light unstroked fills above the area threshold when not hairline-thin.",
    )
    parser.add_argument("--very-light-min-thickness", type=float, default=28)
    parser.add_argument(
        "--medium-compact-luminance",
        type=float,
        default=135,
        help="Reject medium/bright compact unstroked fills. Use -1 to disable.",
    )
    parser.add_argument("--medium-compact-min-area", type=float, default=5000)
    parser.add_argument("--medium-compact-max-aspect", type=float, default=1.45)
    parser.add_argument(
        "--nonlinear-fill-luminance",
        type=float,
        default=70,
        help="Reject unstroked filled blobs at/above this luminance when they are thick and not wall-like linear.",
    )
    parser.add_argument("--nonlinear-fill-min-area", type=float, default=3000)
    parser.add_argument("--nonlinear-fill-max-aspect", type=float, default=2.0)
    parser.add_argument("--nonlinear-fill-min-thickness", type=float, default=45)
    parser.add_argument("--reject-round", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-compact-small", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compact-area", type=float, default=900)
    parser.add_argument("--min-compact-aspect", type=float, default=1.8)
    parser.add_argument("--min-length", type=float, default=0)
    parser.add_argument("--min-length-aspect", type=float, default=2.0)
    parser.add_argument("--max-thin-stroke-width", type=float, default=1.5)
    parser.add_argument("--thin-stroke-min-area", type=float, default=500)
    parser.add_argument("--thin-stroke-min-length", type=float, default=80)
    parser.add_argument("--max-thickness", type=float, default=0, help="Reject thick blobs above this thickness. 0 disables.")
    parser.add_argument("--min-aspect-for-thick", type=float, default=2.0)
    parser.add_argument("--overlay-opacity", type=float, default=0.42)
    parser.add_argument("--show-ids", action="store_true")
    args = parser.parse_args()

    if not args.svg.exists():
        print(f"SVG not found: {args.svg}")
        return 2

    out_dir = args.out_dir / args.svg.stem
    manifest = run(args.svg, out_dir, args)
    print(f"elements: {manifest['element_count']}")
    print(f"kept: {manifest['kept_count']}")
    print(f"rejected: {manifest['rejected_count']}")
    print("rejected by reason:")
    for reason, count in manifest["rejected_by_reason"].items():
        print(f"  {reason}: {count}")
    print(f"filtered overlay: {manifest['filtered_overlay_svg']}")
    print(f"kept mask: {manifest['kept_mask_svg']}")
    print(f"report: {manifest['filter_report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
