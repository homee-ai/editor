#!/usr/bin/env python3
"""Generate SVG VLM candidates from a PNG using a CubiCasa wall/railing mask.

The script traces the PNG with the same colour-cutout vtracer settings as
png2svg.py, then keeps SVG elements whose rasterized shape overlaps the semantic
Wall/Railing mask. It writes a red overlay SVG for tuning the candidate set.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from svg_inventory import GEOMETRY_TAGS, build_inventory, element_bounds, local_name, parse_float, points_for


DEFAULT_OUT_DIR = Path(__file__).parent / "out/png-mask-svg-candidates"
SVG_NS = "http://www.w3.org/2000/svg"
OVERLAY_GROUP_ID = "floor2ifc-mask-filter-kept-overlay"
MASK_GROUP_ID = "floor2ifc-mask-filter-kept-mask"
DEFAULT_CLASSES = ("Wall", "Railing")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def class_key(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def mask_from_image(path: Path, target_size: tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    if mask.size != target_size:
        mask = mask.resize(target_size, Image.Resampling.NEAREST)
    return mask.point(lambda value: 255 if value > 0 else 0)


def combine_masks(mask_paths: list[Path], target_size: tuple[int, int]) -> Image.Image:
    combined = Image.new("L", target_size, 0)
    for path in mask_paths:
        combined = ImageChops.lighter(combined, mask_from_image(path, target_size))
    return combined


def resolve_tags_path(mask_path: Path) -> Path | None:
    if mask_path.is_dir():
        tags = sorted(mask_path.glob("*_tags.json"))
        return tags[0] if tags else None
    tags = mask_path.with_name(mask_path.name.replace("_wall_mask.png", "_tags.json"))
    if tags.exists():
        return tags
    tags = sorted(mask_path.parent.glob("*_tags.json"))
    return tags[0] if tags else None


def mask_paths_from_tags(tags_path: Path, class_names: set[str]) -> list[Path]:
    data = json.loads(tags_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for item in data.get("rooms", {}).get("present", []):
        name = str(item.get("name") or "")
        if class_key(name) not in class_names:
            continue
        rel = item.get("mask")
        if not isinstance(rel, str):
            continue
        path = tags_path.parent / rel
        if path.exists():
            paths.append(path)
    return paths


def resolve_semantic_mask(mask_path: Path, png_size: tuple[int, int], class_names: set[str]) -> tuple[Image.Image, list[Path]]:
    tags_path = resolve_tags_path(mask_path)
    mask_paths: list[Path] = []
    if tags_path is not None:
        mask_paths = mask_paths_from_tags(tags_path, class_names)

    if not mask_paths:
        if mask_path.is_dir():
            for name in sorted(class_names):
                mask_paths.extend(sorted(mask_path.glob(f"masks/*_{name}.png")))
                mask_paths.extend(sorted(mask_path.glob(f"*_{name}.png")))
        elif mask_path.exists():
            mask_paths = [mask_path]

    if not mask_paths:
        raise FileNotFoundError(f"No semantic mask found for classes: {', '.join(sorted(class_names))}")
    return combine_masks(mask_paths, png_size), mask_paths


def dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    size = radius * 2 + 1
    return mask.filter(ImageFilter.MaxFilter(size))


def bbox_pixels(bbox: dict[str, Any], image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    width, height = image_size
    left = max(0, min(width, int(float(bbox.get("min_x", 0)))))
    top = max(0, min(height, int(float(bbox.get("min_y", 0)))))
    right = max(0, min(width, int(float(bbox.get("max_x", 0)) + 0.999)))
    bottom = max(0, min(height, int(float(bbox.get("max_y", 0)) + 0.999)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def element_points(elem: ET.Element, element: dict[str, Any], offset: tuple[int, int]) -> list[tuple[float, float]]:
    bounds = element_bounds(elem)
    if bounds is None:
        return []
    ox, oy = offset
    return [(x - ox, y - oy) for x, y in points_for(elem, bounds)]


def element_mask(elem: ET.Element, element: dict[str, Any], image_size: tuple[int, int]) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    box = bbox_pixels(element.get("bbox", {}), image_size)
    if box is None:
        return None

    left, top, right, bottom = box
    pad = max(2, int(float(element.get("effective_stroke_width") or 0)) + 2)
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(image_size[0], right + pad)
    bottom = min(image_size[1], bottom + pad)
    box = (left, top, right, bottom)
    mask = Image.new("L", (right - left, bottom - top), 0)
    draw = ImageDraw.Draw(mask)

    tag = local_name(elem.tag)
    points = element_points(elem, element, (left, top))
    stroke_width = max(1, round(float(element.get("effective_stroke_width") or parse_float(elem.attrib.get("stroke-width"), 0) or 1)))
    stroke_only = is_stroke_only(elem, element)

    if tag == "rect":
        x = parse_float(elem.attrib.get("x")) - left
        y = parse_float(elem.attrib.get("y")) - top
        w = parse_float(elem.attrib.get("width"))
        h = parse_float(elem.attrib.get("height"))
        draw.rectangle([x, y, x + w, y + h], fill=255)
    elif tag == "line" and len(points) >= 2:
        draw.line(points[:2], fill=255, width=stroke_width)
    elif tag == "polyline" and len(points) >= 2:
        draw.line(points, fill=255, width=stroke_width, joint="curve")
    elif tag == "polygon" and len(points) >= 3:
        draw.polygon(points, fill=255)
    elif tag == "circle":
        cx = parse_float(elem.attrib.get("cx")) - left
        cy = parse_float(elem.attrib.get("cy")) - top
        r = parse_float(elem.attrib.get("r"))
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    elif tag == "ellipse":
        cx = parse_float(elem.attrib.get("cx")) - left
        cy = parse_float(elem.attrib.get("cy")) - top
        rx = parse_float(elem.attrib.get("rx"))
        ry = parse_float(elem.attrib.get("ry"))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    elif tag == "path" and len(points) >= 2:
        if stroke_only:
            draw.line(points, fill=255, width=stroke_width, joint="curve")
        elif len(points) >= 3:
            draw.polygon(points, fill=255)
        else:
            draw.line(points, fill=255, width=stroke_width)

    return mask, box


def mask_stats_for_element(
    elem: ET.Element | None,
    element: dict[str, Any],
    semantic_mask: Image.Image,
) -> dict[str, Any]:
    if elem is None:
        return {"overlap_pixels": 0, "element_pixels": 0, "element_overlap": 0.0}
    rendered = element_mask(elem, element, semantic_mask.size)
    if rendered is None:
        return {"overlap_pixels": 0, "element_pixels": 0, "element_overlap": 0.0}

    element_image, box = rendered
    semantic_crop = semantic_mask.crop(box)
    overlap_image = ImageChops.multiply(element_image, semantic_crop)
    element_pixels = sum(element_image.histogram()[1:])
    overlap_pixels = sum(overlap_image.histogram()[1:])
    return {
        "overlap_pixels": overlap_pixels,
        "element_pixels": element_pixels,
        "element_overlap": overlap_pixels / max(element_pixels, 1),
    }


def keep_element(element: dict[str, Any], stats: dict[str, Any], args: argparse.Namespace) -> tuple[bool, str]:
    area = float(element.get("area") or 0)
    width = float(element.get("width") or 0)
    height = float(element.get("height") or 0)
    thickness = min(width, height)
    aspect = float(element.get("aspect_ratio") or 0)
    overlap_pixels = int(stats["overlap_pixels"])
    ratio = float(stats["element_overlap"])

    if area < args.min_area:
        return False, "small_area"
    if overlap_pixels < args.min_overlap_pixels:
        return False, "low_overlap_pixels"
    if ratio >= args.min_element_overlap:
        return True, "element_overlap"
    if aspect >= args.thin_aspect and thickness <= args.thin_max_thickness and overlap_pixels >= args.thin_min_overlap_pixels:
        return True, "thin_mask_hit"
    if overlap_pixels >= args.large_overlap_pixels and ratio >= args.large_min_element_overlap:
        return True, "large_mask_hit"
    return False, "low_element_overlap"


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


def color_token(value: Any, missing: str) -> str:
    text = str(value or "").strip().lower().replace(" ", "")
    if not text:
        return missing
    if text in {"none", "transparent"}:
        return "none"
    return text


def ensure_stroke_width(elem: ET.Element, fallback: float) -> None:
    if parse_float(elem.attrib.get("stroke-width"), 0) <= 0:
        elem.attrib["stroke-width"] = f"{fallback:g}"


def is_stroke_only(elem: ET.Element, item: dict[str, Any]) -> bool:
    if local_name(elem.tag) in {"line", "polyline"}:
        return True
    style = item.get("style", {})
    fill = color_token(style.get("fill"), "default")
    stroke = color_token(style.get("stroke"), "none")
    return local_name(elem.tag) == "path" and fill == "none" and stroke != "none"


def overlay_element(elem: ET.Element, item: dict[str, Any], opacity: float) -> str:
    clone = deepcopy(elem)
    clone.attrib.pop("class", None)
    if is_stroke_only(elem, item):
        clone.attrib["fill"] = "none"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = f"{opacity:g}"
        ensure_stroke_width(clone, fallback=5)
    else:
        clone.attrib["fill"] = "#ff2d55"
        clone.attrib["fill-opacity"] = f"{opacity:g}"
        clone.attrib["stroke"] = "#ff2d55"
        clone.attrib["stroke-opacity"] = "0.95"
        ensure_stroke_width(clone, fallback=1)
    text = ET.tostring(clone, encoding="unicode")
    return text.replace(f' xmlns="{SVG_NS}"', "", 1)


def mask_element(elem: ET.Element, item: dict[str, Any]) -> str:
    clone = deepcopy(elem)
    clone.attrib.pop("class", None)
    clone.attrib["fill"] = "#000"
    clone.attrib["fill-opacity"] = "1"
    clone.attrib["stroke"] = "#000"
    clone.attrib["stroke-opacity"] = "1"
    if is_stroke_only(elem, item):
        clone.attrib["fill"] = "none"
        ensure_stroke_width(clone, fallback=5)
    else:
        clone.attrib["stroke-width"] = "0"
    text = ET.tostring(clone, encoding="unicode")
    return text.replace(f' xmlns="{SVG_NS}"', "", 1)


def render_outputs(
    svg_path: Path,
    kept: list[dict[str, Any]],
    out_dir: Path,
    opacity: float,
    show_ids: bool,
) -> tuple[Path, Path]:
    root, by_id = index_source_elements(svg_path)
    view_box = root.attrib.get("viewBox")
    if view_box:
        svg_open = f'<svg xmlns="{SVG_NS}" viewBox="{escape(view_box)}">'
    else:
        width = root.attrib.get("width", "100%")
        height = root.attrib.get("height", "100%")
        svg_open = f'<svg xmlns="{SVG_NS}" width="{escape(width)}" height="{escape(height)}">'

    overlay_parts = [
        f'\n<g id="{OVERLAY_GROUP_ID}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700">',
    ]
    mask_parts = [svg_open, f'<g id="{MASK_GROUP_ID}">']
    for item in kept:
        elem = by_id.get(item["id"])
        if elem is None:
            continue
        overlay_parts.append(overlay_element(elem, item, opacity))
        mask_parts.append(mask_element(elem, item))
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

    overlay_path = out_dir / "candidate_overlay.svg"
    kept_mask_path = out_dir / "candidate_mask.svg"
    original = svg_path.read_text(encoding="utf-8", errors="ignore")
    overlay_path.write_text(svg_close_insert(original, "\n".join(overlay_parts)), encoding="utf-8")
    kept_mask_path.write_text("\n".join(mask_parts) + "\n", encoding="utf-8")
    return overlay_path, kept_mask_path


def run(png_path: Path, mask_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_size = Image.open(png_path).size
    semantic_mask, semantic_sources = resolve_semantic_mask(mask_path, image_size, {class_key(item) for item in args.classes})
    semantic_mask = dilate_mask(semantic_mask, args.mask_dilate)
    semantic_mask_path = out_dir / "semantic_filter_mask.png"
    semantic_mask.save(semantic_mask_path)

    svg_path = out_dir / f"{png_path.stem}.svg"
    if args.svg is not None:
        svg_path.write_text(args.svg.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    else:
        from png2svg import convert

        with tempfile.TemporaryDirectory(prefix="floor2ifc-mask-svg-") as tmp:
            convert(png_path, svg_path, False, args.filter_speckle, args.color_precision, Path(tmp))

    inventory = build_inventory(svg_path)
    _root, by_id = index_source_elements(svg_path)
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for element in inventory["elements"]:
        stats = mask_stats_for_element(by_id.get(element["id"]), element, semantic_mask)
        keep, reason = keep_element(element, stats, args)
        row = {
            **element,
            "overlap_pixels": stats["overlap_pixels"],
            "element_pixels": stats["element_pixels"],
            "element_overlap": round(float(stats["element_overlap"]), 6),
            "selection_reason": reason,
        }
        if keep:
            kept.append(row)
        else:
            rejected.append({"id": element["id"], "reason": reason, **stats})
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    overlay_path, kept_mask_path = render_outputs(svg_path, kept, out_dir, args.overlay_opacity, args.show_ids)
    report = {
        "png": str(png_path),
        "mask_input": str(mask_path),
        "semantic_mask_sources": [str(path) for path in semantic_sources],
        "semantic_filter_mask": str(semantic_mask_path),
        "generated_svg": str(svg_path),
        "element_count": inventory["element_count"],
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "rejected_by_reason": dict(sorted(reason_counts.items())),
        "classes": args.classes,
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"png", "mask", "out_dir", "classes"}
        },
        "kept_ids": [item["id"] for item in kept],
        "kept": kept,
        "rejected": rejected,
        "candidate_overlay_svg": str(overlay_path),
        "candidate_mask_svg": str(kept_mask_path),
    }
    report_path = out_dir / "candidate_report.json"
    write_json(report_path, report)
    return {**report, "report": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", type=Path)
    parser.add_argument("mask", type=Path, help="CubiCasa mask file or output directory for this PNG.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--svg", type=Path, default=None, help="Use an existing png2svg output instead of tracing the PNG.")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES), help="Semantic room classes to use as filter mask.")
    parser.add_argument("--mask-dilate", type=int, default=1, help="Dilate semantic mask by this many pixels before overlap.")
    parser.add_argument("--min-area", type=float, default=80)
    parser.add_argument("--min-overlap-pixels", type=int, default=6)
    parser.add_argument("--min-element-overlap", type=float, default=0.08)
    parser.add_argument("--thin-aspect", type=float, default=5.0)
    parser.add_argument("--thin-max-thickness", type=float, default=18)
    parser.add_argument("--thin-min-overlap-pixels", type=int, default=3)
    parser.add_argument("--large-overlap-pixels", type=int, default=120)
    parser.add_argument("--large-min-element-overlap", type=float, default=0.02)
    parser.add_argument("--filter-speckle", type=int, default=4)
    parser.add_argument("--color-precision", type=int, default=6)
    parser.add_argument("--overlay-opacity", type=float, default=0.42)
    parser.add_argument("--show-ids", action="store_true")
    args = parser.parse_args()

    if not args.png.exists():
        print(f"PNG not found: {args.png}")
        return 2
    if not args.mask.exists():
        print(f"Mask path not found: {args.mask}")
        return 2
    if args.svg is not None and not args.svg.exists():
        print(f"SVG not found: {args.svg}")
        return 2

    out_dir = args.out_dir / args.png.stem
    manifest = run(args.png, args.mask, out_dir, args)
    print(f"elements: {manifest['element_count']}")
    print(f"kept: {manifest['kept_count']}")
    print(f"rejected: {manifest['rejected_count']}")
    print("rejected by reason:")
    for reason, count in manifest["rejected_by_reason"].items():
        print(f"  {reason}: {count}")
    print(f"generated svg: {manifest['generated_svg']}")
    print(f"semantic mask: {manifest['semantic_filter_mask']}")
    print(f"candidate overlay: {manifest['candidate_overlay_svg']}")
    print(f"candidate mask: {manifest['candidate_mask_svg']}")
    print(f"report: {manifest['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
