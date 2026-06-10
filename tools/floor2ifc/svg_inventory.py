#!/usr/bin/env python3
"""Build a compact geometry/style inventory for a floor-plan SVG."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GEOMETRY_TAGS = {"polygon", "polyline", "rect", "path", "line", "circle", "ellipse"}


@dataclass(frozen=True)
class Bounds:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return self.width * self.height


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else default


def parse_points(value: str | None) -> list[tuple[float, float]]:
    if not value:
        return []
    numbers = [float(item) for item in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)]
    return list(zip(numbers[0::2], numbers[1::2]))


def class_names(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"\s+", value.strip()) if item]


def parse_style_attr(style_attr: str | None) -> dict[str, str]:
    styles: dict[str, str] = {}
    if not style_attr:
        return styles
    for declaration in style_attr.split(";"):
        if ":" not in declaration:
            continue
        key, value = declaration.split(":", 1)
        styles[key.strip()] = value.strip()
    return styles


def parse_css_classes(root: ET.Element) -> dict[str, dict[str, str]]:
    css: dict[str, dict[str, str]] = {}
    for elem in root.iter():
        if local_name(elem.tag) != "style":
            continue
        text = "".join(elem.itertext())
        for selector_text, body in re.findall(r"([^{}]+)\{([^}]*)\}", text):
            styles = parse_style_attr(body)
            for selector in selector_text.split(","):
                selector = selector.strip()
                if not selector.startswith("."):
                    continue
                css.setdefault(selector[1:], {}).update(styles)
    return css


def resolved_style(elem: ET.Element, css_classes: dict[str, dict[str, str]]) -> dict[str, str]:
    style: dict[str, str] = {}
    for cls in class_names(elem.attrib.get("class")):
        style.update(css_classes.get(cls, {}))
    style.update(parse_style_attr(elem.attrib.get("style")))
    for attr in ("fill", "stroke", "stroke-width", "opacity", "display", "visibility"):
        if attr in elem.attrib:
            style[attr] = elem.attrib[attr]
    return style


def rough_path_points(d_attr: str | None) -> list[tuple[float, float]]:
    if not d_attr:
        return []
    tokens = re.findall(
        r"[MmZzLlHhVvCcSsQqTtAa]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        d_attr,
    )
    param_counts = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}
    points: list[tuple[float, float]] = []
    index = 0
    command = ""
    current = (0.0, 0.0)
    subpath_start = (0.0, 0.0)

    def is_command(token: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z]", token))

    def read_number() -> float | None:
        nonlocal index
        if index >= len(tokens) or is_command(tokens[index]):
            return None
        value = float(tokens[index])
        index += 1
        return value

    def point_from(x: float, y: float, relative: bool) -> tuple[float, float]:
        if relative:
            return current[0] + x, current[1] + y
        return x, y

    while index < len(tokens):
        token = tokens[index]
        if is_command(token):
            command = token
            index += 1
        elif not command:
            break

        upper = command.upper()
        relative = command.islower()
        if upper == "Z":
            current = subpath_start
            points.append(current)
            command = ""
            continue

        count = param_counts.get(upper)
        if count is None:
            break

        first_move = upper == "M"
        while index < len(tokens) and not is_command(tokens[index]):
            values: list[float] = []
            for _ in range(count):
                number = read_number()
                if number is None:
                    break
                values.append(number)
            if len(values) != count:
                break

            if upper in {"M", "L", "T"}:
                current = point_from(values[0], values[1], relative)
                points.append(current)
                if first_move:
                    subpath_start = current
                    first_move = False
                    upper = "L"
                    command = "l" if relative else "L"
                    count = 2
            elif upper == "H":
                current = (current[0] + values[0] if relative else values[0], current[1])
                points.append(current)
            elif upper == "V":
                current = (current[0], current[1] + values[0] if relative else values[0])
                points.append(current)
            elif upper == "C":
                points.extend(
                    [
                        point_from(values[0], values[1], relative),
                        point_from(values[2], values[3], relative),
                    ]
                )
                current = point_from(values[4], values[5], relative)
                points.append(current)
            elif upper in {"S", "Q"}:
                points.append(point_from(values[0], values[1], relative))
                current = point_from(values[2], values[3], relative)
                points.append(current)
            elif upper == "A":
                current = point_from(values[5], values[6], relative)
                points.append(current)
    return points


def bounds_from_points(points: list[tuple[float, float]]) -> Bounds | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return Bounds(min(xs), min(ys), max(xs), max(ys))


def element_bounds(elem: ET.Element) -> Bounds | None:
    tag = local_name(elem.tag)
    if tag == "rect":
        x = parse_float(elem.attrib.get("x"))
        y = parse_float(elem.attrib.get("y"))
        return Bounds(x, y, x + parse_float(elem.attrib.get("width")), y + parse_float(elem.attrib.get("height")))
    if tag == "line":
        x1 = parse_float(elem.attrib.get("x1"))
        y1 = parse_float(elem.attrib.get("y1"))
        x2 = parse_float(elem.attrib.get("x2"))
        y2 = parse_float(elem.attrib.get("y2"))
        return Bounds(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
    if tag in {"polygon", "polyline"}:
        return bounds_from_points(parse_points(elem.attrib.get("points")))
    if tag == "circle":
        cx = parse_float(elem.attrib.get("cx"))
        cy = parse_float(elem.attrib.get("cy"))
        r = parse_float(elem.attrib.get("r"))
        return Bounds(cx - r, cy - r, cx + r, cy + r)
    if tag == "ellipse":
        cx = parse_float(elem.attrib.get("cx"))
        cy = parse_float(elem.attrib.get("cy"))
        rx = parse_float(elem.attrib.get("rx"))
        ry = parse_float(elem.attrib.get("ry"))
        return Bounds(cx - rx, cy - ry, cx + rx, cy + ry)
    if tag == "path":
        return bounds_from_points(rough_path_points(elem.attrib.get("d")))
    return None


def estimate_length(tag: str, bounds: Bounds, points: list[tuple[float, float]]) -> float:
    if tag == "line" and len(points) >= 2:
        return math.dist(points[0], points[1])
    if tag in {"polygon", "polyline"} and len(points) >= 2:
        return sum(math.dist(points[index - 1], points[index]) for index in range(1, len(points)))
    return max(bounds.width, bounds.height)


def points_for(elem: ET.Element, bounds: Bounds) -> list[tuple[float, float]]:
    tag = local_name(elem.tag)
    if tag in {"polygon", "polyline"}:
        return parse_points(elem.attrib.get("points"))
    if tag == "line":
        return [
            (parse_float(elem.attrib.get("x1")), parse_float(elem.attrib.get("y1"))),
            (parse_float(elem.attrib.get("x2")), parse_float(elem.attrib.get("y2"))),
        ]
    if tag == "rect":
        return [
            (bounds.min_x, bounds.min_y),
            (bounds.max_x, bounds.min_y),
            (bounds.max_x, bounds.max_y),
            (bounds.min_x, bounds.max_y),
        ]
    if tag == "path":
        return rough_path_points(elem.attrib.get("d"))
    return []


def build_inventory(svg_path: Path) -> dict[str, Any]:
    root = ET.parse(svg_path).getroot()
    css = parse_css_classes(root)
    elements: list[dict[str, Any]] = []
    style_counts: Counter[tuple[str, str, str, str, str]] = Counter()

    for elem in root.iter():
        tag = local_name(elem.tag)
        if tag not in GEOMETRY_TAGS:
            continue
        bounds = element_bounds(elem)
        if bounds is None:
            continue
        style = resolved_style(elem, css)
        classes = class_names(elem.attrib.get("class"))
        class_key = " ".join(classes) or "(none)"
        fill = style.get("fill", "") or "(default)"
        stroke = style.get("stroke", "") or "(none)"
        stroke_width = style.get("stroke-width", "") or "(none)"
        points = points_for(elem, bounds)
        length = estimate_length(tag, bounds, points)
        thickness = min(bounds.width, bounds.height)
        aspect_ratio = max(bounds.width, bounds.height) / max(thickness, 1e-6)

        style_counts[(tag, class_key, fill, stroke, stroke_width)] += 1
        elements.append(
            {
                "id": f"e{len(elements) + 1:05d}",
                "tag": tag,
                "class": classes,
                "style": {
                    key: style[key]
                    for key in ("fill", "stroke", "stroke-width", "opacity", "display", "visibility")
                    if key in style
                },
                "bbox": asdict(bounds),
                "center": [
                    round((bounds.min_x + bounds.max_x) / 2, 3),
                    round((bounds.min_y + bounds.max_y) / 2, 3),
                ],
                "width": round(bounds.width, 3),
                "height": round(bounds.height, 3),
                "area": round(bounds.area, 3),
                "length_estimate": round(length, 3),
                "aspect_ratio": round(aspect_ratio, 3),
                "transform": elem.attrib.get("transform"),
            }
        )

    return {
        "source_svg": str(svg_path),
        "viewBox": root.attrib.get("viewBox"),
        "element_count": len(elements),
        "elements": elements,
        "style_summary": [
            {
                "count": count,
                "tag": key[0],
                "class": key[1],
                "fill": key[2],
                "stroke": key[3],
                "stroke_width": key[4],
            }
            for key, count in style_counts.most_common()
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("svg", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.svg.exists():
        print(f"SVG not found: {args.svg}", file=sys.stderr)
        return 2

    data = build_inventory(args.svg)
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
