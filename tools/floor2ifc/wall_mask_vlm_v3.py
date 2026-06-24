#!/usr/bin/env python3
"""v3 wall labeling: colour-coded candidate groups judged per-colour.

Instead of overlaying per-object id text (which collides when objects are
dense), v3 fills each candidate with one of a small palette of well-separated,
nameable colours and asks the VLM to bucket each COLOUR as wall / non_wall /
mixed:

  - wall      -> every shape of that colour is kept as wall.
  - non_wall  -> every shape of that colour is dropped.
  - mixed     -> that colour's shapes are re-coloured and re-asked (recurse),
                 until a group is <= the palette size (one colour per object).

Colours are reused across non-adjacent objects (adjacency-distinct greedy
colouring), so a colour is a transient group label, not an id. We hold the
colour->ids map each round, so the VLM only ever names colours.

Streams (case1 / case2 / case3 from png_mask_svg_candidates.py) are processed
independently. Each image holds at most --batch-size objects, chunked spatially.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from svg_inventory import build_inventory, local_name, parse_float
from vlm_client import REPO_ROOT, VlmClient
from wall_mask_vlm import (
    SVG_NS,
    index_source_elements,
    is_stroke_only,
    parse_vlm_json,
    svg_close_insert,
    write_json,
)
from wall_mask_vlm_v2 import load_candidate_report, resolve_rows

DEFAULT_OUT_DIR = Path(__file__).parent / "out/wall-mask-vlm-v3"

# 8 well-separated, nameable colours (core). Extended 4 appended for >8 groups.
PALETTE: list[tuple[str, str]] = [
    ("Red", "#E6194B"),
    ("Orange", "#F58231"),
    ("Yellow", "#FFD500"),
    ("Green", "#3CB44B"),
    ("Cyan", "#42D4F4"),
    ("Blue", "#4363D8"),
    ("Purple", "#911EB4"),
    ("Magenta", "#F032E6"),
    ("Brown", "#9A6324"),
    ("Teal", "#469990"),
    ("Lime", "#BFEF45"),
    ("Pink", "#FF8FB1"),
]


# --------------------------------------------------------------------------- #
# Colour assignment + spatial batching
# --------------------------------------------------------------------------- #
def bbox_gap(a: dict[str, Any], b: dict[str, Any]) -> float:
    dx = max(0.0, max(float(a["min_x"]), float(b["min_x"])) - min(float(a["max_x"]), float(b["max_x"])))
    dy = max(0.0, max(float(a["min_y"]), float(b["min_y"])) - min(float(a["max_y"]), float(b["max_y"])))
    return (dx * dx + dy * dy) ** 0.5


def assign_colors(objects: list[dict[str, Any]], num_colors: int, near_px: float) -> dict[str, int]:
    """id -> palette index. Unique when small; else adjacency-distinct greedy."""
    n = len(objects)
    if n <= num_colors:
        return {obj["id"]: i for i, obj in enumerate(objects)}

    boxes = [obj["bbox"] for obj in objects]
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if bbox_gap(boxes[i], boxes[j]) <= near_px:
                adjacency[i].add(j)
                adjacency[j].add(i)

    color_index: dict[int, int] = {}
    usage = [0] * num_colors
    for i in sorted(range(n), key=lambda k: -len(adjacency[k])):
        blocked = {color_index[j] for j in adjacency[i] if j in color_index}
        allowed = [c for c in range(num_colors) if c not in blocked] or list(range(num_colors))
        # adjacency-distinct, but spread usage evenly so each colour stays a small group
        chosen = min(allowed, key=lambda c: (usage[c], c))
        color_index[i] = chosen
        usage[chosen] += 1
    return {objects[i]["id"]: color_index[i] for i in range(n)}


def spatial_batches(objects: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    if len(objects) <= batch_size:
        return [objects]
    ordered = sorted(objects, key=lambda o: (o["center"][1], o["center"][0]))
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def hex_to_rgb(value: Any) -> tuple[int, int, int] | None:
    text = str(value or "").strip()
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", text)
    if m:
        h = m.group(1)
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    m = re.fullmatch(r"#?([0-9a-fA-F]{3})", text)
    if m:
        h = m.group(1)
        return int(h[0] * 2, 16), int(h[1] * 2, 16), int(h[2] * 2, 16)
    return None


def representative_rgb(obj: dict[str, Any]) -> tuple[int, int, int]:
    style = obj.get("style", {})
    for key in ("fill", "stroke"):
        value = style.get(key)
        if value and str(value).strip().lower() not in ("none", "transparent", ""):
            rgb = hex_to_rgb(value)
            if rgb is not None:
                return rgb
    return (0, 0, 0)


def cluster_fills(objects: list[dict[str, Any]], num_colors: int) -> dict[str, int]:
    """id -> palette index, grouping objects with similar SVG fill RGB.

    Agglomerative: start each object its own cluster, merge the two closest
    (RGB Euclidean) until <= num_colors remain. Palette indices ordered by
    luminance for stable output.
    """
    n = len(objects)
    if n <= num_colors:
        return {obj["id"]: i for i, obj in enumerate(objects)}

    # cluster = [centroid(list rgb), count, member object-indices]
    clusters = [[list(representative_rgb(obj)), 1, [i]] for i, obj in enumerate(objects)]

    def distance(a: list[float], b: list[float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    while len(clusters) > num_colors:
        best = None
        pair = (0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = distance(clusters[a][0], clusters[b][0])
                if best is None or d < best:
                    best = d
                    pair = (a, b)
        a, b = pair
        ca, cb = clusters[a], clusters[b]
        total = ca[1] + cb[1]
        centroid = [(ca[0][k] * ca[1] + cb[0][k] * cb[1]) / total for k in range(3)]
        merged = [centroid, total, ca[2] + cb[2]]
        clusters = [clusters[i] for i in range(len(clusters)) if i not in (a, b)]
        clusters.append(merged)

    clusters.sort(key=lambda c: 0.299 * c[0][0] + 0.587 * c[0][1] + 0.114 * c[0][2])
    idx_map: dict[str, int] = {}
    for color_index, cluster in enumerate(clusters):
        for member in cluster[2]:
            idx_map[objects[member]["id"]] = color_index
    return idx_map


# --------------------------------------------------------------------------- #
# Rendering (colour fill, no id text)
# --------------------------------------------------------------------------- #
def canvas_of(svg_path: Path) -> tuple[str, float, float]:
    root = ET.parse(svg_path).getroot()
    view_box = root.attrib.get("viewBox")
    if view_box:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", view_box)]
        if len(nums) == 4:
            return f'viewBox="{escape(view_box)}"', nums[2], nums[3]
    width = parse_float(root.attrib.get("width"), 1000)
    height = parse_float(root.attrib.get("height"), 1000)
    return f'width="{width:g}" height="{height:g}"', width, height


def colored_element(elem: ET.Element, hex_color: str, opacity: float) -> str:
    clone = deepcopy(elem)
    clone.attrib.pop("class", None)
    if is_stroke_only(elem, None):
        clone.attrib["fill"] = "none"
        clone.attrib["stroke"] = hex_color
        clone.attrib["stroke-opacity"] = f"{opacity:g}"
        if parse_float(clone.attrib.get("stroke-width"), 0) <= 0:
            clone.attrib["stroke-width"] = "6"
    else:
        clone.attrib["fill"] = hex_color
        clone.attrib["fill-opacity"] = f"{opacity:g}"
        clone.attrib["stroke"] = "#1a1a1a"
        clone.attrib["stroke-opacity"] = "0.9"
        if parse_float(clone.attrib.get("stroke-width"), 0) <= 0:
            clone.attrib["stroke-width"] = "1"
    text = ET.tostring(clone, encoding="unicode")
    return re.sub(r'\sxmlns="[^"]+"', "", text, count=1)


def render_colored_mask(svg_path: Path, by_id: dict[str, ET.Element], idx_map: dict[str, int], out_svg: Path) -> None:
    open_attr, width, height = canvas_of(svg_path)
    parts = [
        f'<svg xmlns="{SVG_NS}" {open_attr}>',
        f'<rect x="0" y="0" width="{width:g}" height="{height:g}" fill="#ffffff"/>',
        "<g>",
    ]
    for oid, idx in idx_map.items():
        elem = by_id.get(oid)
        if elem is not None:
            parts.append(colored_element(elem, PALETTE[idx][1], 1.0))
    parts.extend(["</g>", "</svg>"])
    out_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_colored_overlay(svg_path: Path, by_id: dict[str, ET.Element], idx_map: dict[str, int], out_svg: Path) -> None:
    original = svg_path.read_text(encoding="utf-8", errors="ignore")
    parts = ['\n<g id="floor2ifc-v3-colored" opacity="0.8">']
    for oid, idx in idx_map.items():
        elem = by_id.get(oid)
        if elem is not None:
            parts.append(colored_element(elem, PALETTE[idx][1], 0.85))
    parts.append("</g>")
    out_svg.write_text(svg_close_insert(original, "\n".join(parts)), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Prompt + verdict
# --------------------------------------------------------------------------- #
def build_v3_prompt(present_colors: list[str]) -> str:
    colors = ", ".join(present_colors)
    return f"""You are classifying candidate shapes in a floor-plan by COLOUR.

You will see THREE images:
1. Clean floor-plan image.
2. The candidate shapes, each filled with a colour, on a white background.
3. The same coloured shapes overlaid on the floor plan (spatial context).

Colours present in this batch: {colors}.
Several shapes may share the same colour. Judge each colour as a whole.

For EVERY colour above, decide:
- wall: every shape of that colour is a structural wall.
- non_wall: every shape of that colour is NOT a wall.
- mixed: shapes of that colour are some wall and some not.

What counts as a wall:
- Exterior/interior walls, wall outlines, room-dividing boundaries, and short wall stubs.
- ANY straight segment that lies on / along the wall line is a wall. This includes windows set into a wall, door-opening lines, door headers, window sills, jambs, and thresholds. If a thin straight shape sits inside the wall band, it is wall — keep it (it keeps the wall continuous).
- The ONLY door/window parts that are NOT walls are the curved door swing arc (the quarter-circle sweep) and a clearly free-standing door leaf/panel drawn out in the room (not on the wall line).
- Also NOT walls: furniture, fixtures, room fills, text/labels, dimension arrows, stairs, page borders, logos, decorative details.

Output rules:
- Return ONLY one minified JSON object. No markdown, no prose.

JSON shape:
{{"wall_colors":[],"non_wall_colors":[],"mixed_colors":[],"reason":"short visual reason","confidence":0.0}}

- Use only colour names from: {colors}.
- Every present colour MUST appear in exactly one of the three lists.
"""


def classify_colors(review: dict[str, Any], present: list[str]) -> tuple[set[str], set[str], set[str]]:
    lower_to_name = {name.lower(): name for name in present}

    def bucket(key: str) -> set[str]:
        result: set[str] = set()
        for item in review.get(key, []) or []:
            if isinstance(item, str) and item.strip().lower() in lower_to_name:
                result.add(lower_to_name[item.strip().lower()])
        return result

    wall = bucket("wall_colors")
    non_wall = bucket("non_wall_colors")
    mixed = bucket("mixed_colors")

    # A colour in more than one bucket is ambiguous -> treat as mixed.
    duplicates = (wall & non_wall) | (wall & mixed) | (non_wall & mixed)
    wall -= duplicates
    non_wall -= duplicates
    mixed |= duplicates

    # Any present colour the VLM omitted -> mixed (re-judged next round).
    for name in present:
        if name not in wall and name not in non_wall and name not in mixed:
            mixed.add(name)
    return wall, non_wall, mixed


# --------------------------------------------------------------------------- #
# Stream processing (BFS colour recursion)
# --------------------------------------------------------------------------- #
def process_stream(
    stream_name: str,
    objects: list[dict[str, Any]],
    svg_path: Path,
    by_id: dict[str, ET.Element],
    groups_dir: Path,
    client: VlmClient | None,
    args: argparse.Namespace,
    counter: list[int],
) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    logs: list[dict[str, Any]] = []
    if not objects:
        return selected, logs

    num_colors = min(args.max_colors, len(PALETTE))

    def cap_for(depth: int) -> int:
        return args.l0_batch_size if depth == 0 else args.batch_size

    queue: deque[tuple[list[dict[str, Any]], int]] = deque(
        (batch, 0) for batch in spatial_batches(objects, cap_for(0))
    )

    while queue:
        group, depth = queue.popleft()
        if not group:
            continue
        if len(group) > cap_for(depth):
            queue.extendleft((batch, depth) for batch in reversed(spatial_batches(group, cap_for(depth))))
            continue

        # Level 0: group by SVG-fill RGB similarity. Deeper: adjacency-distinct.
        if depth == 0:
            idx_map = cluster_fills(group, num_colors)
        else:
            idx_map = assign_colors(group, num_colors, args.adjacency_px)
        color_to_ids: dict[str, list[str]] = {}
        for oid, idx in idx_map.items():
            color_to_ids.setdefault(PALETTE[idx][0], []).append(oid)
        present = [PALETTE[i][0] for i in range(num_colors) if PALETTE[i][0] in color_to_ids]

        counter[0] += 1
        call_dir = groups_dir / f"{counter[0]:03d}_{stream_name}_d{depth}"
        call_dir.mkdir(parents=True, exist_ok=True)
        mask_svg = call_dir / "candidate_mask.svg"
        overlay_svg = call_dir / "candidate_overlay.svg"
        prompt_path = call_dir / "prompt.txt"
        raw_path = call_dir / "vlm_raw.txt"

        render_colored_mask(svg_path, by_id, idx_map, mask_svg)
        render_colored_overlay(svg_path, by_id, idx_map, overlay_svg)
        prompt = build_v3_prompt(present)
        prompt_path.write_text(prompt, encoding="utf-8")

        if args.prepare_only or client is None:
            logs.append({"call": counter[0], "stream": stream_name, "depth": depth,
                         "group_size": len(group), "present_colors": present, "status": "prepared"})
            continue

        call_start = time.time()
        raw = client.chat_with_svgs(
            prompt, [svg_path, mask_svg, overlay_svg],
            image_size=args.image_size, dump_request=args.dump_request, max_tokens=args.max_output_tokens,
        )
        elapsed = time.time() - call_start
        raw_path.write_text(raw, encoding="utf-8")
        try:
            review = parse_vlm_json(raw, raw_path)
        except ValueError:
            review = {"wall_colors": [], "non_wall_colors": [], "mixed_colors": present, "reason": "unparseable"}
        write_json(call_dir / "verdict.json", review)
        print(f"    call {counter[0]:>3} {stream_name} d{depth} n={len(group):<3} -> {elapsed:5.1f}s", flush=True)

        wall_c, non_c, mixed_c = classify_colors(review, present)
        for color in present:
            ids = color_to_ids[color]
            if color in wall_c:
                selected.extend(ids)
            elif color in non_c:
                continue
            else:  # mixed
                objs = [obj for obj in group if obj["id"] in set(ids)]
                if len(objs) <= 1 or depth >= args.max_depth:
                    selected.extend(ids)  # cannot subdivide further -> keep (recall bias)
                else:
                    queue.append((objs, depth + 1))

        logs.append({
            "call": counter[0], "stream": stream_name, "depth": depth, "group_size": len(group),
            "present_colors": present,
            "wall_colors": sorted(wall_c), "non_wall_colors": sorted(non_c), "mixed_colors": sorted(mixed_c),
            "elapsed_sec": round(elapsed, 2),
            "reason": review.get("reason", ""),
        })
    return selected, logs


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def split_streams(report: dict[str, Any], inventory_by_id: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    auto = resolve_rows(report.get("auto_wall"), inventory_by_id)
    kept = resolve_rows(report.get("kept"), inventory_by_id)
    case2 = [row for row in kept if row.get("case") == 2]
    case3 = [row for row in kept if row.get("case") == 3]
    # rows whose case is missing fall back into case3 (treated as straddlers)
    other = [row for row in kept if row.get("case") not in (2, 3)]
    return {"case1": auto, "case2": case2, "case3": case3 + other}


def run_candidate_report(report_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report = load_candidate_report(report_path)
    svg_path = Path(report["generated_svg"])
    if not svg_path.exists():
        fallback = Path(report["_candidate_report_path"]).parent / svg_path.name
        if not fallback.exists():
            raise FileNotFoundError(f"Generated SVG not found: {svg_path}")
        svg_path = fallback

    run_start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(svg_path)
    inventory_by_id = {item["id"]: item for item in inventory["elements"]}
    _root, by_id = index_source_elements(svg_path)
    streams = split_streams(report, inventory_by_id)

    groups_dir = out_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.prepare_only else VlmClient(args.env_file_root)

    counter = [0]
    selected_ids: list[str] = []
    all_logs: list[dict[str, Any]] = []
    stream_summary: dict[str, Any] = {}
    for name in ("case1", "case2", "case3"):
        picks, logs = process_stream(name, streams[name], svg_path, by_id, groups_dir, client, args, counter)
        selected_ids.extend(picks)
        all_logs.extend(logs)
        stream_summary[name] = {"input": len(streams[name]), "selected": len(set(picks))}

    selected_ids = sorted(set(selected_ids))
    all_candidates = [row for rows in streams.values() for row in rows]
    mask_path = out_dir / "wall_mask.svg"
    overlay_path = out_dir / "wall_mask_overlay.svg"
    _render_wall_outputs(svg_path, selected_ids, by_id, mask_path, overlay_path)

    total_elapsed = time.time() - run_start
    vlm_elapsed = round(sum(float(log.get("elapsed_sec") or 0) for log in all_logs), 2)
    manifest = {
        "source_svg": str(svg_path),
        "candidate_report": report["_candidate_report_path"],
        "source_png": report.get("png"),
        "status": "prepared" if args.prepare_only else "completed",
        "vlm_calls": counter[0],
        "elapsed_sec": round(total_elapsed, 2),
        "vlm_elapsed_sec": vlm_elapsed,
        "stream_summary": stream_summary,
        "selected_wall_count": len(selected_ids),
        "selected_wall_ids": selected_ids,
        "wall_mask_svg": str(mask_path),
        "wall_mask_overlay_svg": str(overlay_path),
        "groups_dir": str(groups_dir),
    }
    write_json(out_dir / "result.json", manifest)
    write_json(out_dir / "rounds.json", {"calls": all_logs})
    return manifest


def _render_wall_outputs(svg_path: Path, selected_ids: list[str], by_id: dict[str, ET.Element], mask_svg: Path, overlay_svg: Path) -> None:
    open_attr, _w, _h = canvas_of(svg_path)
    idx_map = {oid: 0 for oid in selected_ids}  # single colour for the final mask
    mask_parts = [f'<svg xmlns="{SVG_NS}" {open_attr}>', '<g fill="#000000" stroke="#000000">']
    overlay_parts = ['\n<g id="floor2ifc-v3-wall" opacity="0.7">']
    for oid in selected_ids:
        elem = by_id.get(oid)
        if elem is None:
            continue
        mask_parts.append(colored_element(elem, "#000000", 1.0))
        overlay_parts.append(colored_element(elem, "#ff2d55", 0.85))
    mask_parts.extend(["</g>", "</svg>"])
    overlay_parts.append("</g>")
    mask_svg.write_text("\n".join(mask_parts) + "\n", encoding="utf-8")
    original = svg_path.read_text(encoding="utf-8", errors="ignore")
    overlay_svg.write_text(svg_close_insert(original, "\n".join(overlay_parts)), encoding="utf-8")


def output_name_for_report(report_path: Path) -> str:
    report = load_candidate_report(report_path)
    for key in ("png", "generated_svg"):
        value = report.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem
    return report_path.parent.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_report", type=Path, nargs="+", help="candidate_report.json or output dir(s).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-file-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--image-size", type=int, default=1800)
    parser.add_argument("--l0-batch-size", type=int, default=64, help="Max candidates per image at level 0 (RGB-clustered).")
    parser.add_argument("--batch-size", type=int, default=32, help="Max candidates per image at deeper levels (adjacency-coloured).")
    parser.add_argument("--max-colors", type=int, default=8, help="Colours per image (<= palette size).")
    parser.add_argument("--adjacency-px", type=float, default=24.0, help="bbox gap under which two objects are adjacent (deeper levels).")
    parser.add_argument("--max-depth", type=int, default=4, help="Max recolour-recursion depth before keeping mixed as wall.")
    parser.add_argument("--prepare-only", action="store_true", help="Render first-round images/prompts, no VLM calls.")
    parser.add_argument("--dump-request", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, default=1500)
    args = parser.parse_args()

    reports: list[Path] = []
    for path in args.candidate_report:
        if path.is_dir() and not (path / "candidate_report.json").exists():
            reports.extend(sorted(path.glob("*/candidate_report.json")))
        else:
            reports.append(path)
    if not reports:
        print("No candidate reports found.")
        return 2

    batch_start = time.time()
    total_calls = 0
    for report_path in reports:
        name = output_name_for_report(report_path)
        manifest = run_candidate_report(report_path, args.out_dir / name, args)
        total_calls += manifest["vlm_calls"]
        print(
            f"[{name}] status={manifest['status']} calls={manifest['vlm_calls']} "
            f"walls={manifest['selected_wall_count']} "
            f"time={manifest.get('elapsed_sec', 0)}s (vlm {manifest.get('vlm_elapsed_sec', 0)}s)"
        )
        for stream, info in manifest["stream_summary"].items():
            print(f"    {stream}: in={info['input']} selected={info['selected']}")
    if len(reports) > 1:
        print(f"TOTAL: {len(reports)} reports, {total_calls} vlm calls, {time.time() - batch_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
