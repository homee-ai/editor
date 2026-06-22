#!/usr/bin/env python3
"""Run VLM wall labeling from png_mask_svg_candidates.py candidate reports.

This is v2 of wall_mask_vlm.py. It keeps the original VLM grouping/review/mask
pipeline, but takes pre-filtered SVG candidates from candidate_report.json
instead of selecting candidates only from SVG geometry/style heuristics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from svg_inventory import build_inventory
from vlm_client import REPO_ROOT, VlmClient
from wall_mask_vlm import (
    build_group_prompt,
    candidate_summary,
    compact_style_summary,
    group_candidates,
    group_with_candidates,
    parse_vlm_json,
    render_candidate_overlay,
    render_group_mask,
    render_group_overlay,
    render_mask_svgs,
    repair_vlm_json,
    review_candidate_ids,
    should_split_none_wall,
    split_for_all_wall_confirmation,
    split_group,
    write_json,
)


DEFAULT_OUT_DIR = Path(__file__).parent / "out/wall-mask-vlm-v2"


def resolve_candidate_report(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "candidate_report.json"
        if candidate.exists():
            return candidate
        matches = sorted(path.glob("*/candidate_report.json"))
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(f"Could not resolve one candidate_report.json from directory: {path}")
    return path


def load_candidate_report(path: Path) -> dict[str, Any]:
    report_path = resolve_candidate_report(path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if "generated_svg" not in data:
        raise ValueError(f"candidate report is missing generated_svg: {report_path}")
    if "kept" not in data and "kept_ids" not in data:
        raise ValueError(f"candidate report is missing kept candidates: {report_path}")
    data["_candidate_report_path"] = str(report_path)
    return data


_PRESERVED_KEYS = ("mask_pixels", "mask_bbox_ratio", "selection_reason", "case", "a_object_in_mask", "b_maskblob_in_object")


def resolve_rows(rows: Any, inventory_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    """Map report rows (id + metadata) onto fresh SVG-inventory geometry."""
    resolved: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return resolved
    for item in rows:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or item_id not in inventory_by_id:
            continue
        merged = dict(inventory_by_id[item_id])
        for key in _PRESERVED_KEYS:
            if key in item:
                merged[key] = item[key]
        resolved.append(merged)
    return resolved


def make_group(group_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "group_id": group_id,
        "base_group_id": group_id,
        "depth": 0,
        "candidate_count": len(candidates),
        "candidate_ids": [item["id"] for item in candidates],
        "candidates": candidates,
    }


def candidates_from_report(report: dict[str, Any], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    inventory_by_id = {item["id"]: item for item in inventory["elements"]}
    candidates = resolve_rows(report.get("kept"), inventory_by_id)
    if candidates:
        return candidates

    kept_ids = {item for item in report.get("kept_ids", []) if isinstance(item, str)}
    return [item for item in inventory["elements"] if item["id"] in kept_ids]


def build_case1_verify_prompt(inventory: dict[str, Any], group: dict[str, Any]) -> str:
    summary = {
        "source_svg": inventory.get("source_svg"),
        "viewBox": inventory.get("viewBox"),
        "group_id": group["group_id"],
        "candidate_count": group["candidate_count"],
        "style_summary_top": compact_style_summary(inventory),
        "candidates": candidate_summary(group["candidates"]),
    }
    return f"""You are verifying candidate shapes that ALREADY lie on the structural wall mask of a floor plan (each overlaps the wall mask almost entirely). They are wall material by construction, so DEFAULT TO WALL.

You will see THREE images:
1. Original clean floor-plan image.
2. Isolated candidate mask image (magenta) on a white background.
3. The same candidates overlaid on the floor plan with candidate ids.

Task:
- These candidates sit on the structural wall. Treat EVERY candidate as a wall unless you are confident it is not.
- Walls include exterior/interior walls, wall outlines, room-dividing boundaries, WINDOWS set into a wall, door-opening wall lines, door jambs, and short wall stubs. All of these are walls — keep them.
- A thin line or short bar sitting in the wall at a door opening is a door-opening wall line / jamb, NOT a door leaf. Keep it as wall. Do NOT reject it as a door leaf/panel.
- ONLY mark a candidate as non-wall if it is clearly a free-standing non-structural mark that merely happens to overlap a thick wall: a furniture/fixture icon, a text label, a dimension number/arrow, a logo, or decorative detail. When in doubt, it is a wall.

Output rules:
- Return ONLY one minified JSON object. No markdown, no code fence, no prose.

JSON shape:
{{
  "group_id": "{group['group_id']}",
  "verdict": "all_wall|none_wall|mixed",
  "wall_ids": [],
  "non_wall_ids": [],
  "reason": "short visual reason",
  "confidence": 0.0
}}

Verdict rules:
- all_wall: every candidate is wall (this is the expected default here).
- mixed: only if some candidate is a clearly non-structural free-standing mark. Put those in non_wall_ids and ALL other ids in wall_ids. Every candidate id MUST appear in exactly one of wall_ids or non_wall_ids.
- none_wall: only if NONE are walls (very unlikely for this set).
- Use only ids shown in image 3 and listed in the candidate summary.
"""


def verify_case1(
    auto_candidates: list[dict[str, Any]],
    svg_path: Path,
    inventory: dict[str, Any],
    groups_dir: Path,
    client: VlmClient | None,
    args: argparse.Namespace,
    mock_review: dict[str, Any] | None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """One VLM call over all case-1 auto-walls.

    all_wall -> confirm every case-1 object as wall.
    mixed    -> wall_ids and unclassified ids confirmed as wall; non_wall_ids
                are pushed back into the candidate pool for the normal pipeline.
    none_wall-> push the whole batch back into the candidate pool.
    """
    group = make_group("case1_auto_wall_verify", auto_candidates)
    group_dir = groups_dir / group["group_id"]
    group_dir.mkdir(parents=True, exist_ok=True)
    mask_svg = group_dir / "candidate_mask.svg"
    overlay_svg = group_dir / "candidate_overlay.svg"
    prompt_path = group_dir / "wall_label_prompt.txt"
    raw_path = group_dir / "vlm_raw.txt"
    review_out = group_dir / "vlm_wall_labels.json"

    render_group_mask(svg_path, group, mask_svg, args.show_ids)
    render_group_overlay(svg_path, group, overlay_svg)
    prompt = build_case1_verify_prompt(inventory, group)
    prompt_path.write_text(prompt, encoding="utf-8")

    review: dict[str, Any] = {
        "group_id": group["group_id"],
        "verdict": "mixed" if args.prepare_only else "all_wall",
        "reason": "prepare_only" if args.prepare_only else "",
        "confidence": 0,
    }
    if mock_review is not None:
        review = {**review, **mock_review, "group_id": group["group_id"]}
        raw_path.write_text(json.dumps(review, ensure_ascii=False), encoding="utf-8")
    elif not args.prepare_only and client is not None:
        raw = client.chat_with_svgs(
            prompt,
            [svg_path, mask_svg, overlay_svg],
            image_size=args.image_size,
            dump_request=args.dump_request,
            max_tokens=args.max_output_tokens,
        )
        raw_path.write_text(raw, encoding="utf-8")
        try:
            review = parse_vlm_json(raw, raw_path)
        except ValueError:
            repaired = repair_vlm_json(client, raw, args.dump_request, args.max_output_tokens)
            repair_path = group_dir / "vlm_repair_raw.txt"
            repair_path.write_text(repaired, encoding="utf-8")
            review = parse_vlm_json(repaired, repair_path)

    verdict = str(review.get("verdict") or "").strip().lower()
    if verdict not in {"all_wall", "none_wall", "mixed"}:
        verdict = "mixed"

    non_wall_ids = review_candidate_ids(review, "non_wall_ids", group)
    if verdict == "none_wall":
        confirmed_ids: list[str] = []
        pushed_back = list(auto_candidates)
    else:  # all_wall, or mixed (wall + unclassified -> wall)
        confirmed_ids = [item["id"] for item in auto_candidates if item["id"] not in non_wall_ids]
        pushed_back = [item for item in auto_candidates if item["id"] in non_wall_ids]

    write_json(review_out, review)
    meta = {
        "verdict": verdict,
        "auto_wall_count": len(auto_candidates),
        "confirmed_count": len(confirmed_ids),
        "pushed_back_count": len(pushed_back),
        "vlm_wall_labels": str(review_out),
    }
    return confirmed_ids, pushed_back, meta


def run_candidate_report(report_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    report = load_candidate_report(report_path)
    svg_path = Path(report["generated_svg"])
    if not svg_path.exists():
        candidate_report_dir = Path(report["_candidate_report_path"]).parent
        fallback = candidate_report_dir / svg_path.name
        if fallback.exists():
            svg_path = fallback
        else:
            raise FileNotFoundError(f"Generated SVG not found: {svg_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    inventory = build_inventory(svg_path)
    inventory_by_id = {item["id"]: item for item in inventory["elements"]}
    candidates = candidates_from_report(report, inventory)[: args.max_candidates]
    auto_candidates = resolve_rows(report.get("auto_wall"), inventory_by_id)

    inventory_path = out_dir / "inventory.json"
    candidates_path = out_dir / "candidates.json"
    candidate_overlay_path = out_dir / "candidate_overlay.svg"
    groups_dir = out_dir / "groups"
    groups_path = out_dir / "candidate_groups.json"
    review_path = out_dir / "vlm_wall_labels.json"
    mask_path = out_dir / "wall_mask.svg"
    overlay_path = out_dir / "wall_mask_overlay.svg"
    manifest_path = out_dir / "result.json"

    groups_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.prepare_only or args.mock_response else VlmClient(args.env_file_root)
    mock_review = json.loads(args.mock_response.read_text(encoding="utf-8")) if args.mock_response else None

    # Verify case-1 auto-walls in a single VLM call; confirmed ones skip the
    # pipeline, rejected ones rejoin case-2/case-3 as ordinary candidates.
    case1_confirmed_ids: list[str] = []
    case1_meta: dict[str, Any] = {"verdict": None, "auto_wall_count": len(auto_candidates), "confirmed_count": 0, "pushed_back_count": 0}
    if auto_candidates:
        case1_confirmed_ids, pushed_back, case1_meta = verify_case1(
            auto_candidates, svg_path, inventory, groups_dir, client, args, mock_review
        )
        candidates = candidates + pushed_back

    initial_groups = group_candidates(candidates, args.max_group_size)

    write_json(inventory_path, inventory)
    write_json(
        candidates_path,
        {
            "source_svg": str(svg_path),
            "candidate_report": report["_candidate_report_path"],
            "source_png": report.get("png"),
            "semantic_filter_mask": report.get("semantic_filter_mask"),
            "input_candidate_count": report.get("kept_count", len(candidates)),
            "candidate_count": len(candidates),
            "case1_verification": case1_meta,
            "candidates": candidates,
        },
    )
    write_json(
        groups_path,
        {
            "source_svg": str(svg_path),
            "candidate_report": report["_candidate_report_path"],
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

    group_reviews: list[dict[str, Any]] = []
    selected_id_set: set[str] = set()
    queue = list(initial_groups)
    processed_count = 0

    while queue:
        group = queue.pop(0)
        processed_count += 1
        group_dir = groups_dir / group["group_id"]
        group_dir.mkdir(parents=True, exist_ok=True)
        group_mask_path = group_dir / "candidate_mask.svg"
        group_overlay_path = group_dir / "candidate_overlay.svg"
        group_prompt_path = group_dir / "wall_label_prompt.txt"
        group_raw_path = group_dir / "vlm_raw.txt"
        group_review_path = group_dir / "vlm_wall_labels.json"
        group_prompt = build_group_prompt(inventory, group)

        render_group_mask(svg_path, group, group_mask_path, args.show_ids)
        render_group_overlay(svg_path, group, group_overlay_path)
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
                [svg_path, group_mask_path, group_overlay_path],
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
            children = split_for_all_wall_confirmation(
                group,
                args.max_group_size,
                args.max_split_depth,
                args.confirm_all_wall_size,
            )
            if children:
                queue.extend(children)
                split_group_ids = [child["group_id"] for child in children]
            else:
                selected_ids = list(group["candidate_ids"])
                selected_id_set.update(selected_ids)
        elif verdict == "none_wall" and should_split_none_wall(
            group,
            args.max_group_size,
            args.max_split_depth,
            args.max_terminal_none_size,
        ):
            children = split_group(group, args.max_group_size, args.max_split_depth)
            if children:
                queue.extend(children)
                split_group_ids = [child["group_id"] for child in children]
        elif verdict == "mixed":
            explicit_wall_ids = review_candidate_ids(review, "wall_ids", group)
            explicit_non_wall_ids = review_candidate_ids(review, "non_wall_ids", group)
            if explicit_wall_ids:
                selected_ids = sorted(explicit_wall_ids)
                selected_id_set.update(selected_ids)

            remaining_candidates = [
                item
                for item in group["candidates"]
                if item["id"] not in explicit_wall_ids and item["id"] not in explicit_non_wall_ids
            ]
            split_source = group_with_candidates(group, remaining_candidates) if remaining_candidates else group
            children = split_group(split_source, args.max_group_size, args.max_split_depth) if remaining_candidates else []
            if children:
                queue.extend(children)
                split_group_ids = [child["group_id"] for child in children]
            elif remaining_candidates:
                # Mixed contract: every candidate must be wall or non_wall. Anything the
                # VLM left unclassified that we can no longer subdivide is kept as wall
                # rather than silently dropped into neither bucket.
                fallback_ids = [item["id"] for item in remaining_candidates]
                selected_ids = sorted(set(selected_ids) | set(fallback_ids))
                selected_id_set.update(fallback_ids)

        write_json(group_review_path, review)
        group_reviews.append(
            {
                "group_id": group["group_id"],
                "base_group_id": group["base_group_id"],
                "parent_group_id": group.get("parent_group_id"),
                "confirming_all_wall_parent": group.get("confirming_all_wall_parent"),
                "split_key": group.get("split_key"),
                "depth": group["depth"],
                "candidate_count": group["candidate_count"],
                "verdict": verdict,
                "selected_wall_count": len(selected_ids),
                "selected_wall_ids": selected_ids,
                "review_wall_ids": sorted(review_candidate_ids(review, "wall_ids", group)),
                "review_non_wall_ids": sorted(review_candidate_ids(review, "non_wall_ids", group)),
                "split_group_ids": split_group_ids,
                "candidate_mask": str(group_mask_path),
                "candidate_overlay": str(group_overlay_path),
                "prompt": str(group_prompt_path),
                "vlm_raw": str(group_raw_path) if group_raw_path.exists() else None,
                "vlm_wall_labels": str(group_review_path),
                "confidence": review.get("confidence"),
            }
        )

    vlm_selected_ids = sorted(selected_id_set)
    # Final walls = pipeline-selected (case-2/3 + pushed-back case-1) U confirmed case-1.
    selected_ids = sorted(set(vlm_selected_ids) | set(case1_confirmed_ids))
    write_json(
        review_path,
        {
            "source_svg": str(svg_path),
            "candidate_report": report["_candidate_report_path"],
            "initial_group_count": len(initial_groups),
            "processed_group_count": processed_count,
            "vlm_selected_count": len(vlm_selected_ids),
            "vlm_selected_wall_ids": vlm_selected_ids,
            "case1_verification": case1_meta,
            "case1_confirmed_count": len(case1_confirmed_ids),
            "case1_confirmed_ids": sorted(case1_confirmed_ids),
            "selected_wall_count": len(selected_ids),
            "selected_wall_ids": selected_ids,
            "groups": group_reviews,
        },
    )
    render_mask_svgs(svg_path, selected_ids, candidates + auto_candidates, overlay_path, mask_path)

    manifest = {
        "source_svg": str(svg_path),
        "candidate_report": report["_candidate_report_path"],
        "source_png": report.get("png"),
        "semantic_filter_mask": report.get("semantic_filter_mask"),
        "status": "prepared" if args.prepare_only and not args.mock_response else "completed",
        "candidate_count": len(candidates),
        "input_candidate_count": report.get("kept_count"),
        "initial_group_count": len(initial_groups),
        "processed_group_count": processed_count,
        "vlm_selected_count": len(vlm_selected_ids),
        "case1_verdict": case1_meta.get("verdict"),
        "case1_confirmed_count": len(case1_confirmed_ids),
        "case1_pushed_back_count": case1_meta.get("pushed_back_count", 0),
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


def output_name_for_report(report_path: Path) -> str:
    report = load_candidate_report(report_path)
    png = report.get("png")
    if isinstance(png, str) and png:
        return Path(png).stem
    svg = report.get("generated_svg")
    if isinstance(svg, str) and svg:
        return Path(svg).stem
    return resolve_candidate_report(report_path).parent.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "candidate_report",
        type=Path,
        nargs="+",
        help="candidate_report.json or output directory from png_mask_svg_candidates.py.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--env-file-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--image-size", type=int, default=1800)
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--max-group-size", type=int, default=90)
    parser.add_argument("--max-split-depth", type=int, default=2)
    parser.add_argument("--max-terminal-none-size", type=int, default=12)
    parser.add_argument("--confirm-all-wall-size", type=int, default=12)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--show-ids", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mock-response", type=Path, default=None)
    parser.add_argument("--dump-request", action="store_true")
    args = parser.parse_args()

    if args.mock_response and not args.mock_response.exists():
        print(f"Mock response not found: {args.mock_response}", file=sys.stderr)
        return 2

    manifests: list[dict[str, Any]] = []
    try:
        for input_path in args.candidate_report:
            report_path = resolve_candidate_report(input_path)
            if not report_path.exists():
                print(f"candidate report not found: {input_path}", file=sys.stderr)
                return 2
            out_dir = args.out_dir / output_name_for_report(report_path)
            manifest = run_candidate_report(report_path, out_dir, args)
            manifests.append(manifest)
            print(f"\n{report_path}")
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
    except (RuntimeError, ValueError, json.JSONDecodeError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        return 1

    if len(manifests) > 1:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.out_dir / "summary.json"
        write_json(summary_path, {"count": len(manifests), "results": manifests})
        print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
