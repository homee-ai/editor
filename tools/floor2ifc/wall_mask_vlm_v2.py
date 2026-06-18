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


def candidates_from_report(report: dict[str, Any], inventory: dict[str, Any]) -> list[dict[str, Any]]:
    inventory_by_id = {item["id"]: item for item in inventory["elements"]}
    kept = report.get("kept")
    if isinstance(kept, list) and kept:
        candidates: list[dict[str, Any]] = []
        for item in kept:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or item_id not in inventory_by_id:
                continue
            # Use fresh geometry/style from the SVG inventory, and preserve mask
            # filter metadata for reports/debugging.
            merged = dict(inventory_by_id[item_id])
            for key in ("mask_pixels", "mask_bbox_ratio", "selection_reason"):
                if key in item:
                    merged[key] = item[key]
            candidates.append(merged)
        if candidates:
            return candidates

    kept_ids = {item for item in report.get("kept_ids", []) if isinstance(item, str)}
    return [item for item in inventory["elements"] if item["id"] in kept_ids]


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
    candidates = candidates_from_report(report, inventory)[: args.max_candidates]
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
    write_json(
        candidates_path,
        {
            "source_svg": str(svg_path),
            "candidate_report": report["_candidate_report_path"],
            "source_png": report.get("png"),
            "semantic_filter_mask": report.get("semantic_filter_mask"),
            "input_candidate_count": report.get("kept_count", len(candidates)),
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

    groups_dir.mkdir(parents=True, exist_ok=True)
    client = None if args.prepare_only or args.mock_response else VlmClient(args.env_file_root)
    group_reviews: list[dict[str, Any]] = []
    selected_id_set: set[str] = set()
    queue = list(initial_groups)
    processed_count = 0

    mock_review = None
    if args.mock_response:
        mock_review = json.loads(args.mock_response.read_text(encoding="utf-8"))

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

    selected_ids = sorted(selected_id_set)
    write_json(
        review_path,
        {
            "source_svg": str(svg_path),
            "candidate_report": report["_candidate_report_path"],
            "initial_group_count": len(initial_groups),
            "processed_group_count": processed_count,
            "selected_wall_count": len(selected_ids),
            "selected_wall_ids": selected_ids,
            "groups": group_reviews,
        },
    )
    render_mask_svgs(svg_path, selected_ids, candidates, overlay_path, mask_path)

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
