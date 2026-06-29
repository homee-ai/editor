#!/usr/bin/env python3
"""Method A, stage C unified: wall_mask.svg -> regularised wall-axis model.

Runs the whole C chain in one process by reusing the per-step functions (no logic
is duplicated, so the result is identical to running the steps separately):

  0-3  wall_mask_skeleton.prepare_graph   (clean + skeleton + thickness + graph)
  4a/4b wall_mask_step4.prune_spurs / dissolve_degree2
  4b.5 not_wall  (optional, --run-notwall): wall_mask_review colour-batch VLM;
       otherwise the existing review_graph_not_wall.pkl is used if present
  4c   wall_mask_step4c.rdp straighten -> wall_mask_step4de.build_axis_graph
  4d/4e wall_mask_step4de.regularize    (H/V snap + junction share + D-merge)
  4f   wall_mask_step4f.quantize_thickness

Outputs wall_model.json (final walls) + a 6-panel pipeline_debug.png. With
--compare it also reports the max coordinate/thickness difference vs the separate
step4f output (should be 0).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from wall_mask_skeleton import add_pipeline_args, find_masks, prepare_graph
from wall_mask_step4 import dissolve_degree2, global_thickness, prune_spurs
from wall_mask_step4de import build_axis_graph, merge_collinear, regularize
from wall_mask_step4f import draw_footprint, quantize_thickness, walls_to_records, BIN_COLORS


def _faded(original: np.ndarray | None, shape: tuple[int, int], a: float) -> np.ndarray:
    if original is None:
        return np.full((*shape, 3), 255, dtype=np.uint8)
    return (np.asarray(Image.fromarray(original).convert("RGB")).astype(float) * a + 255 * (1 - a)).astype(np.uint8)


def _draw_graph(ax: Any, base: np.ndarray, graph: Any, title: str) -> None:
    ax.set_title(title)
    ax.imshow(base)
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color="#1f77b4", linewidth=1.1)
    for n in graph.nodes():
        y, x = graph.nodes[n]["o"]
        deg = graph.degree(n)
        ax.plot(x, y, "o", markersize=3,
                color="#2ca02c" if deg == 1 else ("#d62728" if deg >= 3 else "#ff7f0e"))
    ax.axis("off")


def _draw_axes(ax: Any, base: np.ndarray, verts: dict[Any, np.ndarray],
               edges: list[tuple[Any, Any, float]], title: str) -> None:
    ax.set_title(title)
    ax.imshow(base)
    for a, b, _ in edges:
        pa, pb = verts[a], verts[b]
        ax.plot([pa[1], pb[1]], [pa[0], pb[0]], color="#1f77b4", linewidth=1.4)
        ax.plot([pa[1], pb[1]], [pa[0], pb[0]], ".", color="#d62728", markersize=2.5)
    ax.axis("off")


def run_pipeline(mask_svg: Path, args: argparse.Namespace) -> dict[str, Any]:
    p = prepare_graph(mask_svg, args)
    sub_dir, dist, original, connected = p["sub_dir"], p["dist"], p["original"], p["connected"]

    # 4a/4b on a fresh graph (kept for the debug panel)
    g4 = p["graph"]
    gt = global_thickness(g4, dist)
    prune_spurs(g4, dist, args.k, gt)
    dissolve_degree2(g4, dist)

    # 4b.5 source: not_wall-cleaned pkl if present (same as the separate 4c/4de/4f)
    pkl = sub_dir / "review_graph_not_wall.pkl"
    if pkl.exists() and not args.fresh:
        graph = pickle.loads(pkl.read_bytes())
        source = "not_wall-cleaned"
    else:
        graph = g4
        source = "4a+4b"

    global_t = global_thickness(graph, dist)
    verts, edges = build_axis_graph(graph, dist, args.dp_frac, global_t)          # 4c
    reg_verts, stats = regularize(verts, edges, args.angle_tol, args.d_merge_factor)  # 4d/4e
    kept = [(a, b, t) for a, b, t in edges if np.hypot(*(reg_verts[a] - reg_verts[b])) > 1e-6]
    kept = merge_collinear(reg_verts, kept)  # remove collinear pass-throughs (editor miter-spike fix)
    gap = args.thickness_gap_frac * global_t
    quant_t, bins = quantize_thickness(kept, reg_verts, gap, args.max_bins, args.reliable_factor)  # 4f
    records = walls_to_records(kept, reg_verts, quant_t)

    (sub_dir / "wall_model.json").write_text(
        json.dumps({"walls": records, "bins": bins, "source": source,
                    "canvas": [int(connected.shape[0]), int(connected.shape[1])]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")

    # 6-panel pipeline figure
    base = _faded(original, connected.shape, 0.45)
    bin_reps = sorted({round(b[0], 1) for b in bins})
    bin_index = {r: i for i, r in enumerate(bin_reps)}
    quant_colors = [BIN_COLORS[bin_index[round(t, 1)] % len(BIN_COLORS)] for t in quant_t]
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(f"{sub_dir.name}  src={source}  walls={len(kept)}  "
                 f"bins={', '.join(f'{r}px x{n}' for r, n, _ in bins)}", fontsize=13)
    ax[0, 0].set_title("original"); ax[0, 0].imshow(original if original is not None else connected,
                                                    cmap=None if original is not None else "gray"); ax[0, 0].axis("off")
    ax[0, 1].set_title("0-3 connected clean mask"); ax[0, 1].imshow(connected, cmap="gray"); ax[0, 1].axis("off")
    _draw_graph(ax[0, 2], base, g4, "4a+4b graph")
    _draw_axes(ax[1, 0], base, verts, edges, "4c straightened")
    _draw_axes(ax[1, 1], base, reg_verts, kept, "4d+4e regularised")
    draw_footprint(ax[1, 2], base, connected, reg_verts, kept, quant_t, quant_colors, "4f footprint (quantised)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = sub_dir / "pipeline_debug.png"
    fig.savefig(out_png, dpi=100)
    plt.close(fig)

    return {"image": str(out_png), "walls": len(kept), "bins": bins, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    parser.add_argument("--k", type=float, default=1.5)
    parser.add_argument("--dp-frac", type=float, default=0.6)
    parser.add_argument("--angle-tol", type=float, default=20.0)
    parser.add_argument("--d-merge-factor", type=float, default=1.5)
    parser.add_argument("--thickness-gap-frac", type=float, default=0.25)
    parser.add_argument("--max-bins", type=int, default=3)
    parser.add_argument("--reliable-factor", type=float, default=2.0)
    parser.add_argument("--fresh", action="store_true", help="ignore not_wall pkl; use 4a+4b directly.")
    parser.add_argument("--tag", type=str, default="", help="(used only by the --compare step4f call).")
    parser.add_argument("--compare", action="store_true", help="diff wall_model.json vs a fresh step4f model.")
    args = parser.parse_args()

    masks: list[Path] = []
    for target in args.target:
        if target.exists():
            masks.extend(find_masks(target))
    if not masks:
        print("No wall_mask.svg found.")
        return 2

    for mask_svg in masks:
        info = run_pipeline(mask_svg, args)
        line = (f"[{mask_svg.parent.name}] walls={info['walls']} "
                f"bins=[{', '.join(f'{r}px(x{n})' for r, n, _ in info['bins'])}] -> {info['image']}")
        if args.compare:
            from wall_mask_step4f import process_mask as step4f_process
            step4f_process(mask_svg, args)  # writes wall_model.json (separate flow)
            sep = json.loads((mask_svg.parent / "wall_model.json").read_text())["walls"]
            uni = info["records"]
            same = sep == uni
            line += f"  | compare: {'IDENTICAL' if same else f'DIFFER ({len(sep)} vs {len(uni)})'}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
