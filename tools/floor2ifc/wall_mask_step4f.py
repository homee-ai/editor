#!/usr/bin/env python3
"""Method A, step 4f: per-wall thickness assignment + quantisation.

Runs the chain prepare_graph -> (not_wall-cleaned graph or 4a+4b) -> 4c straighten
-> 4d+4e regularise + D-merge (from wall_mask_step4de), then:

  4f  every wall axis already carries a thickness (median EDT*2 of its skeleton
      run). Cluster all thicknesses into a few standard values (1D gap-based,
      length-weighted) and snap each wall to its cluster, so walls of the same
      kind share one thickness instead of each being slightly different.

Draws a before/after panel as actual wall FOOTPRINTS (axis +/- thickness/2 filled
rectangles) — the 2D plan that step 5 will extrude to 3D. left = raw per-wall
thickness, right = quantised, with the bin list. Diagnostic only.
"""

from __future__ import annotations

import argparse
import json
import math
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

BIN_COLORS = ["#4363D8", "#E6194B", "#3CB44B", "#F58231", "#911EB4", "#469990"]


def quantize_thickness(edges: list[tuple[Any, Any, float]], verts: dict[Any, np.ndarray],
                       gap: float, max_bins: int, rel_factor: float
                       ) -> tuple[list[float], list[tuple[float, int, float]]]:
    """Agglomerative 1D clustering of per-wall thickness, length-weighted reps.

    Bins are formed only from RELIABLE walls (length >= rel_factor * thickness);
    short walls have junction-inflated thickness and would otherwise spawn a bogus
    thick bin. Bins are merged (smallest-gap-first) until <= max_bins and all gaps
    >= gap. Every wall (reliable or not) is then snapped to the nearest bin.
    Returns (per-edge quantised thickness, bins as (rep_px, n_walls, total_len)).
    """
    lengths = [float(np.hypot(*(verts[b] - verts[a]))) for a, b, _ in edges]
    reliable = [i for i in range(len(edges)) if lengths[i] >= rel_factor * edges[i][2]]
    pool = reliable if reliable else list(range(len(edges)))
    order = sorted(pool, key=lambda i: edges[i][2])
    clusters = [{"idx": [i], "len": lengths[i], "wsum": edges[i][2] * lengths[i]} for i in order]

    def rep(c: dict[str, Any]) -> float:
        return c["wsum"] / c["len"] if c["len"] > 0 else 0.0

    while len(clusters) > 1:
        gaps = [rep(clusters[i + 1]) - rep(clusters[i]) for i in range(len(clusters) - 1)]
        j = int(np.argmin(gaps))
        if len(clusters) <= max_bins and gaps[j] >= gap:
            break
        a, b = clusters[j], clusters[j + 1]
        clusters[j] = {"idx": a["idx"] + b["idx"], "len": a["len"] + b["len"], "wsum": a["wsum"] + b["wsum"]}
        del clusters[j + 1]

    reps = sorted(rep(c) for c in clusters)
    quant = [min(reps, key=lambda r: abs(edges[i][2] - r)) for i in range(len(edges))]
    counts: dict[float, list[float]] = {r: [] for r in reps}
    for i, q in enumerate(quant):
        counts[q].append(lengths[i])
    bins = [(round(r, 1), len(counts[r]), round(sum(counts[r]), 1)) for r in reps]
    return quant, bins


def walls_to_records(kept: list[tuple[Any, Any, float]], reg_verts: dict[Any, np.ndarray],
                     quant_t: list[float]) -> list[dict[str, float]]:
    """Canonical, sorted final wall list (for output + cross-run comparison)."""
    recs: list[dict[str, float]] = []
    for (a, b, _), t in zip(kept, quant_t):
        e1 = (round(float(reg_verts[a][0]), 3), round(float(reg_verts[a][1]), 3))
        e2 = (round(float(reg_verts[b][0]), 3), round(float(reg_verts[b][1]), 3))
        if e2 < e1:
            e1, e2 = e2, e1
        recs.append({"y1": e1[0], "x1": e1[1], "y2": e2[0], "x2": e2[1], "t": round(float(t), 3)})
    recs.sort(key=lambda r: (r["y1"], r["x1"], r["y2"], r["x2"], r["t"]))
    return recs


def wall_poly(pa: np.ndarray, pb: np.ndarray, t: float) -> tuple[list[float], list[float]] | None:
    """Footprint rectangle (xs, ys) of a wall axis pa-pb with width t. [y,x] in."""
    d = pb - pa
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-6:
        return None
    u = d / length
    n = np.array([-u[1], u[0]]) * (t / 2.0)
    corners = [pa + n, pb + n, pb - n, pa - n]
    return [c[1] for c in corners], [c[0] for c in corners]


def draw_footprint(ax: Any, base: np.ndarray | None, fallback: np.ndarray,
                   verts: dict[Any, np.ndarray], edges: list[tuple[Any, Any, float]],
                   thickness: list[float], colors: list[str], title: str) -> None:
    ax.set_title(title)
    ax.imshow(base if base is not None else fallback, cmap=None if base is not None else "gray")
    for (a, b, _), t, col in zip(edges, thickness, colors):
        poly = wall_poly(verts[a], verts[b], t)
        if poly:
            ax.fill(poly[0], poly[1], color=col, alpha=0.75, linewidth=0)
    ax.axis("off")


def process_mask(mask_svg: Path, args: argparse.Namespace) -> dict[str, Any]:
    p = prepare_graph(mask_svg, args)
    sub_dir, dist, original, connected = p["sub_dir"], p["dist"], p["original"], p["connected"]

    pkl = sub_dir / "review_graph_not_wall.pkl"
    if pkl.exists() and not args.fresh:
        graph = pickle.loads(pkl.read_bytes())
        source = "not_wall-cleaned"
    else:
        graph = p["graph"]
        gt = global_thickness(graph, dist)
        prune_spurs(graph, dist, args.k, gt)
        dissolve_degree2(graph, dist)
        source = "4a+4b"

    global_t = global_thickness(graph, dist)
    verts, edges = build_axis_graph(graph, dist, args.dp_frac, global_t)
    reg_verts, _ = regularize(verts, edges, args.angle_tol, args.d_merge_factor)
    kept = [(a, b, t) for a, b, t in edges if np.hypot(*(reg_verts[a] - reg_verts[b])) > 1e-6]
    kept = merge_collinear(reg_verts, kept)  # remove collinear pass-throughs (editor miter-spike fix)

    raw_t = [t for _, _, t in kept]
    gap = args.thickness_gap_frac * global_t
    quant_t, bins = quantize_thickness(kept, reg_verts, gap, args.max_bins, args.reliable_factor)
    bin_reps = sorted({round(b[0], 1) for b in bins})
    bin_index = {rep: i for i, rep in enumerate(bin_reps)}
    quant_colors = [BIN_COLORS[bin_index[round(t, 1)] % len(BIN_COLORS)] for t in quant_t]

    base = None
    if original is not None:
        base = (np.asarray(Image.fromarray(original).convert("RGB")).astype(float) * 0.4 + 255 * 0.6).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    bins_txt = ", ".join(f"{r:.1f}px x{n}" for r, n, _ in bins)
    fig.suptitle(f"{sub_dir.name}  src={source}  walls={len(kept)}  thickness bins: {bins_txt}", fontsize=12)
    draw_footprint(axes[0], base, connected, reg_verts, kept, raw_t, ["#1f77b4"] * len(kept),
                   "raw per-wall thickness (footprint)")
    draw_footprint(axes[1], base, connected, reg_verts, kept, quant_t, quant_colors,
                   "4f quantised thickness (colour = bin)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"step4f_{args.tag}.png" if args.tag else "step4f_debug.png"
    out_png = sub_dir / name
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    records = walls_to_records(kept, reg_verts, quant_t)
    (sub_dir / "wall_model.json").write_text(
        json.dumps({"walls": records, "bins": bins}, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"image": str(out_png), "walls": len(kept), "bins": bins}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    parser.add_argument("--k", type=float, default=1.5, help="4a spur length threshold = k * wall thickness.")
    parser.add_argument("--dp-frac", type=float, default=0.6, help="4c RDP tolerance = frac * wall thickness.")
    parser.add_argument("--angle-tol", type=float, default=20.0, help="4d degrees within H/V to snap.")
    parser.add_argument("--d-merge-factor", type=float, default=1.5, help="4e collapse diagonals shorter than factor * thickness.")
    parser.add_argument("--thickness-gap-frac", type=float, default=0.25, help="4f min bin gap = frac * median thickness.")
    parser.add_argument("--max-bins", type=int, default=3, help="4f max number of standard thicknesses.")
    parser.add_argument("--reliable-factor", type=float, default=2.0, help="4f a wall defines a bin only if length >= factor * thickness.")
    parser.add_argument("--fresh", action="store_true", help="ignore the not_wall-cleaned graph; run 4a+4b from scratch.")
    parser.add_argument("--tag", type=str, default="", help="suffix for the output filename.")
    args = parser.parse_args()

    masks: list[Path] = []
    for target in args.target:
        if target.exists():
            masks.extend(find_masks(target))
    if not masks:
        print("No wall_mask.svg found.")
        return 2

    for mask_svg in masks:
        info = process_mask(mask_svg, args)
        bins_txt = ", ".join(f"{r}px(x{n})" for r, n, _ in info["bins"])
        print(f"[{mask_svg.parent.name}] walls={info['walls']} bins=[{bins_txt}] -> {info['image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
