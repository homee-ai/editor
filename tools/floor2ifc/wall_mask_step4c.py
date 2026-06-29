#!/usr/bin/env python3
"""Method A, step 4c: simplify + straighten each edge into straight wall axes.

Pipeline: prepare_graph (steps 0-3) -> 4a prune spurs -> 4b dissolve degree-2 ->
4c straighten. Each graph edge is a polyline tracing one wall run; 4c runs
Douglas-Peucker on it (tolerance scaled by wall thickness) so a near-straight run
collapses to a single segment and a real corner (an L traced as one edge) splits
into two straight segments at the bend.

Output of 4c is a flat list of straight wall axes: (p1, p2, thickness_px). Writes
a before(polylines)/after(straight axes) comparison panel next to each mask.
Diagnostic only; no 3D geometry yet.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from wall_mask_skeleton import (
    add_pipeline_args,
    edge_thickness,
    find_masks,
    prepare_graph,
)
from wall_mask_step4 import dissolve_degree2, global_thickness, prune_spurs


def rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification. pts are [row, col]."""
    if len(pts) < 3:
        return pts
    start, end = pts[0].astype(float), pts[-1].astype(float)
    line = end - start
    norm = float(np.hypot(line[0], line[1]))
    if norm < 1e-9:
        d = np.hypot(pts[:, 0] - start[0], pts[:, 1] - start[1])
    else:
        # perpendicular distance of each point to the start-end line
        rel = pts.astype(float) - start
        d = np.abs(rel[:, 0] * line[1] - rel[:, 1] * line[0]) / norm
    idx = int(np.argmax(d))
    if d[idx] > eps:
        left = rdp(pts[: idx + 1], eps)
        right = rdp(pts[idx:], eps)
        return np.vstack([left[:-1], right])
    return np.vstack([pts[0], pts[-1]])


def straighten(graph: Any, dist: np.ndarray, dp_frac: float, global_t: float) -> list[dict[str, Any]]:
    """Split every edge polyline into straight wall-axis segments via RDP."""
    segments: list[dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) < 2:
            continue
        thickness = edge_thickness(dist, pts) or global_t
        eps = max(1.0, dp_frac * thickness)
        simplified = rdp(pts, eps)
        for a, b in zip(simplified[:-1], simplified[1:]):
            if np.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6:
                continue
            segments.append({"p1": a.astype(float), "p2": b.astype(float),
                             "thickness_px": round(thickness, 2)})
    return segments


def draw_polylines(ax: Any, base: np.ndarray | None, fallback: np.ndarray, graph: Any, title: str) -> None:
    ax.set_title(title)
    ax.imshow(base if base is not None else fallback, cmap=None if base is not None else "gray")
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color="#1f77b4", linewidth=1.3)
    for node in graph.nodes():
        y, x = graph.nodes[node]["o"]
        deg = graph.degree(node)
        color = "#2ca02c" if deg == 1 else ("#d62728" if deg >= 3 else "#ff7f0e")
        ax.plot(x, y, "o", color=color, markersize=4)
    ax.axis("off")


def draw_segments(ax: Any, base: np.ndarray | None, fallback: np.ndarray, segments: list[dict[str, Any]], title: str) -> None:
    ax.set_title(title)
    ax.imshow(base if base is not None else fallback, cmap=None if base is not None else "gray")
    for seg in segments:
        p1, p2 = seg["p1"], seg["p2"]
        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], color="#1f77b4", linewidth=1.6)
        ax.plot([p1[1], p2[1]], [p1[0], p2[0]], ".", color="#d62728", markersize=4)
    ax.axis("off")


def process_mask(mask_svg: Path, args: argparse.Namespace) -> dict[str, Any]:
    p = prepare_graph(mask_svg, args)
    sub_dir, dist, original, connected = p["sub_dir"], p["dist"], p["original"], p["connected"]

    # prefer the not_wall-cleaned graph from wall_mask_review.py (already 4a+4b +
    # endpoint pruning); fall back to computing 4a+4b here.
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
    edges_after = graph.number_of_edges()

    segments = straighten(graph, dist, args.dp_frac, global_t)

    base = None
    if original is not None:
        base = (np.asarray(Image.fromarray(original).convert("RGB")).astype(float) * 0.45 + 255 * 0.55).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{sub_dir.name}  src={source}  dp_frac={args.dp_frac}  global_thickness={global_t:.1f}px  "
        f"edges {edges_after} -> {len(segments)} straight axes",
        fontsize=12,
    )
    draw_polylines(axes[0], base, connected, graph, f"input ({source})  edges={edges_after}")
    draw_segments(axes[1], base, connected, segments, f"4c straightened  segments={len(segments)}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"step4c_straighten_{args.tag}.png" if args.tag else "step4c_straighten_debug.png"
    out_png = sub_dir / name
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    return {"image": str(out_png), "edges": edges_after, "segments": len(segments),
            "global_thickness": round(global_t, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    parser.add_argument("--k", type=float, default=1.5, help="4a spur length threshold = k * wall thickness.")
    parser.add_argument("--dp-frac", type=float, default=0.6, help="4c RDP tolerance = frac * wall thickness.")
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
        print(
            f"[{mask_svg.parent.name}] edges {info['edges']} -> {info['segments']} straight axes "
            f"(t={info['global_thickness']}px) -> {info['image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
