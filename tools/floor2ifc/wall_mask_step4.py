#!/usr/bin/env python3
"""Method A, step 4a+4b: prune spurs + dissolve degree-2 nodes (collinear merge).

Consumes the cleaned/connected skeleton graph from wall_mask_skeleton.py
(thin-removal + door/window merge + reconnect), then cleans the graph:

  4a  prune spurs: drop leaf (degree-1) edges shorter than k * wall-thickness,
      iteratively (a pruned spur can turn a junction into a pass-through).
  4b  dissolve degree-2 nodes: merge the two incident edges into one polyline,
      collapsing T-junction stubs that became straight pass-throughs and
      merging runs between real junctions.

Length/distance thresholds are scaled by wall thickness (median EDT*2), since a
real wall is long relative to its thickness while a spur is ~thickness-long.

Writes a before/after comparison panel next to each wall mask. Diagnostic only.
"""

from __future__ import annotations

import argparse
import statistics
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


def path_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    diff = np.diff(pts.astype(float), axis=0)
    return float(np.hypot(diff[:, 0], diff[:, 1]).sum())


def _near(p: np.ndarray, coord: np.ndarray) -> float:
    return float(np.hypot(p[0] - coord[0], p[1] - coord[1]))


def global_thickness(graph: Any, dist: np.ndarray) -> float:
    values = [edge_thickness(dist, d["pts"]) for _, _, d in graph.edges(data=True)]
    values = [v for v in values if v > 0]
    return statistics.median(values) if values else 4.0


def prune_spurs(graph: Any, dist: np.ndarray, k: float, global_t: float) -> int:
    """Iteratively drop short leaf edges. Returns number removed."""
    removed = 0
    while True:
        victim = None
        for u, v, data in graph.edges(data=True):
            if graph.degree(u) != 1 and graph.degree(v) != 1:
                continue
            thickness = edge_thickness(dist, data["pts"]) or global_t
            if path_length(data["pts"]) < k * thickness:
                victim = (u, v)
                break
        if victim is None:
            break
        u, v = victim
        graph.remove_edge(u, v)
        for node in (u, v):
            if graph.degree(node) == 0:
                graph.remove_node(node)
        removed += 1
    return removed


def _one_edge_pts(graph: Any, u: Any, v: Any) -> np.ndarray:
    """pts of one edge between u and v (MultiGraph stores them keyed)."""
    return np.asarray(next(iter(graph.get_edge_data(u, v).values()))["pts"])


def dissolve_degree2(graph: Any, dist: np.ndarray) -> int:
    """Merge the two edges at each degree-2 node into one. Returns merges done."""
    merged_count = 0
    skip: set[Any] = set()
    while True:
        target = None
        for node in graph.nodes():
            if node in skip or graph.degree(node) != 2:
                continue
            neighbours = list(graph.neighbors(node))
            # len==1 means a self-loop or two parallel edges to the same node
            # (a bare loop bubble) — not a pass-through, leave it alone.
            if len(neighbours) != 2:
                skip.add(node)
                continue
            target = node
            break
        if target is None:
            break

        node = target
        a, b = list(graph.neighbors(node))
        coord = graph.nodes[node]["o"]
        e1 = _one_edge_pts(graph, a, node)
        e2 = _one_edge_pts(graph, node, b)
        if len(e1) and _near(e1[0], coord) < _near(e1[-1], coord):
            e1 = e1[::-1]  # e1 now ends at node
        if len(e2) and _near(e2[-1], coord) < _near(e2[0], coord):
            e2 = e2[::-1]  # e2 now starts at node
        merged = np.vstack([e1, np.asarray([coord]), e2]) if len(e1) and len(e2) else (e1 if len(e1) else e2)

        graph.remove_node(node)
        graph.add_edge(a, b, pts=merged, weight=path_length(merged))
        merged_count += 1
    return merged_count


def draw_graph(ax: Any, base: np.ndarray | None, connected: np.ndarray, graph: Any, title: str) -> None:
    ax.set_title(title)
    if base is not None:
        ax.imshow(base)
    else:
        ax.imshow(connected, cmap="gray")
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


def process_mask(mask_svg: Path, args: argparse.Namespace) -> dict[str, Any]:
    p = prepare_graph(mask_svg, args)
    sub_dir = p["sub_dir"]
    connected, dist, original = p["connected"], p["dist"], p["original"]

    # faithful "before" graph for side-by-side (the working one gets mutated)
    graph_before = prepare_graph(mask_svg, args)["graph"]
    graph = p["graph"]

    before = (graph.number_of_nodes(), graph.number_of_edges())
    global_t = global_thickness(graph, dist)
    pruned = prune_spurs(graph, dist, args.k, global_t)
    merged = dissolve_degree2(graph, dist)
    after = (graph.number_of_nodes(), graph.number_of_edges())

    base = None
    if original is not None:
        base = (np.asarray(Image.fromarray(original).convert("RGB")).astype(float) * 0.45 + 255 * 0.55).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{sub_dir.name}  k={args.k}  global_thickness={global_t:.1f}px  "
        f"pruned {pruned} spurs, merged {merged} nodes",
        fontsize=12,
    )
    draw_graph(axes[0], base, connected, graph_before, f"before  nodes={before[0]} edges={before[1]}")
    draw_graph(axes[1], base, connected, graph, f"after 4a+4b  nodes={after[0]} edges={after[1]}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"step4_prune_{args.tag}.png" if args.tag else "step4_prune_debug.png"
    out_png = sub_dir / name
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    endpoints_before = sum(1 for n in graph_before.nodes() if graph_before.degree(n) == 1)
    endpoints_after = sum(1 for n in graph.nodes() if graph.degree(n) == 1)
    return {
        "image": str(out_png),
        "before": before,
        "after": after,
        "pruned": pruned,
        "merged": merged,
        "endpoints_before": endpoints_before,
        "endpoints_after": endpoints_after,
        "global_thickness": round(global_t, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    parser.add_argument("--k", type=float, default=1.5, help="spur length threshold = k * wall thickness.")
    parser.add_argument("--tag", type=str, default="", help="suffix for the output filename (e.g. k2.0).")
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
            f"[{mask_svg.parent.name}] nodes {info['before'][0]}->{info['after'][0]} "
            f"edges {info['before'][1]}->{info['after'][1]} "
            f"endpoints {info['endpoints_before']}->{info['endpoints_after']} "
            f"(pruned {info['pruned']}, merged {info['merged']}, t={info['global_thickness']}px) -> {info['image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
