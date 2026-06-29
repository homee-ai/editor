#!/usr/bin/env python3
"""Method A, step 4d+4e: angle regularisation + junction snapping (coupled).

Consumes the not_wall-cleaned graph (wall_mask_review.py pkl) or, with --fresh,
the 4a+4b graph. Builds an "axis graph" by RDP-straightening every edge (4c) into
straight segments while KEEPING vertex identity (graph junctions are shared; DP
corners are new degree-2 vertices). Then:

  4d  angle regularisation: find the building's dominant orientation (angle
      histogram mod 90), rotate into that frame, classify each segment as
      Horizontal / Vertical / Diagonal (within --angle-tol).
  4e  junction snapping: union-find so every chain of collinear H segments shares
      one y and every chain of V segments shares one x (mean of the group). Each
      vertex's new position = (x from its V-group, y from its H-group), so axis
      walls become exactly H/V and the junctions where they meet land on one
      shared point (watertight). Diagonal segments impose no constraint and keep
      their (possibly moved) endpoints.

Writes a before(4c straight)/after(regularised) comparison panel. Diagnostic only.
"""

from __future__ import annotations

import argparse
import cmath
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from wall_mask_skeleton import add_pipeline_args, edge_thickness, find_masks, prepare_graph
from wall_mask_step4 import dissolve_degree2, global_thickness, prune_spurs
from wall_mask_step4c import rdp


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_axis_graph(graph: Any, dist: np.ndarray, dp_frac: float, global_t: float
                     ) -> tuple[dict[Any, np.ndarray], list[tuple[Any, Any, float]]]:
    """RDP-straighten each edge into segments, keeping shared vertex identity.

    Vertices: ("n", node) for graph junctions/endpoints (shared), ("c", k) for
    DP corners (internal to one edge). Returns (verts: vid->[y,x], edges).
    """
    verts: dict[Any, np.ndarray] = {}
    edges: list[tuple[Any, Any, float]] = []
    corner = [0]

    for u, v, data in graph.edges(data=True):
        pts_raw = np.asarray(data["pts"])
        if len(pts_raw) < 2:
            continue
        pts = pts_raw.astype(float)
        cu = np.asarray(graph.nodes[u]["o"], dtype=float)
        cv = np.asarray(graph.nodes[v]["o"], dtype=float)
        if np.hypot(*(pts[0] - cu)) > np.hypot(*(pts[0] - cv)):
            pts = pts[::-1]  # orient pts from u to v
        thickness = edge_thickness(dist, pts_raw) or global_t
        simp = rdp(pts, max(1.0, dp_frac * thickness))

        chain: list[Any] = []
        for i, pt in enumerate(simp):
            if i == 0:
                vid = ("n", u); verts[vid] = cu
            elif i == len(simp) - 1:
                vid = ("n", v); verts[vid] = cv
            else:
                vid = ("c", corner[0]); corner[0] += 1; verts[vid] = pt
            chain.append(vid)
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append((a, b, thickness))
    return verts, edges


def dominant_angle(verts: dict[Any, np.ndarray], edges: list[tuple[Any, Any, float]]) -> float:
    """Length-weighted dominant orientation in degrees, folded to [0, 90)."""
    z = 0j
    for a, b, _ in edges:
        d = verts[b] - verts[a]
        length = math.hypot(d[0], d[1])
        if length < 1e-6:
            continue
        ang = math.degrees(math.atan2(d[0], d[1])) % 90  # fold to [0,90)
        z += length * cmath.exp(1j * math.radians(4 * ang))  # 90-period -> full circle
    if z == 0:
        return 0.0
    theta = (math.degrees(cmath.phase(z)) / 4) % 90
    return theta - 90 if theta > 45 else theta  # minimal rotation, (-45, 45]


def _rot(p: np.ndarray, phi: float, c: np.ndarray) -> np.ndarray:
    """Rotate point [y,x] by phi (rad) about centre c [y,x]."""
    dy, dx = p[0] - c[0], p[1] - c[1]
    cs, sn = math.cos(phi), math.sin(phi)
    return np.array([c[0] + dx * sn + dy * cs, c[1] + dx * cs - dy * sn])


def regularize(verts: dict[Any, np.ndarray], edges: list[tuple[Any, Any, float]],
               angle_tol: float, d_merge_factor: float) -> tuple[dict[Any, np.ndarray], dict[str, int]]:
    """4d+4e: snap axis segments to H/V in the dominant frame and share gridlines.

    Short diagonal (D) segments (length < d_merge_factor * thickness) are collapsed
    — their two endpoints merge to one point (unioned in BOTH x and y) — to remove
    the little kinks near junctions; long diagonals stay free (real angled walls).
    """
    theta = dominant_angle(verts, edges)
    phi = -math.radians(theta)
    centre = np.mean(np.stack(list(verts.values())), axis=0)
    rot = {vid: _rot(p, phi, centre) for vid, p in verts.items()}

    ufx, ufy = UnionFind(), UnionFind()
    classes: list[str] = []
    for a, b, t in edges:
        d = rot[b] - rot[a]
        ang = math.degrees(math.atan2(d[0], d[1])) % 180  # [0,180)
        if min(ang, 180 - ang) <= angle_tol:        # horizontal: dy ~ 0
            ufy.union(a, b); classes.append("H")
        elif abs(ang - 90) <= angle_tol:            # vertical: dx ~ 0
            ufx.union(a, b); classes.append("V")
        elif math.hypot(d[0], d[1]) < d_merge_factor * t:  # short diagonal -> collapse
            ufx.union(a, b); ufy.union(a, b); classes.append("Dm")
        else:
            classes.append("D")                      # long diagonal -> keep free

    xs: dict[Any, list[float]] = defaultdict(list)
    ys: dict[Any, list[float]] = defaultdict(list)
    for vid, p in rot.items():
        xs[ufx.find(vid)].append(p[1])
        ys[ufy.find(vid)].append(p[0])
    xmean = {r: float(np.mean(v)) for r, v in xs.items()}
    ymean = {r: float(np.mean(v)) for r, v in ys.items()}

    inv = math.radians(theta)
    out: dict[Any, np.ndarray] = {}
    for vid in verts:
        snapped = np.array([ymean[ufy.find(vid)], xmean[ufx.find(vid)]])
        out[vid] = _rot(snapped, inv, centre)
    counts = {k: classes.count(k) for k in ("H", "V", "D", "Dm")}
    return out, {"theta": round(theta, 1), **counts}


def merge_collinear(verts: dict[Any, np.ndarray], edges: list[tuple[Any, Any, float]],
                    tol_deg: float = 8.0) -> list[tuple[Any, Any, float]]:
    """Merge the two walls at a degree-2 vertex when they are ~collinear.

    Exactly-collinear adjacent walls sharing an endpoint make the editor's wall
    miter (parallel offset edges) shoot to infinity — a spike that blows up the
    2D view. Dissolving such pass-through joints removes the spike and the
    redundant split. Real corners (perpendicular) are kept.
    """
    edges = [list(e) for e in edges]
    changed = True
    while changed:
        changed = False
        inc: dict[Any, list[int]] = defaultdict(list)
        for i, e in enumerate(edges):
            if e[0] is None:
                continue
            inc[e[0]].append(i)
            inc[e[1]].append(i)
        for v, ids in inc.items():
            ids = [i for i in ids if edges[i][0] is not None]
            if len(ids) < 2:
                continue
            # unit direction from v along each incident edge
            dirs = []
            for i in ids:
                a, b, t = edges[i]
                o = b if a == v else a
                d = verts[o] - verts[v]
                n = math.hypot(d[0], d[1])
                if n >= 1e-6:
                    dirs.append((i, o, d / n, t, n))
            # find ANY collinear-opposite pair at this vertex (degree may be >2: a
            # straight wall passing through a T-junction; the T-stem then ends on
            # the merged through-wall's body, which mitres cleanly).
            pair = None
            for p in range(len(dirs)):
                for q in range(p + 1, len(dirs)):
                    cos = float(np.clip(dirs[p][2] @ dirs[q][2], -1.0, 1.0))
                    if math.degrees(math.acos(cos)) >= 180 - tol_deg:
                        pair = (dirs[p], dirs[q])
                        break
                if pair:
                    break
            if not pair:
                continue
            (i1, o1, _, t1, n1), (i2, o2, _, t2, n2) = pair
            if o1 == o2 or o1 == v or o2 == v:
                continue
            edges[i1] = [o1, o2, (t1 * n1 + t2 * n2) / (n1 + n2)]
            edges[i2] = [None, None, 0]
            changed = True
            break
    return [(e[0], e[1], e[2]) for e in edges if e[0] is not None]


def draw_axes(ax: Any, base: np.ndarray | None, fallback: np.ndarray,
              verts: dict[Any, np.ndarray], edges: list[tuple[Any, Any, float]], title: str) -> None:
    ax.set_title(title)
    ax.imshow(base if base is not None else fallback, cmap=None if base is not None else "gray")
    for a, b, _ in edges:
        pa, pb = verts[a], verts[b]
        ax.plot([pa[1], pb[1]], [pa[0], pb[0]], color="#1f77b4", linewidth=1.6)
        ax.plot([pa[1], pb[1]], [pa[0], pb[0]], ".", color="#d62728", markersize=3)
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
    reg_verts, stats = regularize(verts, edges, args.angle_tol, args.d_merge_factor)

    # drop collapsed short-D segments, then merge collinear pass-throughs (miter-spike fix)
    kept = [(a, b, t) for a, b, t in edges if np.hypot(*(reg_verts[a] - reg_verts[b])) > 1e-6]
    kept = merge_collinear(reg_verts, kept)

    base = None
    if original is not None:
        base = (np.asarray(Image.fromarray(original).convert("RGB")).astype(float) * 0.45 + 255 * 0.55).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{sub_dir.name}  src={source}  theta={stats['theta']}deg  segments {len(edges)}->{len(kept)}  "
        f"(H{stats['H']} V{stats['V']} D{stats['D']} merged{stats['Dm']})",
        fontsize=12,
    )
    draw_axes(axes[0], base, connected, verts, edges, "4c straightened (input)")
    draw_axes(axes[1], base, connected, reg_verts, kept, "4d+4e + D-merge (H/V snapped, junctions shared)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"step4de_{args.tag}.png" if args.tag else "step4de_debug.png"
    out_png = sub_dir / name
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    return {"image": str(out_png), "segments": len(edges), "kept": len(kept), **stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    parser.add_argument("--k", type=float, default=1.5, help="4a spur length threshold = k * wall thickness.")
    parser.add_argument("--dp-frac", type=float, default=0.6, help="4c RDP tolerance = frac * wall thickness.")
    parser.add_argument("--angle-tol", type=float, default=20.0, help="degrees within H/V (in the dominant frame) to snap.")
    parser.add_argument("--d-merge-factor", type=float, default=1.5, help="collapse diagonal segments shorter than factor * wall thickness.")
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
            f"[{mask_svg.parent.name}] theta={info['theta']}deg segments {info['segments']}->{info['kept']} "
            f"(H{info['H']} V{info['V']} D{info['D']} merged{info['Dm']}) -> {info['image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
