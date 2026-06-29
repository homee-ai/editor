#!/usr/bin/env python3
"""Method A, steps 0-3: wall mask -> cleaned mask -> skeleton + thickness + graph.

Input is a wall_mask_vlm_v3.py output folder (one holding wall_mask.svg, or a
parent holding <stem>/wall_mask.svg). For each wall mask it runs:

  step 0  rasterise wall_mask.svg -> binary, morphological close + drop specks
  step 1  skeletonize -> 1px centerlines
  step 2  distance transform -> per-pixel wall thickness (2 x EDT)
  step 3  skeleton graph (sknw): junction/endpoint nodes + centerline edges,
          with a median thickness per edge

It writes a 2x3 comparison panel (original + steps 0-3) next to each mask so the
skeleton/graph quality can be eyeballed before doing regularisation / IFC.
Diagnostic only; produces no 3D geometry yet.
"""

from __future__ import annotations

import argparse
import json
import statistics
from io import BytesIO
from pathlib import Path
from typing import Any

import cairosvg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sknw
from PIL import Image
from scipy import ndimage
from skimage.morphology import skeletonize

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path
    return None


def find_masks(target: Path) -> list[Path]:
    if (target / "wall_mask.svg").exists():
        return [target / "wall_mask.svg"]
    return sorted(target.glob("*/wall_mask.svg"))


def canvas_size(mask_svg: Path, source_png: Path | None) -> tuple[int, int]:
    if source_png is not None:
        return Image.open(source_png).size
    # fall back to the svg's own width/height
    png = cairosvg.svg2png(url=str(mask_svg), background_color="white")
    image = Image.open(BytesIO(png))
    return image.size


DEFAULT_MASK_ROOT = REPO_ROOT / "tools/floor2ifc/fixtures/pngMask"


def rasterize_mask(mask_svg: Path, size: tuple[int, int]) -> np.ndarray:
    png = cairosvg.svg2png(url=str(mask_svg), output_width=size[0], output_height=size[1], background_color="white")
    gray = np.asarray(Image.open(BytesIO(png)).convert("L"))
    return gray < 128  # walls are drawn black on white


def load_openings(stem: str, mask_root: Path, size: tuple[int, int], dilate: int = 0,
                  classes: tuple[str, ...] = ("Door", "Window")) -> np.ndarray:
    """Union of the CubiCasa Door/Window icon masks for this image (filled openings)."""
    base = mask_root / stem
    paths: list[Path] = []
    for cls in classes:
        paths += sorted(base.glob(f"masks/*_{cls}.png")) + sorted(base.glob(f"*_{cls}.png"))
    union = np.zeros((size[1], size[0]), dtype=bool)
    for path in paths:
        m = Image.open(path).convert("L")
        if m.size != size:
            m = m.resize(size, Image.Resampling.NEAREST)
        union |= np.asarray(m) > 0
    if dilate > 0:
        union = ndimage.binary_dilation(union, structure=np.ones((dilate * 2 + 1, dilate * 2 + 1), dtype=bool))
    return union


def step0_clean(mask: np.ndarray, close_radius: int, min_area: int) -> np.ndarray:
    if close_radius > 0:
        structure = np.ones((close_radius * 2 + 1, close_radius * 2 + 1), dtype=bool)
        mask = ndimage.binary_closing(mask, structure=structure)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = {i for i in range(1, n + 1) if sizes[i] >= min_area}
    return np.isin(labels, list(keep))


def connect_clean_mask(
    kept: np.ndarray, removed: np.ndarray, close_radius: int, min_area: int, bridge_max_len: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reconnect the thin-removed mask.

    Restore removed thin pieces that BRIDGE >=2 separate kept components AND are
    short (a wrongly-cut wall neck, not a long thin wall or a hatching mesh),
    then morphological-close residual gaps and drop tiny components.
    Returns (connected_mask, restored_pieces).
    """
    s8 = np.ones((3, 3), dtype=bool)
    kept_lbl, _ = ndimage.label(kept, structure=s8)
    rem_lbl, n_rem = ndimage.label(removed, structure=s8)
    restored = np.zeros_like(kept)
    height, width = kept.shape
    for idx, sl in enumerate(ndimage.find_objects(rem_lbl), start=1):
        if sl is None:
            continue
        if max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) > bridge_max_len:
            continue  # too long to be a gap-filling bridge (e.g. hatching line)
        rows = slice(max(0, sl[0].start - 1), min(height, sl[0].stop + 1))
        cols = slice(max(0, sl[1].start - 1), min(width, sl[1].stop + 1))
        piece = rem_lbl[rows, cols] == idx
        touching = np.unique(kept_lbl[rows, cols][ndimage.binary_dilation(piece, structure=s8)])
        if touching[touching > 0].size >= 2:  # piece links two kept components -> keep it
            restored[rows, cols] |= piece

    connected = kept | restored
    if close_radius > 0:
        st = np.ones((close_radius * 2 + 1, close_radius * 2 + 1), dtype=bool)
        connected = ndimage.binary_closing(connected, structure=st)
    labels, n = ndimage.label(connected, structure=s8)
    if n:
        sizes = np.bincount(labels.ravel())
        keep_ids = [i for i in range(1, n + 1) if sizes[i] >= min_area]
        connected = np.isin(labels, keep_ids)
    return connected, restored


def edge_thickness(dist: np.ndarray, pts: np.ndarray) -> float:
    # pts are [row, col]; thickness = 2 x EDT. Trim the ends (junction inflation).
    values = 2.0 * dist[pts[:, 0], pts[:, 1]]
    if len(values) > 6:
        cut = max(1, len(values) // 6)
        values = values[cut:-cut]
    return float(statistics.median(values)) if len(values) else 0.0


def build_graph(skeleton: np.ndarray, dist: np.ndarray) -> tuple[Any, list[dict[str, Any]]]:
    # multi=True keeps BOTH edges of a loop (bay window, closed room, walled
    # void): such a loop is two parallel routes between the same pair of
    # junctions, and multi=False would silently drop one of them.
    graph = sknw.build_sknw(skeleton.astype(np.uint8), multi=True, iso=True, ring=True)
    edges: list[dict[str, Any]] = []
    for s, e, data in graph.edges(data=True):
        pts = data["pts"]
        edges.append({
            "s": s,
            "e": e,
            "pts": pts,
            "thickness_px": round(edge_thickness(dist, pts), 2),
            "length_px": round(float(data.get("weight", len(pts))), 2),
        })
    return graph, edges


def render_panels(
    out_png: Path,
    title: str,
    original: np.ndarray | None,
    kept: np.ndarray,
    openings: np.ndarray,
    restored: np.ndarray,
    dropped: np.ndarray,
    connected: np.ndarray,
    skeleton: np.ndarray,
    thickness: np.ndarray,
    graph: Any,
    min_thickness: float,
) -> None:
    """Thin removal (+ door/window merge + bridge restore) -> connected mask -> graph."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(title, fontsize=12)
    faded = None
    if original is not None:
        faded = (np.asarray(Image.fromarray(original).convert("RGB")).astype(float) * 0.4 + 255 * 0.6).astype(np.uint8)

    ax = axes[0, 0]
    ax.set_title("original")
    if original is not None:
        ax.imshow(original)
    ax.axis("off")

    ax = axes[0, 1]
    ax.set_title(f"cut<{min_thickness:.1f}px  blue=keep orange=door/window green=bridge red=drop")
    canvas = faded.copy() if faded is not None else np.full((*kept.shape, 3), 255, dtype=np.uint8)
    canvas[dropped] = (231, 76, 60)               # dropped thin clutter
    canvas[kept] = (40, 120, 255)                 # kept thick walls
    canvas[openings & ~kept] = (245, 130, 49)     # merged door/window openings
    canvas[restored] = (46, 204, 113)             # restored bridges
    ax.imshow(canvas)
    ax.axis("off")

    ax = axes[0, 2]
    ax.set_title("step0 connected clean mask (+door/window)")
    ax.imshow(connected, cmap="gray")
    ax.axis("off")

    ax = axes[1, 0]
    ax.set_title("step2 thickness (px) = 2 x EDT")
    im = ax.imshow(np.where(connected, thickness, np.nan), cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")

    ax = axes[1, 1]
    ax.set_title("step1 skeleton (cleaned)")
    ax.imshow(skeleton, cmap="gray")
    ax.axis("off")

    ax = axes[1, 2]
    ax.set_title("step3 graph (red=junction, green=end)")
    ax.imshow(faded if faded is not None else kept, cmap=None if faded is not None else "gray")
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color="#1f77b4", linewidth=1.2)
    for node in graph.nodes():
        y, x = graph.nodes[node]["o"]
        deg = graph.degree(node)
        color = "#2ca02c" if deg == 1 else ("#d62728" if deg >= 3 else "#ff7f0e")
        ax.plot(x, y, "o", color=color, markersize=4)
    ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def add_pipeline_args(parser: argparse.ArgumentParser) -> None:
    """Steps 0-3 knobs, shared by wall_mask_skeleton.py and wall_mask_step4.py."""
    parser.add_argument("--thickness-frac", type=float, default=0.5, help="auto threshold = frac x median thickness (per image).")
    parser.add_argument("--min-thickness", type=float, default=None, help="absolute thickness cut (px); overrides --thickness-frac.")
    parser.add_argument("--close-radius", type=int, default=2, help="step0 closing radius to bridge residual gaps (px).")
    parser.add_argument("--min-area", type=int, default=30, help="step0 drop connected components smaller than this (px).")
    parser.add_argument("--bridge-max-len", type=float, default=None, help="max bbox side of a restored bridge piece (px); default max(6, 2xcut).")
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT, help="folder with <stem>/masks/*_Door.png, *_Window.png.")
    parser.add_argument("--opening-dilate", type=int, default=0, help="dilate door/window masks by this many px before merging.")


def prepare_graph(mask_svg: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Run steps 0-3 (clean + thin-removal + opening merge + connect + graph).

    Returns the arrays, the skeleton graph, and its distance transform so callers
    (skeleton debug panel, step4 regularisation) share one cleaned pipeline.
    """
    sub_dir = mask_svg.parent
    source_png = None
    result_json = sub_dir / "result.json"
    if result_json.exists():
        rel = json.loads(result_json.read_text(encoding="utf-8")).get("source_png", "")
        source_png = resolve_existing(Path(rel), REPO_ROOT / rel, Path.cwd() / rel)

    size = canvas_size(mask_svg, source_png)
    mask = rasterize_mask(mask_svg, size)  # step0 cleaning removed: use the raw mask
    skeleton = skeletonize(mask)
    dist = ndimage.distance_transform_edt(mask)
    thickness_map = dist * 2.0
    skel_thickness = thickness_map[skeleton]  # one thickness value per skeleton pixel

    # threshold from the per-image thickness distribution: T = frac x median
    # (absolute --min-thickness overrides). median sits on the real walls, so a
    # fraction of it cuts the thin cluster without touching a uniform-thick plan.
    median_t = float(np.median(skel_thickness)) if len(skel_thickness) else 0.0
    if args.min_thickness is not None:
        threshold = float(args.min_thickness)
        basis = "abs"
    else:
        threshold = max(2.0, round(args.thickness_frac * median_t, 1))
        basis = f"{args.thickness_frac:g}xmedian({median_t:.1f})"

    # method 2: every mask pixel inherits the thickness of its nearest skeleton
    # pixel (the wall it belongs to); drop pixels whose owning wall is too thin.
    _edt, inds = ndimage.distance_transform_edt(~skeleton, return_indices=True)
    owner_thickness = thickness_map[inds[0], inds[1]]
    removed = mask & (owner_thickness < threshold)
    kept = mask & ~removed

    # merge the door/window opening masks into the clean mask so walls stay
    # continuous through openings (they were thin and got cut above).
    stem = Path(source_png).stem if source_png else sub_dir.name
    openings = load_openings(stem, args.mask_root, size, args.opening_dilate)
    merged = kept | openings
    removed_pool = removed & ~openings  # door/window pixels are not "dropped clutter"

    # step0 (after thin removal + opening merge): reconnect — restore SHORT removed
    # pieces that bridge two components, then close residual gaps + drop tiny ones.
    bridge_max_len = args.bridge_max_len if args.bridge_max_len is not None else max(6.0, 2.0 * threshold)
    connected, restored = connect_clean_mask(merged, removed_pool, args.close_radius, args.min_area, bridge_max_len)
    dropped = removed_pool & ~restored

    # continue on the connected clean mask: re-skeletonize and build the step3 graph
    skeleton2 = skeletonize(connected)
    dist2 = ndimage.distance_transform_edt(connected)
    graph, edges = build_graph(skeleton2, dist2)

    original = np.asarray(Image.open(source_png).convert("RGB")) if source_png else None
    removed_frac = float(removed.sum() / max(int(mask.sum()), 1))
    return {
        "sub_dir": sub_dir,
        "source_png": source_png,
        "original": original,
        "size": size,
        "kept": kept,
        "openings": openings,
        "restored": restored,
        "dropped": dropped,
        "connected": connected,
        "skeleton": skeleton2,
        "dist": dist2,
        "graph": graph,
        "edges": edges,
        "median_t": median_t,
        "threshold": threshold,
        "basis": basis,
        "removed_frac": removed_frac,
    }


def process_mask(mask_svg: Path, args: argparse.Namespace) -> dict[str, Any]:
    p = prepare_graph(mask_svg, args)
    graph, edges = p["graph"], p["edges"]
    n_junction = sum(1 for n in graph.nodes() if graph.degree(n) >= 3)
    n_endpoint = sum(1 for n in graph.nodes() if graph.degree(n) == 1)

    out_png = p["sub_dir"] / "skeleton_debug.png"
    title = (
        f"{p['sub_dir'].name}  median={p['median_t']:.1f}px  cut<{p['threshold']:.1f}px [{p['basis']}]  "
        f"removed {p['removed_frac']:.0%}  +door/window  ->  "
        f"nodes={graph.number_of_nodes()} (junc {n_junction}, end {n_endpoint}) edges={len(edges)}"
    )
    render_panels(
        out_png, title, p["original"], p["kept"], p["openings"], p["restored"], p["dropped"],
        p["connected"], p["skeleton"], p["dist"] * 2.0, graph, p["threshold"],
    )

    return {
        "mask": str(mask_svg),
        "image": str(out_png),
        "median_thickness_px": round(p["median_t"], 2),
        "cut_threshold_px": round(p["threshold"], 2),
        "removed_frac": round(p["removed_frac"], 4),
        "nodes": graph.number_of_nodes(),
        "junctions": n_junction,
        "endpoints": n_endpoint,
        "edges": len(edges),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    args = parser.parse_args()

    masks: list[Path] = []
    for target in args.target:
        if not target.exists():
            print(f"Not found: {target}")
            continue
        masks.extend(find_masks(target))
    if not masks:
        print("No wall_mask.svg found.")
        return 2

    for mask_svg in masks:
        info = process_mask(mask_svg, args)
        print(
            f"[{Path(info['mask']).parent.name}] median={info['median_thickness_px']}px "
            f"cut<{info['cut_threshold_px']}px removed={info['removed_frac']:.0%} -> "
            f"nodes={info['nodes']} (junc {info['junctions']}, end {info['endpoints']}) edges={info['edges']} "
            f"-> {info['image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
