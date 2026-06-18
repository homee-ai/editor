#!/usr/bin/env python3
"""Extract walls from floor-plan PNGs and trace them to SVG.

Walls vs. furniture are separated on the *raster*, by stroke thickness, before
vtracer ever runs: thin ink (furniture lines, dimensions, text, hatching) is
removed by a morphological opening sized to the wall thickness, then isolated
blobs that survive are dropped by connected-component area. Only the connected
wall network is handed to vtracer, which traces it into clean polygons.

The opening radius is derived per-image from the distance transform, so it
adapts to plan scale instead of being hard-coded.

Usage:
    python png2wallsvg.py                 # fixtures/png/*.png -> fixtures/wall-svg/
    python png2wallsvg.py -i dir -o out
    python png2wallsvg.py --radius 3      # override auto thickness estimate
    python png2wallsvg.py --no-trace      # only emit the cleaned wall PNG masks
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import vtracer
from PIL import Image
from scipy import ndimage as ndi

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "fixtures" / "png"
DEFAULT_OUT = HERE / "fixtures" / "wall-svg"

INK_THRESHOLD = 128       # grayscale < this = ink (black)
MIN_BLOB_FRAC = 0.002     # drop connected blobs smaller than this fraction of the image


def load_ink(src: Path) -> np.ndarray:
    """Return a boolean ink mask (True = dark stroke), compositing transparency on white."""
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    gray = np.asarray(im.convert("L"))
    return gray < INK_THRESHOLD


def estimate_radius(ink: np.ndarray) -> int:
    """Opening radius from the ink half-thickness (distance transform).

    Walls are the thick strokes; the 95th percentile distance is a robust proxy
    for the wall half-thickness. An opening at ~40% of that erases thin furniture
    lines while leaving walls intact, and never drops below 2px.
    """
    dt = ndi.distance_transform_edt(ink)
    wall_half = np.percentile(dt[ink], 95) if ink.any() else 0.0
    return max(2, round(wall_half * 0.4))


def extract_walls(ink: np.ndarray, radius: int) -> np.ndarray:
    """Keep the connected thick-stroke network; drop thin ink and isolated blobs."""
    st = ndi.generate_binary_structure(2, 1)
    opened = ndi.binary_erosion(ink, st, iterations=radius)
    opened = ndi.binary_dilation(opened, st, iterations=radius)

    labels, n = ndi.label(opened)
    if n == 0:
        return opened
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    min_size = MIN_BLOB_FRAC * ink.size
    keep_ids = {i for i, s in enumerate(sizes, start=1) if s >= min_size}
    return np.isin(labels, list(keep_ids))


def mask_to_png(mask: np.ndarray, path: Path) -> None:
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def trace_walls(mask_png: Path, svg_path: Path) -> None:
    vtracer.convert_image_to_svg_py(
        str(mask_png), str(svg_path),
        colormode="binary", mode="polygon", filter_speckle=2,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract walls (morphology) and trace to SVG")
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--radius", type=int, default=None,
                    help="opening radius in px (default: auto from distance transform)")
    ap.add_argument("--no-trace", action="store_true", help="emit cleaned wall PNG only")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.input.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no PNG files in {args.input}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for src in pngs:
            ink = load_ink(src)
            radius = args.radius if args.radius is not None else estimate_radius(ink)
            walls = extract_walls(ink, radius)
            kept = walls.sum() / max(ink.sum(), 1)

            mask_png = args.output / (src.stem + "_wall_mask.png")
            mask_to_png(walls, mask_png)

            if args.no_trace:
                print(f"{src.name:>10}  r={radius:<2}  kept {kept:5.0%} of ink  -> {mask_png.name}")
                continue

            svg = args.output / (src.stem + "_wall.svg")
            tmp_png = tmp / (src.stem + "_mask.png")
            mask_to_png(walls, tmp_png)
            trace_walls(tmp_png, svg)
            print(f"{src.name:>10}  r={radius:<2}  kept {kept:5.0%} of ink  -> {svg.name} ({svg.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
