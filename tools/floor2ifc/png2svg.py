#!/usr/bin/env python3
"""Batch-convert PNG floorplans to SVG using vtracer.

Usage:
    python png2svg.py                          # colour cutout: many distinct objects (default)
    python png2svg.py --color-precision 8      # more, finer objects
    python png2svg.py --filter-speckle 12      # fewer fragments (for noisy renders)
    python png2svg.py --binary                 # black/white polygon trace (sharp lines)
    python png2svg.py -i some/dir -o out       # custom input/output dirs
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import vtracer
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE / "fixtures" / "png"
DEFAULT_OUT = HERE / "fixtures" / "svg"


def normalize_to_png(src: Path, tmp_dir: Path) -> Path:
    """Re-encode any image (JPEG/WEBP/palette/RGBA) to a clean RGB PNG.

    vtracer decodes by file extension and chokes on mislabelled or
    transparent images, so we composite onto white and write true PNG.
    """
    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im).convert("RGB")
    else:
        im = im.convert("RGB")
    out = tmp_dir / (src.stem + ".png")
    im.save(out, format="PNG")
    return out


def convert(src: Path, dst: Path, binary: bool, filter_speckle: int,
            color_precision: int, tmp_dir: Path) -> None:
    png = normalize_to_png(src, tmp_dir)
    if binary:
        # Sharp black/white line tracing. NOTE: connected black ink fuses into a
        # single compound <path>, so walls and touching furniture become one
        # un-separable object. Prefer the colour path for per-object workflows.
        opts = dict(colormode="binary", mode="polygon", filter_speckle=filter_speckle)
    else:
        # Colour + cutout: emit many distinct, non-overlapping objects while
        # preserving the image, so a downstream classifier can judge each one.
        # filter_speckle drops tiny fragments; color_precision raises object count.
        opts = dict(colormode="color", mode="polygon", hierarchical="cutout",
                    filter_speckle=filter_speckle, color_precision=color_precision)
    vtracer.convert_image_to_svg_py(str(png), str(dst), **opts)


def main() -> None:
    ap = argparse.ArgumentParser(description="PNG -> SVG via vtracer")
    ap.add_argument("-i", "--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--binary", action="store_true",
                    help="black/white polygon trace instead of colour cutout")
    ap.add_argument("--filter-speckle", type=int, default=4,
                    help="drop blobs up to this px size (raise to reduce fragments)")
    ap.add_argument("--color-precision", type=int, default=6,
                    help="colour bits in colour mode (raise for more, finer objects)")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.input.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no PNG files in {args.input}")

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for src in pngs:
            dst = args.output / (src.stem + ".svg")
            convert(src, dst, args.binary, args.filter_speckle,
                    args.color_precision, tmp_dir)
            objects = dst.read_text().count("<path")
            print(f"{src.name:>10}  ->  {dst.name:<12} "
                  f"({dst.stat().st_size/1024:6.1f} KB, {objects} objects)")


if __name__ == "__main__":
    main()
