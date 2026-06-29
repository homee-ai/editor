#!/usr/bin/env python3
"""Convert a pixel-space wall_model.json into a Pascal editor scene.

px -> metres via --px-per-meter. Emits next to the input:
  walls_m.json  flat list of metre-space walls with WallNode-ready fields
                ({start:[x,y], end:[x,y], thickness, height}).
  scene.json    a loadable Pascal scene graph (Site -> Building -> Level -> Walls),
                ids in the <prefix>_<id> form the schema accepts.

Image coords are [row=y, col=x]; editor wall points are [x, y] in level space. With
--flip-y the y axis is flipped about the canvas height (json's "canvas":[H,W]).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def gen_id(prefix: str, i: int) -> str:
    return f"{prefix}_{i:012d}"


def to_metres(data: dict[str, Any], px_per_meter: float, wall_height: float, flip_y: bool,
              center: bool = True) -> list[dict[str, Any]]:
    scale = 1.0 / px_per_meter
    canvas = data.get("canvas")
    height_px = float(canvas[0]) if canvas else max(
        (max(w["y1"], w["y2"]) for w in data["walls"]), default=0.0)
    pts: list[tuple[float, float, float, float, float]] = []
    for w in data["walls"]:
        y1, y2 = w["y1"], w["y2"]
        if flip_y:
            y1, y2 = height_px - y1, height_px - y2
        pts.append((w["x1"] * scale, y1 * scale, w["x2"] * scale, y2 * scale, w["t"] * scale))
    # centre the plan on the origin so the editor's default camera frames it (2D + 3D)
    cx = cy = 0.0
    if center and pts:
        xs = [p[0] for p in pts] + [p[2] for p in pts]
        ys = [p[1] for p in pts] + [p[3] for p in pts]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
    return [{
        "start": [round(x1 - cx, 4), round(y1 - cy, 4)],
        "end": [round(x2 - cx, 4), round(y2 - cy, 4)],
        "thickness": round(t, 4),
        "height": wall_height,
    } for (x1, y1, x2, y2, t) in pts]


def build_scene(walls_m: list[dict[str, Any]]) -> dict[str, Any]:
    site, building, level = gen_id("site", 0), gen_id("building", 0), gen_id("level", 0)
    nodes: dict[str, Any] = {}
    wall_ids: list[str] = []
    for i, w in enumerate(walls_m):
        wid = gen_id("wall", i)
        wall_ids.append(wid)
        nodes[wid] = {
            "object": "node", "id": wid, "type": "wall", "parentId": level,
            "visible": True, "metadata": {}, "children": [],
            "start": w["start"], "end": w["end"],
            "thickness": w["thickness"], "height": w["height"],
            "frontSide": "unknown", "backSide": "unknown",
        }
    nodes[level] = {"object": "node", "id": level, "type": "level", "parentId": building,
                    "visible": True, "metadata": {}, "children": wall_ids, "level": 0}
    nodes[building] = {"object": "node", "id": building, "type": "building", "parentId": site,
                       "visible": True, "metadata": {}, "children": [level],
                       "position": [0, 0, 0], "rotation": [0, 0, 0]}
    nodes[site] = {"object": "node", "id": site, "type": "site", "parentId": None,
                   "visible": True, "metadata": {}, "children": [building],
                   "polygon": {"type": "polygon", "points": [[-15, -15], [15, -15], [15, 15], [-15, 15]]}}
    return {"nodes": nodes, "rootNodeIds": [site]}


def convert(wall_model_path: Path, px_per_meter: float, wall_height: float, flip_y: bool,
            out_dir: Path | None = None, center: bool = True) -> dict[str, Any]:
    data = json.loads(wall_model_path.read_text(encoding="utf-8"))
    walls_m = to_metres(data, px_per_meter, wall_height, flip_y, center)
    scene = build_scene(walls_m)
    out = out_dir or wall_model_path.parent
    (out / "walls_m.json").write_text(
        json.dumps({"px_per_meter": px_per_meter, "wall_height": wall_height,
                    "flip_y": flip_y, "walls": walls_m}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / "scene.json").write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"walls": len(walls_m), "walls_m": str(out / "walls_m.json"), "scene": str(out / "scene.json")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wall_model", type=Path, nargs="+", help="wall_model.json file(s) or folder(s) holding one.")
    ap.add_argument("--px-per-meter", type=float, default=60.0)
    ap.add_argument("--wall-height", type=float, default=2.5)
    ap.add_argument("--flip-y", action="store_true", help="flip the y axis about the canvas height.")
    ap.add_argument("--no-center", action="store_true", help="keep pixel-origin coords (default centres the plan on the origin).")
    args = ap.parse_args()

    paths: list[Path] = []
    for p in args.wall_model:
        if p.is_dir():
            paths += sorted(p.glob("**/wall_model.json"))
        elif p.exists():
            paths.append(p)
    if not paths:
        print("No wall_model.json found.")
        return 2

    for path in paths:
        info = convert(path, args.px_per_meter, args.wall_height, args.flip_y, center=not args.no_center)
        print(f"[{path.parent.name}] {info['walls']} walls -> {info['scene']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
