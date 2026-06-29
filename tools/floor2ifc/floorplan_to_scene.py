#!/usr/bin/env python3
"""End-to-end: one floorplan PNG -> Pascal scene.json (+ wall_model / footprint).

Orchestrates the whole chain across the two conda envs via subprocess:

  0  CubiCasa5k segmentation            (cubicasa env)  -> masks/ + tags.json
  1-2 png2svg + png_mask_svg_candidates (floor2ifc env) -> candidate_report.json
  3  wall_mask_vlm_v3  (VLM)                            -> wall_mask.svg
  4  wall_mask_review --task not_wall (VLM)             -> review_graph_not_wall.pkl
  5  wall_mask_pipeline (geometry)                      -> wall_model.json
  6  wall_model_to_scene (px -> metres)                 -> walls_m.json + scene.json

Everything for one PNG lands under <work-root>/<stem>/. Stages 3 and 4 call a VLM
and take minutes. Use --skip-cubicasa / --from to resume.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
F2I = REPO_ROOT / "tools/floor2ifc"
CUBI = REPO_ROOT / "tools/CubiCasa5k"
DEFAULT_CUBI_PY = Path(os.environ.get("CUBICASA_PYTHON") or Path.home() / "miniforge3/envs/cubicasa/bin/python")
DEFAULT_F2I_PY = Path(os.environ.get("FLOOR2IFC_PYTHON") or Path.home() / "miniforge3/envs/floor2ifc/bin/python")
STAGES = ["cubicasa", "candidates", "vlm", "notwall", "pipeline", "scene"]


def run(cmd: list[str], cwd: Path, label: str, extra_env: dict[str, str] | None = None) -> None:
    print(f"\n=== [{label}] ===\n>> {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    env = {**os.environ, **(extra_env or {})}
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True, env=env)
    print(f"=== [{label}] done in {time.time() - t0:.0f}s ===", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("png", type=Path, help="floorplan PNG.")
    ap.add_argument("--work-root", type=Path, default=F2I / "out/e2e", help="output root; per-PNG dir is <root>/<stem>.")
    ap.add_argument("--cubicasa-python", type=Path, default=DEFAULT_CUBI_PY)
    ap.add_argument("--floor2ifc-python", type=Path, default=DEFAULT_F2I_PY)
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto", help="CubiCasa device.")
    # VLM config
    ap.add_argument("--vlm-env-root", type=Path, default=REPO_ROOT, help="dir with .env.local for stage 3 (candidate labelling).")
    ap.add_argument("--review-env-file", type=Path, default=REPO_ROOT / ".envOpenAI.local", help="env file for stage 4 not_wall (omit if missing).")
    ap.add_argument("--review-effort", default="low", help="stage 4 reasoning effort.")
    # px -> metres
    ap.add_argument("--px-per-meter", type=float, default=60.0)
    ap.add_argument("--wall-height", type=float, default=2.5)
    ap.add_argument("--flip-y", action="store_true")
    # control
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default="cubicasa", help="resume from this stage.")
    ap.add_argument("--skip-cubicasa", action="store_true", help="alias for --from candidates (masks already present).")
    args = ap.parse_args()

    png = args.png.resolve()
    if not png.exists():
        print(f"PNG not found: {png}")
        return 2
    stem = png.stem
    work = (args.work_root / stem).resolve()
    mask_dir = work / "cubicasa"
    cand_dir = work / "candidates"
    vlm_dir = work / "vlm"
    result_dir = vlm_dir / stem  # holds wall_mask.svg, result.json, pkl, wall_model.json, scene.json
    for d in (mask_dir, cand_dir, vlm_dir):
        d.mkdir(parents=True, exist_ok=True)

    start = STAGES.index("candidates" if args.skip_cubicasa else args.from_stage)
    cubi, f2i = args.cubicasa_python, args.floor2ifc_python

    def active(name: str) -> bool:
        return STAGES.index(name) >= start

    if active("cubicasa"):
        run([cubi, "cubi_predict.py", png, "--out-dir", mask_dir, "--device", args.device],
            cwd=CUBI, label="0 CubiCasa", extra_env={"KMP_DUPLICATE_LIB_OK": "TRUE"})
    if active("candidates"):
        run([f2i, "png_mask_svg_candidates.py", png, mask_dir / stem, "--out-dir", cand_dir],
            cwd=F2I, label="1-2 candidates")
    if active("vlm"):
        run([f2i, "wall_mask_vlm_v3.py", cand_dir / stem / "candidate_report.json",
             "--out-dir", vlm_dir, "--env-file-root", args.vlm_env_root],
            cwd=F2I, label="3 vlm_v3 (VLM)")
    if active("notwall"):
        cmd = [f2i, "wall_mask_review.py", result_dir, "--task", "not_wall",
               "--mask-root", mask_dir, "--reasoning-effort", args.review_effort]
        if args.review_env_file and Path(args.review_env_file).exists():
            cmd += ["--env-file", args.review_env_file, "--max-tokens", "64000"]
        run(cmd, cwd=F2I, label="4 not_wall (VLM)")
    if active("pipeline"):
        run([f2i, "wall_mask_pipeline.py", result_dir, "--mask-root", mask_dir],
            cwd=F2I, label="5 pipeline (geometry)")
    if active("scene"):
        run([f2i, "wall_model_to_scene.py", result_dir / "wall_model.json",
             "--px-per-meter", args.px_per_meter, "--wall-height", args.wall_height]
            + (["--flip-y"] if args.flip_y else []),
            cwd=F2I, label="6 px->metres scene")

    scene = result_dir / "scene.json"
    print(f"\nDONE. scene: {scene}\n      walls: {result_dir / 'walls_m.json'}\n"
          f"      model: {result_dir / 'wall_model.json'}\n      footprint: {result_dir / 'pipeline_debug.png'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
