#!/usr/bin/env python3
"""Method A, 4b.5: VLM endpoint review before straightening.

Pipeline: prepare_graph (0-3) -> 4a prune spurs -> 4b dissolve degree-2 ->
THIS (VLM endpoint review) -> (later) 4c straighten.

After 4b the only nodes left are junctions (deg>=3) and endpoints (deg==1).
Endpoints are topology defects: a wall that ends in mid-air because the mask had
a gap, plus the occasional non-wall stub the earlier filters missed. We ask a VLM
to look at each GREEN endpoint and decide:

  1. is the segment hanging off this endpoint actually a wall?
  2. should this endpoint connect to something to complete a wall? if so, to a
     node (number) or onto a wall segment (a pair of numbers -> T-junction).

Numbering (option b): every green endpoint is numbered, plus the candidate target
nodes near it (both endpoints of any edge passing within review-radius of a green
endpoint) so the VLM has numbers to point at. Two images go to the VLM: the clean
original (semantics) and the numbered graph (topology).

Edits are applied in a single pass (no re-asking newly created endpoints) and a
before/after panel is written next to each mask. Diagnostic only.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wall_mask_skeleton import add_pipeline_args, find_masks, prepare_graph
from wall_mask_step4 import dissolve_degree2, global_thickness, prune_spurs, path_length
from vlm_client import VlmClient, extract_json_object, normalize_reasoning_effort, parse_env_file


def png_data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def endpoints_and_junctions(graph: Any) -> tuple[list[Any], list[Any]]:
    ends = [n for n in graph.nodes() if graph.degree(n) == 1]
    juncs = [n for n in graph.nodes() if graph.degree(n) >= 3]
    return ends, juncs


def build_label_map(graph: Any, ends: list[Any], radius: float, include_targets: bool = True) -> dict[int, Any]:
    """Number every green endpoint + (optionally) candidate targets near it.

    A candidate target is a node that is an endpoint of an edge passing within
    `radius` of a green endpoint, so the wall an endpoint should T into has both
    its defining nodes numbered. The not_wall task needs only the endpoints, so
    pass include_targets=False to keep the image clean.
    """
    if not include_targets:
        return {i: node for i, node in enumerate(ends)}
    selected: set[Any] = set(ends)
    end_coords = {n: np.asarray(graph.nodes[n]["o"], dtype=float) for n in ends}
    for u, v, data in graph.edges(data=True):
        pts = np.asarray(data["pts"], dtype=float)
        if len(pts) == 0:
            continue
        for ec in end_coords.values():
            d = np.hypot(pts[:, 0] - ec[0], pts[:, 1] - ec[1]).min()
            if d <= radius:
                selected.add(u)
                selected.add(v)
                break
    # stable ordering: endpoints first (so their numbers are low / memorable)
    ordered = list(ends) + [n for n in selected if n not in set(ends)]
    return {i: node for i, node in enumerate(ordered)}


def _place_labels(ax: Any, items: list[dict[str, Any]]) -> None:
    """Greedily place number labels in free space with a leader line to the dot.

    Each label is pushed outward (endpoints first, near their dot) until its box
    no longer overlaps an already-placed label, so clustered numbers spread out
    instead of stacking. A thin leader links the label back to its node.
    """
    placed: list[tuple[float, float, float, float]] = []
    radii = [0, 11, 18, 26, 36, 48, 62, 80, 100]
    angles = [-90, -60, -120, -30, -150, 0, 180, 30, 150, 60, 120, 90]
    for it in sorted(items, key=lambda d: not d["is_end"]):  # endpoints first
        hw = 0.55 * it["fontsize"] * max(1, len(it["text"]))
        hh = 0.8 * it["fontsize"]
        pos = None
        for r in radii:
            for a in angles:
                rad = math.radians(a)
                lx = it["x"] + (r + hw) * math.cos(rad)
                ly = it["y"] + (r + hh) * math.sin(rad)
                if all(abs(lx - px) >= hw + phw + 1 or abs(ly - py) >= hh + phh + 1
                       for px, py, phw, phh in placed):
                    pos = (lx, ly)
                    break
            if pos:
                break
        lx, ly = pos if pos else (it["x"] + hw + 4, it["y"])
        if (lx - it["x"]) ** 2 + (ly - it["y"]) ** 2 > 36:
            ax.plot([it["x"], lx], [it["y"], ly], color=it["color"], lw=0.5, alpha=0.5, zorder=2)
        ax.text(lx, ly, it["text"], fontsize=it["fontsize"], fontweight=it["weight"],
                color=it["color"], ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        placed.append((lx, ly, hw, hh))


DEFAULT_COLORS = (
    "red:#E6194B,orange:#F58231,green:#3CB44B,blue:#4363D8,"
    "purple:#911EB4,magenta:#F032E6,brown:#9A6324,teal:#469990"
)


def parse_colors(spec: str) -> list[tuple[str, str]]:
    """'red:#E6194B,blue:#4363D8' -> [('red','#E6194B'), ('blue','#4363D8')]."""
    out: list[tuple[str, str]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, _, hexv = tok.partition(":")
        out.append((name.strip(), hexv.strip() or name.strip()))
    return out


def render_color_batch(out_png: Path, base: np.ndarray, graph: Any,
                       batch: list[tuple[str, str, Any]],
                       line_width: float = 2.6, line_alpha: float = 0.55) -> None:
    """not_wall view: whole wall network faint GRAY (context); this batch's
    endpoint segments each drawn in a distinct colour (no labels), semi-transparent
    and thin so the underlying wall in the original stays visible. The VLM judges
    the coloured lines, referenced by colour name.
    """
    h, w = base.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 70, h / 70))
    ax.imshow(base)
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color="#9aa0a6", linewidth=0.8, zorder=2)
    for _, hexv, node in batch:
        inc = list(graph.edges(node, keys=True))
        if not inc:
            continue
        u, v, k = inc[0]
        pts = np.asarray(graph[u][v][k]["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color=hexv, linewidth=line_width, alpha=line_alpha,
                    zorder=4, solid_capstyle="round")
        y, x = graph.nodes[node]["o"]
        ax.plot(x, y, "o", color=hexv, markersize=5, alpha=line_alpha, zorder=5)  # loose end (spots short stubs)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_numbered(out_png: Path, base: np.ndarray, graph: Any, label_map: dict[int, Any],
                    end_set: set[Any]) -> None:
    h, w = base.shape[:2]
    fig, ax = plt.subplots(figsize=(w / 70, h / 70))
    ax.imshow(base)
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color="#1f77b4", linewidth=1.2)
    items: list[dict[str, Any]] = []
    for label, node in label_map.items():
        y, x = graph.nodes[node]["o"]
        is_end = node in end_set
        ax.plot(x, y, "o", color="#2ca02c" if is_end else "#888888",
                markersize=8 if is_end else 4, zorder=3)
        items.append({"x": x, "y": y, "text": str(label), "is_end": is_end,
                     "fontsize": 13 if is_end else 9,
                     "weight": "bold" if is_end else "normal",
                     "color": "#0a7d12" if is_end else "#555555"})
    _place_labels(ax, items)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def build_notwall_color_prompt(color_names: list[str]) -> str:
    colors = ", ".join(color_names)
    return (
        "You are checking line segments traced from a floor plan to see which are "
        "real walls.\n\n"
        "Image 1 is the clean original floor plan. Image 2 draws the traced wall "
        "network over it in faint GRAY for context; a few segments are highlighted "
        f"in distinct colours: {colors}. Judge ONLY the coloured segments.\n\n"
        "RULE 1 (apply first, highest priority): Each coloured line attaches to the "
        "gray wall network at one end. If the coloured line simply continues on "
        "from that gray wall, first check whether the wall's drawn style "
        "(thickness, fill/hatching, double-line pattern) changes along the coloured "
        "part. If the style does NOT change, it is the same wall continuing: KEEP "
        "it (do not flag it) and do not analyse it further. Only when the style "
        "clearly changes do you go on to RULE 2.\n\n"
        "RULE 2: For each coloured line, look at what it traces in Image 1 and "
        "decide if it is a wall. It is a WALL if the coloured line runs along a "
        "wall — or along a door or window that sits in the line of the wall (a "
        "solid partition between spaces) — even a short piece, and even if it just "
        "stops. It is NOT a wall if it instead traces furniture, a fixture "
        "(toilet/sink/stove/bathtub), a door swing (opening direction) line, a "
        "stair tread, an appliance, text, or a dimension line.\n\n"
        "List only the colours whose line is NOT a wall.\n\n"
        "Reply with ONE JSON object, nothing else (no prose, no thinking):\n"
        '{ "not_wall": [<colour names that are not walls>] }'
    )


def build_connect_prompt(end_labels: list[int]) -> str:
    return (
        "You are auditing the wall centerline graph of a floor plan.\n\n"
        "Image 1 is the clean original floor plan (semantics). Image 2 is the "
        "extracted wall graph: blue lines are wall axes, GREEN numbered dots are "
        "ENDPOINTS (a wall that stops in mid-air), gray numbered dots are nearby "
        "nodes you may use as connection targets.\n\n"
        f"The endpoints to judge are: {end_labels}.\n\n"
        "For each green endpoint that should connect to something to complete a "
        "wall (the mask had a gap), report it. The target is either a node number "
        "N (connect -> node N) or a pair [A, B] (connect onto the wall segment "
        "between nodes A and B as a T-junction). Genuine free ends need no entry.\n\n"
        "Reply with ONE JSON object, nothing else (no prose, no thinking):\n"
        '{ "connect": [ {"from": <endpoint>, "to": <node number or [A,B]>}, ... ] }'
    )


def project_to_polyline(pt: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Closest point on polyline `poly` to `pt`. Returns (P, seg_index, dist)."""
    best = (poly[0].astype(float), 0, float("inf"))
    for i in range(len(poly) - 1):
        a = poly[i].astype(float)
        b = poly[i + 1].astype(float)
        ab = b - a
        denom = float(ab @ ab)
        t = 0.0 if denom < 1e-9 else float(np.clip((pt - a) @ ab / denom, 0.0, 1.0))
        proj = a + t * ab
        d = float(np.hypot(*(pt - proj)))
        if d < best[2]:
            best = (proj, i, d)
    return best


def apply_edits(graph: Any, label_map: dict[int, Any], result: dict[str, Any]) -> dict[str, Any]:
    added: list[np.ndarray] = []
    removed_ends: list[np.ndarray] = []

    def coord(node: Any) -> np.ndarray:
        return np.asarray(graph.nodes[node]["o"], dtype=float)

    # 1. drop non-wall endpoint stubs
    for label in result.get("not_wall", []):
        node = label_map.get(int(label))
        if node is None or node not in graph or graph.degree(node) != 1:
            continue
        removed_ends.append(coord(node))
        for u, v, k in list(graph.edges(node, keys=True)):
            graph.remove_edge(u, v, k)
        if node in graph and graph.degree(node) == 0:
            graph.remove_node(node)

    # 2. connect endpoints to a node or onto a wall segment (T-junction)
    next_id = (max((n for n in graph.nodes() if isinstance(n, (int, np.integer))), default=0) + 1)
    for item in result.get("connect", []):
        src = label_map.get(int(item.get("from")))
        if src is None or src not in graph:
            continue
        tgt = item.get("to")
        if isinstance(tgt, list) and len(tgt) == 2:        # connect onto segment A-B
            a, b = label_map.get(int(tgt[0])), label_map.get(int(tgt[1]))
            if a is None or b is None or a not in graph or b not in graph:
                continue
            ed = graph.get_edge_data(a, b)
            if not ed:  # A,B not directly joined by one edge -> fall back to nearer node
                tgt = a if np.hypot(*(coord(src) - coord(a))) <= np.hypot(*(coord(src) - coord(b))) else b
            else:
                key = min(ed, key=lambda k: project_to_polyline(coord(src), np.asarray(ed[k]["pts"]))[2])
                poly = np.asarray(ed[key]["pts"]).astype(float)
                if np.hypot(*(poly[0] - coord(a))) > np.hypot(*(poly[-1] - coord(a))):
                    poly = poly[::-1]  # orient A..B
                P, i, _ = project_to_polyline(coord(src), poly)
                pid = next_id; next_id += 1
                graph.add_node(pid, o=P)
                left = np.vstack([poly[: i + 1], P])
                right = np.vstack([P, poly[i + 1:]])
                graph.remove_edge(a, b, key)
                graph.add_edge(a, pid, pts=left, weight=path_length(left))
                graph.add_edge(pid, b, pts=right, weight=path_length(right))
                seg = np.vstack([coord(src), P])
                graph.add_edge(src, pid, pts=seg, weight=path_length(seg))
                added.append(seg)
                continue
        if isinstance(tgt, (int, float)) or (isinstance(tgt, list) and len(tgt) == 1):
            tnode = label_map.get(int(tgt[0] if isinstance(tgt, list) else tgt))
            if tnode is None or tnode not in graph:
                continue
            seg = np.vstack([coord(src), coord(tnode)])
            graph.add_edge(src, tnode, pts=seg, weight=path_length(seg))
            added.append(seg)
    return {"added": added, "removed_ends": removed_ends}


def draw_graph(ax: Any, base: np.ndarray, graph: Any, title: str,
               added: list[np.ndarray] | None = None,
               removed_ends: list[np.ndarray] | None = None) -> None:
    ax.set_title(title)
    ax.imshow(base)
    for _, _, data in graph.edges(data=True):
        pts = np.asarray(data["pts"])
        if len(pts) >= 2:
            ax.plot(pts[:, 1], pts[:, 0], color="#1f77b4", linewidth=1.3)
    for node in graph.nodes():
        y, x = graph.nodes[node]["o"]
        deg = graph.degree(node)
        color = "#2ca02c" if deg == 1 else ("#d62728" if deg >= 3 else "#ff7f0e")
        ax.plot(x, y, "o", color=color, markersize=4)
    for seg in (added or []):
        ax.plot(seg[:, 1], seg[:, 0], color="#e6194b", linewidth=2.2)
    for c in (removed_ends or []):
        ax.plot(c[1], c[0], "x", color="#7f7f7f", markersize=9, markeredgewidth=2)
    ax.axis("off")


def process_mask(mask_svg: Path, args: argparse.Namespace, client: VlmClient) -> dict[str, Any]:
    p = prepare_graph(mask_svg, args)
    sub_dir, dist, original = p["sub_dir"], p["dist"], p["original"]
    graph = p["graph"]

    global_t = global_thickness(graph, dist)
    prune_spurs(graph, dist, args.k, global_t)
    dissolve_degree2(graph, dist)

    ends, _ = endpoints_and_junctions(graph)
    if original is not None:
        base = (original.astype(float) * 0.5 + 255 * 0.5).astype(np.uint8)
    else:
        base = np.full((*p["connected"].shape, 3), 255, dtype=np.uint8)
    client.max_tokens = max(client.max_tokens, args.max_tokens)  # reasoning models burn the budget on thinking

    if args.task == "not_wall":
        return _process_notwall_color(mask_svg, args, client, p, graph, base, ends, global_t)

    radius = args.review_radius if args.review_radius is not None else args.review_radius_mult * global_t
    label_map = build_label_map(graph, ends, radius, include_targets=True)
    end_set = set(ends)
    end_labels = [lbl for lbl, node in label_map.items() if node in end_set]

    numbered_png = sub_dir / f"review_numbered_{args.task}.png"
    render_numbered(numbered_png, base, graph, label_map, end_set)

    if not ends:
        print(f"[{sub_dir.name}] no endpoints; skipping VLM.")
        result: dict[str, Any] = {"not_wall": [], "connect": []}
        reply = "(skipped: no endpoints)"
    else:
        prompt = build_connect_prompt(end_labels)
        images = [png_data_url(p["source_png"]), png_data_url(numbered_png)] if p["source_png"] else [png_data_url(numbered_png)]
        try:
            reply = client.chat(prompt, images, dump_request=args.dump_request)
            result = extract_json_object(reply)
        except (RuntimeError, ValueError) as error:
            reply = f"(error: {error})"
            result = {"not_wall": [], "connect": []}
            print(f"[{sub_dir.name}] VLM error, kept all: {error}")
    (sub_dir / f"review_vlm_{args.task}.json").write_text(
        json.dumps({"task": args.task, "prompt_endpoints": end_labels, "reply": reply,
                    "parsed": result if ends else {}},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    nodes_before, edges_before = graph.number_of_nodes(), graph.number_of_edges()
    edits = apply_edits(graph, label_map, result)
    nodes_after, edges_after = graph.number_of_nodes(), graph.number_of_edges()
    ends_after = sum(1 for n in graph.nodes() if graph.degree(n) == 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    n_nw = len(result.get("not_wall", []))
    n_con = len(result.get("connect", []))
    fig.suptitle(
        f"{sub_dir.name}  endpoints={len(ends)}  VLM: not_wall {n_nw}, connect {n_con}  "
        f"-> endpoints {len(ends)}->{ends_after}",
        fontsize=12,
    )
    draw_graph(axes[0], base, prepare_after_4b(mask_svg, args, global_t), "before (after 4a+4b)")
    draw_graph(axes[1], base, graph, "after VLM review (red=added, x=removed)",
               edits["added"], edits["removed_ends"])
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"review_debug_{args.task}_{args.tag}.png" if args.tag else f"review_debug_{args.task}.png"
    out_png = sub_dir / name
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    return {
        "image": str(out_png),
        "endpoints": len(ends),
        "endpoints_after": ends_after,
        "not_wall": n_nw,
        "connect": n_con,
        "nodes": (nodes_before, nodes_after),
        "edges": (edges_before, edges_after),
    }


def _process_notwall_color(mask_svg: Path, args: argparse.Namespace, client: VlmClient,
                           p: dict[str, Any], graph: Any, base: np.ndarray,
                           ends: list[Any], global_t: float) -> dict[str, Any]:
    """not_wall via v3-style colour batches: <=N coloured segments per image,
    asked over several calls, VLM replies which colours are not walls."""
    sub_dir = p["sub_dir"]
    palette = parse_colors(args.colors)
    bsize = max(1, len(palette))
    orig_url = png_data_url(p["source_png"]) if p["source_png"] else None
    nodes_before, edges_before = graph.number_of_nodes(), graph.number_of_edges()

    # spread each batch across the whole plan: sort by position, then stride, so
    # a batch's colours land in different rooms (easier to tell apart) instead of
    # 8 tiny stubs piled in one corner.
    ends_sorted = sorted(ends, key=lambda n: (graph.nodes[n]["o"][0], graph.nodes[n]["o"][1]))
    nbatches = max(1, -(-len(ends) // bsize))
    batches = [ends_sorted[b::nbatches] for b in range(nbatches)] if ends else []
    removed_nodes: list[Any] = []
    records: list[dict[str, Any]] = []
    for bi, chunk in enumerate(batches):
        batch = [(palette[j][0], palette[j][1], node) for j, node in enumerate(chunk)]
        png = sub_dir / f"review_numbered_not_wall_b{bi}.png"
        render_color_batch(png, base, graph, batch, args.line_width, args.line_alpha)
        names = [c[0] for c in batch]
        images = [orig_url, png_data_url(png)] if orig_url else [png_data_url(png)]
        try:
            reply = client.chat(build_notwall_color_prompt(names), images, dump_request=args.dump_request)
            res = extract_json_object(reply)
        except (RuntimeError, ValueError) as error:
            reply, res = f"(error: {error})", {"not_wall": []}
            print(f"[{sub_dir.name}] batch {bi} VLM error, kept all: {error}")
        flagged = {str(c).strip().lower() for c in res.get("not_wall", [])}
        for cname, _, node in batch:
            if cname.lower() in flagged:
                removed_nodes.append(node)
        records.append({"batch": bi, "colors": names, "reply": reply, "parsed": res})

    (sub_dir / "review_vlm_not_wall.json").write_text(
        json.dumps({"task": "not_wall", "batch_size": bsize, "batches": records},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    removed_ends: list[np.ndarray] = []
    for node in removed_nodes:
        if node in graph and graph.degree(node) == 1:
            removed_ends.append(np.asarray(graph.nodes[node]["o"], dtype=float))
            for u, v, k in list(graph.edges(node, keys=True)):
                graph.remove_edge(u, v, k)
            if node in graph and graph.degree(node) == 0:
                graph.remove_node(node)
    ends_after = sum(1 for n in graph.nodes() if graph.degree(n) == 1)

    # persist the cleaned graph so step 4c straightens the not_wall-cleaned result
    (sub_dir / "review_graph_not_wall.pkl").write_bytes(pickle.dumps(graph))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        f"{sub_dir.name}  endpoints={len(ends)}  not_wall {len(removed_ends)} "
        f"({len(batches)} batches x{bsize})  -> endpoints {len(ends)}->{ends_after}",
        fontsize=12,
    )
    draw_graph(axes[0], base, prepare_after_4b(mask_svg, args, global_t), "before (after 4a+4b)")
    draw_graph(axes[1], base, graph, "after VLM not_wall (x=removed)", None, removed_ends)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    name = f"review_debug_not_wall_{args.tag}.png" if args.tag else "review_debug_not_wall.png"
    out_png = sub_dir / name
    fig.savefig(out_png, dpi=110)
    plt.close(fig)

    return {
        "image": str(out_png),
        "endpoints": len(ends),
        "endpoints_after": ends_after,
        "not_wall": len(removed_ends),
        "connect": 0,
        "nodes": (nodes_before, graph.number_of_nodes()),
        "edges": (edges_before, graph.number_of_edges()),
    }


def prepare_after_4b(mask_svg: Path, args: argparse.Namespace, global_t: float) -> Any:
    """Rebuild the 4a+4b graph for the faithful 'before' panel."""
    p = prepare_graph(mask_svg, args)
    g = p["graph"]
    prune_spurs(g, p["dist"], args.k, global_t)
    dissolve_degree2(g, p["dist"])
    return g


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", type=Path, nargs="+", help="wall_mask_vlm_v3 output folder(s).")
    add_pipeline_args(parser)
    parser.add_argument("--task", choices=["not_wall", "connect"], default="not_wall",
                        help="not_wall: judge which endpoint stubs are non-wall; connect: join gaps.")
    parser.add_argument("--k", type=float, default=1.5, help="4a spur length threshold = k * wall thickness.")
    parser.add_argument("--review-radius", type=float, default=None, help="absolute candidate-target radius (px).")
    parser.add_argument("--review-radius-mult", type=float, default=12.0, help="candidate-target radius = mult * wall thickness.")
    parser.add_argument("--colors", type=str, default=DEFAULT_COLORS,
                        help="not_wall palette as name:hex,name:hex,... (batch size = number of colours).")
    parser.add_argument("--line-width", type=float, default=2.6, help="not_wall coloured line width (px).")
    parser.add_argument("--line-alpha", type=float, default=0.55, help="not_wall coloured line opacity (0-1).")
    parser.add_argument("--max-tokens", type=int, default=16000, help="VLM token budget (reasoning thinking can eat the JSON).")
    parser.add_argument("--reasoning-effort", type=str, default="high", help="override env reasoning effort (minimal/low/medium/high).")
    parser.add_argument("--env-file", type=Path, default=None, help="env file whose EDITOR_LLM_* vars to use (e.g. ../../.envOpenAI.local).")
    parser.add_argument("--tag", type=str, default="", help="suffix for the output filename.")
    parser.add_argument("--dump-request", action="store_true")
    args = parser.parse_args()

    masks: list[Path] = []
    for target in args.target:
        if target.exists():
            masks.extend(find_masks(target))
    if not masks:
        print("No wall_mask.svg found.")
        return 2

    if args.env_file:
        os.environ.update(parse_env_file(args.env_file))  # take precedence over .env.local
    client = VlmClient()
    effort = normalize_reasoning_effort(args.reasoning_effort)
    if effort:
        client.reasoning_effort = effort
    for mask_svg in masks:
        info = process_mask(mask_svg, args, client)
        print(
            f"[{mask_svg.parent.name}] endpoints {info['endpoints']}->{info['endpoints_after']} "
            f"(not_wall {info['not_wall']}, connect {info['connect']}) "
            f"nodes {info['nodes'][0]}->{info['nodes'][1]} edges {info['edges'][0]}->{info['edges'][1]} "
            f"-> {info['image']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
