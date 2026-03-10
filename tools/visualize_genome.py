"""Visualize a saved NEAT genome as DOT and PNG.

Usage:
    python tools/visualize_genome.py runs/run_xxx/best_genome.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class Palette:
    input_node: str = "#2563EB"
    output_node: str = "#EA580C"
    hidden_node: str = "#0D9488"
    node_border: str = "#0F172A"
    edge_positive: str = "#16A34A"
    edge_negative: str = "#DC2626"
    edge_disabled: str = "#A3A3A3"
    figure_background: str = "#F8FAFC"
    canvas_background: str = "#FFFFFF"
    text_title: str = "#0F172A"
    text_body: str = "#334155"
    text_muted: str = "#64748B"
    edge_label_bg: str = "#FFFFFFE8"


@dataclass(frozen=True)
class VisualStyle:
    palette: Palette = Palette()
    font_family: str = "DejaVu Sans"
    title_size: int = 18
    subtitle_size: int = 10
    node_label_size: int = 8
    section_label_size: int = 11
    edge_label_size: int = 7
    legend_size: int = 9
    node_radius: float = 0.032
    max_edge_width: float = 4.8
    min_edge_width: float = 0.7
    figure_size: tuple[float, float] = (14.0, 8.8)


STYLE = VisualStyle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genome", type=Path, help="Path to best_genome.json (or raw genome JSON)")
    parser.add_argument(
        "--dot-out",
        type=Path,
        default=Path("best_genome.dot"),
        help="Output DOT path",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=Path("best_genome.png"),
        help="Output PNG path",
    )
    return parser.parse_args()


def load_genome_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "best_genome" in data and isinstance(data["best_genome"], dict):
            best_genome = data["best_genome"]
            if "genome" in best_genome and isinstance(best_genome["genome"], dict):
                return best_genome["genome"]
        if "genome" in data and isinstance(data["genome"], dict):
            return data["genome"]
    return data


def node_label(node: dict[str, Any], index: int) -> str:
    node_type = str(node.get("type", "hidden"))
    node_id = node.get("id", index)
    if node_type == "input":
        return f"input_{node_id}"
    if node_type == "output":
        return f"output_{node_id}"
    return f"hidden_{node_id}"


def node_color(node_type: str) -> str:
    if node_type == "input":
        return STYLE.palette.input_node
    if node_type == "output":
        return STYLE.palette.output_node
    return STYLE.palette.hidden_node


def dot_from_genome(genome: dict[str, Any]) -> tuple[str, int, int]:
    nodes = genome.get("node_genes", [])
    connections = genome.get("connection_genes", [])

    header_lines = [
        "digraph Genome {",
        "  rankdir=LR;",
        '  graph [bgcolor="white", pad="0.55", nodesep="0.55", ranksep="0.95", splines="curved"];',
        f'  node [shape=circle, style="filled,setlinewidth(1.6)", fontname="{STYLE.font_family}", fontsize=10, width=1.0, fixedsize=true, color="{STYLE.palette.node_border}"];',
        f'  edge [fontname="{STYLE.font_family}", fontsize=8, arrowsize=0.7, penwidth=1.0];',
    ]
    node_lines: list[str] = []
    edge_lines: list[str] = []

    for index, node in enumerate(nodes):
        label = node_label(node, index)
        node_type = str(node.get("type", "hidden"))
        color = node_color(node_type)
        node_lines.append(f'  n{node.get("id", index)} [label="{label}", fillcolor="{color}"];')

    for connection in connections:
        source = connection["in_node"]
        target = connection["out_node"]
        weight = float(connection.get("weight", 0.0))
        enabled = bool(connection.get("enabled", True))
        color = STYLE.palette.edge_positive if weight >= 0 else STYLE.palette.edge_negative
        style = "solid" if enabled else "dashed"
        if not enabled:
            color = STYLE.palette.edge_disabled
        penwidth = max(STYLE.min_edge_width, min(STYLE.max_edge_width, abs(weight) * 1.7))
        edge_lines.append(
            f'  n{source} -> n{target} [label="{weight:+.2f}", color="{color}", fontcolor="{color}", style="{style}", penwidth={penwidth:.2f}, alpha=0.86];'
        )

    dot_lines = [*header_lines, *node_lines, *edge_lines, "}"]
    return "\n".join(dot_lines) + "\n", len(node_lines), len(edge_lines)


def render_png_with_graphviz(dot_path: Path, png_path: Path) -> bool:
    dot_bin = shutil.which("dot")
    if dot_bin is None:
        return False
    subprocess.run([dot_bin, "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    return True


def render_png_with_matplotlib(genome: dict[str, Any], png_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    nodes = genome["node_genes"]
    connections = genome["connection_genes"]

    input_nodes = [n for n in nodes if n.get("type") == "input"]
    hidden_nodes = [n for n in nodes if n.get("type") == "hidden"]
    output_nodes = [n for n in nodes if n.get("type") == "output"]

    positions: dict[Any, tuple[float, float]] = {}

    def place_group(group: list[dict[str, Any]], x: float) -> None:
        count = max(1, len(group))
        for idx, node in enumerate(sorted(group, key=lambda n: n.get("id", 0))):
            y = 1.0 - (idx + 1) / (count + 1)
            positions[node.get("id")] = (x, y)

    place_group(input_nodes, 0.08)
    place_group(hidden_nodes, 0.50)
    place_group(output_nodes, 0.92)

    missing = [node for node in nodes if node.get("id") not in positions]
    for idx, node in enumerate(missing):
        positions[node.get("id")] = (0.50, 0.1 + 0.8 * (idx / max(1, len(missing))))

    fig, ax = plt.subplots(figsize=STYLE.figure_size, facecolor=STYLE.palette.figure_background)
    ax.set_facecolor(STYLE.palette.canvas_background)

    # subtle layer guides for groups
    lane_width = 0.2
    for center_x, title in ((0.08, "Inputs"), (0.50, "Hidden"), (0.92, "Outputs")):
        left = max(0.0, center_x - lane_width / 2)
        right = min(1.0, center_x + lane_width / 2)
        ax.axvspan(left, right, color="#94A3B8", alpha=0.06, zorder=0)
        ax.text(
            center_x,
            0.965,
            title,
            fontsize=STYLE.section_label_size,
            color=STYLE.palette.text_body,
            fontweight="bold",
            ha="center",
            va="center",
            family=STYLE.font_family,
        )

    for conn in connections:
        source = conn["in_node"]
        target = conn["out_node"]
        if source not in positions or target not in positions:
            continue
        (x1, y1) = positions[source]
        (x2, y2) = positions[target]
        weight = float(conn.get("weight", 0.0))
        enabled = bool(conn.get("enabled", True))
        color = STYLE.palette.edge_positive if weight >= 0 else STYLE.palette.edge_negative
        linestyle = "solid" if enabled else (0, (4, 3))
        if not enabled:
            color = STYLE.palette.edge_disabled
        linewidth = max(STYLE.min_edge_width, min(STYLE.max_edge_width, abs(weight) * 1.7))

        curvature = 0.08 if (y2 - y1) >= 0 else -0.08
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={curvature:.3f}",
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            alpha=0.88 if enabled else 0.62,
            zorder=1,
        )
        ax.add_patch(arrow)
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0 + (0.016 if curvature > 0 else -0.016)
        ax.text(
            mx,
            my,
            f"{weight:+.2f}",
            fontsize=STYLE.edge_label_size,
            color=color,
            ha="center",
            va="center",
            family=STYLE.font_family,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=STYLE.palette.edge_label_bg, edgecolor="none"),
        )

    for index, node in enumerate(nodes):
        node_id = node.get("id", index)
        x, y = positions[node_id]
        node_type = str(node.get("type", "hidden"))
        color = node_color(node_type)

        circle = plt.Circle((x, y), STYLE.node_radius, facecolor=color, edgecolor=STYLE.palette.node_border, linewidth=1.6, zorder=3)
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            node_label(node, index),
            ha="center",
            va="center",
            fontsize=STYLE.node_label_size,
            color="white",
            zorder=4,
            fontweight="bold",
            family=STYLE.font_family,
        )

    enabled_edges = sum(1 for c in connections if bool(c.get("enabled", True)))
    ax.set_title(
        "NEAT Genome Topology",
        fontsize=STYLE.title_size,
        fontweight="bold",
        color=STYLE.palette.text_title,
        pad=14,
        family=STYLE.font_family,
    )
    ax.text(
        0.5,
        1.01,
        f"{len(nodes)} nodes • {len(connections)} connections ({enabled_edges} enabled)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=STYLE.subtitle_size,
        color=STYLE.palette.text_muted,
        family=STYLE.font_family,
    )

    ax.text(0.01, 0.02, "solid = enabled  ·  dashed = disabled", transform=ax.transAxes, fontsize=STYLE.legend_size, color=STYLE.palette.text_muted, ha="left", va="bottom", family=STYLE.font_family)
    ax.text(0.99, 0.02, "green = positive weight  ·  red = negative weight", transform=ax.transAxes, fontsize=STYLE.legend_size, color=STYLE.palette.text_muted, ha="right", va="bottom", family=STYLE.font_family)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    margin = max(0.02, min(0.06, 0.75 / math.sqrt(max(1, len(nodes)))))
    ax.margins(x=margin, y=margin)
    plt.tight_layout(pad=1.8)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    genome = load_genome_payload(args.genome)
    assert len(genome["node_genes"]) > 0

    dot_text, node_line_count, edge_line_count = dot_from_genome(genome)
    args.dot_out.parent.mkdir(parents=True, exist_ok=True)
    args.dot_out.write_text(dot_text, encoding="utf-8")
    print(f"DOT node lines emitted: {node_line_count}")
    print(f"DOT edge lines emitted: {edge_line_count}")
    print(f"DOT char length: {len(dot_text)}")

    png_written = False
    try:
        png_written = render_png_with_graphviz(args.dot_out, args.png_out)
    except subprocess.CalledProcessError:
        png_written = False

    if not png_written:
        try:
            render_png_with_matplotlib(genome, args.png_out)
            png_written = True
        except ModuleNotFoundError as exc:
            missing = str(exc).split("'")[1] if "'" in str(exc) else "dependency"
            print(f"Unable to render PNG: missing optional dependency '{missing}'.")

    print(f"Wrote DOT graph: {args.dot_out}")
    if png_written:
        print(f"Wrote PNG graph: {args.png_out}")
    else:
        print("PNG graph was not generated (install graphviz `dot` or matplotlib).")


if __name__ == "__main__":
    main()
