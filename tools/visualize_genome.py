"""Visualize a saved NEAT genome as DOT and PNG.

Usage:
    python tools/visualize_genome.py runs/run_xxx/best_genome.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


INPUT_COLOR = "#4C78A8"
OUTPUT_COLOR = "#F58518"
HIDDEN_COLOR = "#54A24B"
DISABLED_EDGE_COLOR = "#BBBBBB"
POS_EDGE_COLOR = "#2CA02C"
NEG_EDGE_COLOR = "#D62728"


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


def dot_from_genome(genome: dict[str, Any]) -> tuple[str, int, int]:
    nodes = genome["node_genes"]
    connections = genome["connection_genes"]

    header_lines = [
        "digraph Genome {",
        "  rankdir=LR;",
        '  graph [bgcolor="white"];',
        '  node [shape=circle, style=filled, fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]
    node_lines: list[str] = []
    edge_lines: list[str] = []

    for index, node in enumerate(nodes):
        label = node_label(node, index)
        node_type = str(node.get("type", "hidden"))
        color = HIDDEN_COLOR
        if node_type == "input":
            color = INPUT_COLOR
        elif node_type == "output":
            color = OUTPUT_COLOR
        node_lines.append(f'  n{node.get("id", index)} [label="{label}", fillcolor="{color}"];')

    for connection in connections:
        source = connection["in_node"]
        target = connection["out_node"]
        weight = float(connection.get("weight", 0.0))
        enabled = bool(connection.get("enabled", True))
        color = POS_EDGE_COLOR if weight >= 0 else NEG_EDGE_COLOR
        style = "solid" if enabled else "dashed"
        if not enabled:
            color = DISABLED_EDGE_COLOR
        penwidth = max(0.5, min(6.0, abs(weight) * 2.0))
        edge_lines.append(
            f'  n{source} -> n{target} [label="{weight:+.2f}", color="{color}", style="{style}", penwidth={penwidth:.2f}];'
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

    fig, ax = plt.subplots(figsize=(12, 7))

    for conn in connections:
        source = conn["in_node"]
        target = conn["out_node"]
        if source not in positions or target not in positions:
            continue
        (x1, y1) = positions[source]
        (x2, y2) = positions[target]
        weight = float(conn.get("weight", 0.0))
        enabled = bool(conn.get("enabled", True))
        color = POS_EDGE_COLOR if weight >= 0 else NEG_EDGE_COLOR
        linestyle = "-" if enabled else "--"
        if not enabled:
            color = DISABLED_EDGE_COLOR
        linewidth = max(0.5, min(6.0, abs(weight) * 2.0))

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=linewidth, linestyle=linestyle, alpha=0.85),
            zorder=1,
        )
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(mx, my, f"{weight:+.2f}", fontsize=8, color=color, ha="center", va="center")

    for index, node in enumerate(nodes):
        node_id = node.get("id", index)
        x, y = positions[node_id]
        node_type = str(node.get("type", "hidden"))
        color = HIDDEN_COLOR
        if node_type == "input":
            color = INPUT_COLOR
        elif node_type == "output":
            color = OUTPUT_COLOR

        circle = plt.Circle((x, y), 0.03, facecolor=color, edgecolor="#222222", linewidth=1.2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node_label(node, index), ha="center", va="center", fontsize=8, color="white", zorder=4)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    plt.tight_layout()
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
