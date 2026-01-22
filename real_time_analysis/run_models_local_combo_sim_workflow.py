#!/usr/bin/env python3
"""High-level workflow for run_models_local_combo_sim.py (simplified view).

Run:
  python run_models_local_combo_sim_workflow.py
Outputs:
  run_models_local_combo_sim_workflow.png
"""

from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
import networkx as nx


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 15,
        "axes.edgecolor": "#2c2c2c",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

TEXT_FONTSIZE = 16


def add_box(ax, center, text, width=0.8, height=0.4, fc="#f8f8f8", ec="#555"):
    x, y = center
    # Measure text to auto-adjust dimensions
    t = ax.text(x, y, text, ha="center", va="center", fontsize=TEXT_FONTSIZE, zorder=3)
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = t.get_window_extent(renderer=renderer)
    text_width = bbox.width / fig.dpi  # convert pixels to inches
    text_height = bbox.height / fig.dpi
    
    # Ensure box is at least as large as text + generous padding
    padding_x = 0.0
    padding_y = 0.0
    actual_width = max(width, text_width + padding_x)
    actual_height = max(height, text_height + padding_y)
    
    rect = FancyBboxPatch(
        (x - actual_width / 2, y - actual_height / 2),
        actual_width,
        actual_height,
        boxstyle="round,pad=0.08",
        linewidth=1,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(rect)
    # Update node dimensions for arrow calculation
    return actual_width, actual_height


def add_diamond(ax, center, text, width=0.8, height=0.4, fc="#eef2ff", ec="#4a56c0"):
    x, y = center
    verts = [
        (x, y + height / 2),
        (x + width / 2, y),
        (x, y - height / 2),
        (x - width / 2, y),
    ]
    poly = Polygon(verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2)
    ax.add_patch(poly)
    ax.text(x, y, text, ha="center", va="center", fontsize=TEXT_FONTSIZE)
    return width, height


def edge_points(n1: Dict, n2: Dict) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    import math

    def rect_intersection(cx, cy, half_w, half_h, dx, dy):
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return cx, cy
        # Avoid division by zero by using inf when half width/height is zero
        scale_x = abs(dx) / half_w if half_w > 1e-6 else float('inf')
        scale_y = abs(dy) / half_h if half_h > 1e-6 else float('inf')
        scale = max(scale_x, scale_y)
        if scale == 0 or math.isinf(scale):
            return cx, cy
        return cx + dx / scale, cy + dy / scale

    (x1, y1), (x2, y2) = n1["pos"], n2["pos"]
    w1, h1 = n1["w"], n1["h"]
    w2, h2 = n2["w"], n2["h"]
    dx, dy = x2 - x1, y2 - y1
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return (x1, y1), (x2, y2)

    ux, uy = dx / dist, dy / dist

    pad1 = min(w1, h1) * 0.08
    pad2 = min(w2, h2) * 0.08
    start_inner = rect_intersection(x1, y1, w1 / 2 + pad1, h1 / 2 + pad1, dx, dy)
    end_inner = rect_intersection(x2, y2, w2 / 2 + pad2, h2 / 2 + pad2, -dx, -dy)

    span_dx = end_inner[0] - start_inner[0]
    span_dy = end_inner[1] - start_inner[1]
    span_len = math.hypot(span_dx, span_dy)
    if span_len < 1e-6:
        return start_inner, end_inner

    shrink_each = max(0.04, span_len * 0.08)
    shrink_cap = max(span_len / 2 - 1e-4, 0.0)
    shrink_each = min(shrink_each, shrink_cap)
    start = (start_inner[0] + ux * shrink_each, start_inner[1] + uy * shrink_each)
    end = (end_inner[0] - ux * shrink_each, end_inner[1] - uy * shrink_each)
    return start, end


def add_arrow(ax, n1: Dict, n2: Dict, label: str = "", style: str = "solid"):
    start, end = edge_points(n1, n2)
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.4,
        color="#1f1f1f",
        linestyle=style,
        zorder=4,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.2, label, fontsize=8, ha="center", va="center", color="#333")


def auto_layout(nodes_def, edges):
    """Prepare node metadata and topological layer ordering."""
    G = nx.DiGraph()
    for node_name in nodes_def.keys():
        G.add_node(node_name)
    for src, dst, *_ in edges:
        G.add_edge(src, dst)

    layers: List[List[str]] = list(nx.topological_generations(G))

    nodes = {}
    for node_name, props in nodes_def.items():
        nodes[node_name] = {
            **props,
            "pos": (0.0, 0.0),  # will be set after measuring actual sizes
            "w": props.get("w", 0.9),
            "h": props.get("h", 0.4),
        }

    return nodes, layers


def apply_layout(
    nodes: Dict[str, Dict],
    layers: List[List[str]],
    h_spacing: float = 0.8,
    v_spacing: float = 0.1,
    orientation: str = "horizontal",
    width_scale: float = 3.0,
    row_scale: float = 1.0,
    node_offsets: Optional[Dict[str, Tuple[float, float]]] = None,
):
    """Assign horizontal positions based on measured widths/heights."""
    column_centers: List[float] = []
    prev_half_width = 0.0
    for idx, layer in enumerate(layers):
        max_width = max(nodes[name]["w"] for name in layer)
        half_width = max_width / 2
        if idx == 0:
            column_centers.append(0.0)
        else:
            column_centers.append(column_centers[-1] + prev_half_width + h_spacing + half_width)
        prev_half_width = half_width

    for layer_idx, layer in enumerate(layers):
        heights = [nodes[name]["h"] for name in layer]
        total_height = sum(heights) + v_spacing * max(len(layer) - 1, 0)
        start_y = -total_height / 2
        current_y = start_y
        for name, height in zip(layer, heights):
            y_center = current_y + height / 2
            nodes[name]["pos"] = (column_centers[layer_idx], y_center)
            current_y += height + v_spacing

    if orientation == "vertical":
        for info in nodes.values():
            x, y = info["pos"]
            info["pos"] = (y * width_scale, -x * row_scale)

    if node_offsets:
        for name, (dx, dy) in node_offsets.items():
            if name in nodes:
                x, y = nodes[name]["pos"]
                nodes[name]["pos"] = (x + dx, y + dy)

    row_scale=row_scale,

def compute_bounds(nodes: Dict[str, Dict], margin: float = 0.4) -> Tuple[float, float, float, float]:
    min_x = min(node["pos"][0] - node["w"]/2 - margin for node in nodes.values())
    max_x = max(node["pos"][0] + node["w"]/2 + margin for node in nodes.values())
    min_y = min(node["pos"][1] - node["h"]/2 - margin for node in nodes.values())
    max_y = max(node["pos"][1] + node["h"]/2 + margin for node in nodes.values())
    return min_x, max_x, min_y, max_y


def build_diagram():
    # Use muted, academic palette for clarity.
    common_fc, common_ec = "#f5f4f0", "#2f2f2f"
    upper_fc, upper_ec = "#e5ede7", "#355f54"   # Ce/transfer path
    lower_fc, lower_ec = "#efe3d6", "#8a4c28"   # measured invoke path

    # Define nodes (only content, no positions)
    nodes_def = {
        "prep": {"text": "Prep models & combos", "fc": common_fc, "ec": common_ec},
        "run_combo": {"text": "Run combo models\n(K = 2..8)", "fc": lower_fc, "ec": lower_ec},
        "extract_invoke": {"text": "Extract measured\ninvoke time", "fc": lower_fc, "ec": lower_ec},
        "segment_models": {"text": "Run finest-segment\nmodels", "fc": upper_fc, "ec": upper_ec},
        "collect_usb": {"text": "Collect USB data\nvia usbmon", "fc": upper_fc, "ec": upper_ec},
        "record_invoke_meta": {"text": "Capture invoke duration\nand timestamps", "fc": upper_fc, "ec": upper_ec},
        "normalize": {"text": "Normalize timelines\nalign segments/windows", "fc": upper_fc, "ec": upper_ec},
        "compute_stats": {"text": "Compute parameters\nand estimate bounds", "fc": upper_fc, "ec": upper_ec},
        "compare": {"text": "Compare expected vs measured", "fc": common_fc, "ec": common_ec},
    }
    
    # Define edges (graph structure)
    edges = [
        ("prep", "run_combo", ""),
        ("prep", "segment_models", ""),
        ("run_combo", "extract_invoke", ""),
        ("segment_models", "collect_usb", ""),
        ("segment_models", "record_invoke_meta", ""),
        ("collect_usb", "normalize", ""),
        ("record_invoke_meta", "normalize", ""),
        ("normalize", "compute_stats", ""),
        ("compute_stats", "compare", ""),
        ("extract_invoke", "compare", ""),
    ]

    # Auto-calculate initial metadata and layer order
    nodes, _ = auto_layout(nodes_def, edges)

    layers = [
        ["prep"],
        ["run_combo", "segment_models"],
        ["extract_invoke", "record_invoke_meta", "collect_usb"],
        ["compare", "compute_stats", "normalize"],
    ]

    margin = 0.5
    h_spacing = 0.0025
    v_spacing = 1.2
    scale_factor = 1.5
    max_iters = 4
    size_eps = 1e-3

    fig = ax = None
    min_x = max_x = min_y = max_y = 0.0
    orientation = "vertical"
    width_scale = 1.65
    row_scale = 0.5
    node_offsets = {
        "compare": (-1.3, 0.0),
        "compute_stats": (0.0, 0.0),
        "normalize": (1.3, 0.0),
    }

    for attempt in range(max_iters):
        apply_layout(
            nodes,
            layers,
            h_spacing=h_spacing,
            v_spacing=v_spacing,
            orientation=orientation,
            width_scale=width_scale,
            row_scale=row_scale,
            node_offsets=node_offsets,
        )
        min_x, max_x, min_y, max_y = compute_bounds(nodes, margin)
        content_width = max_x - min_x
        content_height = max_y - min_y
        fig_width = max(content_width * scale_factor, 4.0)
        fig_height = max(content_height * scale_factor, 3.0)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("#fdfcf7")
        ax.set_facecolor("#fdfcf7")
        ax.axis("off")

        size_changed = False
        for name, info in nodes.items():
            draw_fn = add_diamond if info.get("shape") == "diamond" else add_box
            actual_w, actual_h = draw_fn(
                ax,
                info["pos"],
                info["text"],
                width=info["w"],
                height=info["h"],
                fc=info.get("fc", "#f8f8f8"),
                ec=info.get("ec", "#555"),
            )
            if abs(actual_w - info["w"]) > size_eps or abs(actual_h - info["h"]) > size_eps:
                nodes[name]["w"] = actual_w
                nodes[name]["h"] = actual_h
                size_changed = True

        if not size_changed:
            break

        plt.close(fig)
        fig = ax = None
    else:
        # Final render if we hit iteration cap
        apply_layout(
            nodes,
            layers,
            h_spacing=h_spacing,
            v_spacing=v_spacing,
            orientation=orientation,
            width_scale=width_scale,
            row_scale=row_scale,
            node_offsets=node_offsets,
        )
        min_x, max_x, min_y, max_y = compute_bounds(nodes, margin)
        content_width = max_x - min_x
        content_height = max_y - min_y
        fig_width = max(content_width * scale_factor, 4.0)
        fig_height = max(content_height * scale_factor, 3.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.patch.set_facecolor("#fdfcf7")
        ax.set_facecolor("#fdfcf7")
        ax.axis("off")
        for name, info in nodes.items():
            draw_fn = add_diamond if info.get("shape") == "diamond" else add_box
            draw_fn(
                ax,
                info["pos"],
                info["text"],
                width=info["w"],
                height=info["h"],
                fc=info.get("fc", "#f8f8f8"),
                ec=info.get("ec", "#555"),
            )
    
    # Draw arrows
    for edge in edges:
        src, dst, *rest = edge
        label = rest[0] if rest else ""
        style = rest[1] if len(rest) > 1 else "solid"
        add_arrow(ax, nodes[src], nodes[dst], label=label, style=style)
    
    # Set axis limits based on final bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    return fig


if __name__ == "__main__":
    fig = build_diagram()
    fig.savefig("run_models_local_combo_sim_workflow.png", dpi=220, bbox_inches="tight")
    print("Generated run_models_local_combo_sim_workflow.png")
