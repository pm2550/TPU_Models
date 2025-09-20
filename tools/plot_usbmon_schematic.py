#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a simple three-lane schematic (CPU / USB / TPU) that illustrates
H2D -> Compute -> D2H serialization using given example numbers. No external
data or heavy plotting libraries required; outputs a self-contained SVG.

Default values (can be overridden by flags) are taken from the user's example:
  - out_span_ms_mean (H2D) = 1.3661 ms
  - in_span_ms_mean  (D2H) = 3.3283 ms
  - compute_ms (actual inference on TPU) = 5.747 ms
  - pure_span_ms_mean = 7.63534 ms (used only to compute host_outside_ms)

Host outside time = max(0, pure_span_ms_mean - compute_ms), split evenly into
CPU pre and CPU post segments (before/after the invoke window).

Usage:
  python tools/plot_usbmon_schematic.py \
      --h2d-ms 1.3661 --d2h-ms 3.3283 --compute-ms 5.747 --pure-ms 7.63534 \
      --h2d-mibps 336.59 --d2h-mibps 62.78 \
      --outfile results/plots/usbmon_schematic.svg

All flags are optional; sensible defaults are provided.
"""
import argparse
from pathlib import Path


def svg_rect(x, y, w, h, fill, stroke="#000", sw=1, rx=3, ry=3, opacity=1.0):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx}" ry="{ry}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{sw}" />')


def svg_text(x, y, s, size=12, anchor='middle', weight='normal', family='DejaVu Sans, Arial'):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}">{s}</text>')

def svg_text_len(x, y, s, length_px, size=12, anchor='start', weight='normal', adjust='spacing'):
    """Emit text that is forced to occupy exactly length_px using SVG textLength.
    This makes the visible end of the text predictable so gaps can be uniform.
    """
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="DejaVu Sans, Arial" font-size="{size}" font-weight="{weight}" '
            f'lengthAdjust="{adjust}" textLength="{length_px:.2f}">{s}</text>')


def main():
    ap = argparse.ArgumentParser(description='Create a USB/TPU H2D->Compute->D2H schematic SVG')
    ap.add_argument('--h2d-ms', type=float, default=1.3661, help='H2D duration in ms (out_span)')
    ap.add_argument('--d2h-ms', type=float, default=3.3283, help='D2H duration in ms (in_span)')
    ap.add_argument('--compute-ms', type=float, default=5.747, help='TPU compute duration in ms')
    ap.add_argument('--pure-ms', type=float, default=7.63534, help='Pure span mean ms (for host-outside calc)')
    ap.add_argument('--h2d-mibps', type=float, default=336.5903, help='Mean H2D speed in MiB/s')
    ap.add_argument('--d2h-mibps', type=float, default=62.7806, help='Mean D2H speed in MiB/s')
    ap.add_argument('--outfile', type=str, default='results/plots/usbmon_schematic.svg', help='Output SVG path')
    args = ap.parse_args()

    # Derived times
    host_outside = max(0.0, args.pure_ms - args.compute_ms)
    # Split host time with a 1:4 ratio (pre:post)
    host_pre = host_outside * (1.0 / 5.0)
    host_post = host_outside * (4.0 / 5.0)

    # Layout constants
    pad_l = 80.0
    pad_r = 40.0
    pad_t = 28.0
    lane_h = 24.0
    lane_gap = 22.0
    lane_round = 4
    # Scale: px per ms
    scale = 80.0

    # Timeline positions (sequential serialization)
    t0 = 0.0
    t_cpu_pre_s = t0
    t_h2d_s = t_cpu_pre_s + host_pre
    t_comp_s = t_h2d_s + args.h2d_ms
    t_d2h_s = t_comp_s + args.compute_ms
    t_cpu_post_s = t_d2h_s + args.d2h_ms
    t_end = t_cpu_post_s + host_post

    total_w = pad_l + scale * (t_end - t0) + pad_r
    # Lanes (top->bottom): CPU, USB, TPU
    y_cpu = pad_t
    y_usb = y_cpu + lane_h + lane_gap
    y_tpu = y_usb + lane_h + lane_gap
    h = y_tpu + lane_h + pad_t

    # Colors
    # Use a distinct color for Host to avoid clashing with Compute's gray.
    col_cpu = '#9467bd'  # purple
    col_h2d = '#1f77b4'  # blue
    col_d2h = '#d62728'  # red
    col_comp = '#7f7f7f'  # gray

    # Build SVG content
    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{h:.0f}">' )
    # Arrowhead definition for baseline time arrows
    out.append(
        '<defs>'
        '<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,7 L9,3.5 z" fill="#4d4d4d" />'
        '</marker>'
        # Heavier, high-contrast hatches for B/W prints
        # Host: horizontal stripes
        '<pattern id="hostHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 3 L 6 3" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        # H2D: 45° diagonal stripes
        '<pattern id="h2dHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        # D2H: -45° diagonal stripes (rotate 135°)
        '<pattern id="d2hHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(135)">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        # Compute: cross-hatch grid
        '<pattern id="computeHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.65" />'
        '<path d="M 0 0 L 6 0" stroke="#000000" stroke-width="1.2" opacity="0.65" />'
        '</pattern>'
        '</defs>'
    )

    # Top legend (no numeric data, no title)
    legend_y = 18
    items = [
        (col_cpu, 'Host (pre/post)', 'hostHatch'),
        (col_h2d, 'H2D', 'h2dHatch'),
        (col_comp, 'Compute', None),  # compute uses pure color (no hatch in legend)
        (col_d2h, 'D2H', 'd2hHatch'),
    ]
    # Compute legend layout with constant inter-item gap so that
    # the distance from text end to the next card is identical.
    # Larger legend blocks to make patterns clearer
    sw_w = 22.0     # legend block width
    sw_h = 14.0     # legend block height
    label_gap = 6.0 # gap between square and its text
    # Use a monospace font for legend labels to avoid kerning variance
    mono_family = 'DejaVu Sans Mono, Menlo, Consolas, monospace'
    mono_char = 7.2  # approx px per character at size=12 for monospace
    gap_between = 10.0  # tightened, constant gap between items (text end -> next square)

    # approximate item widths: square + label_gap + text_width
    widths = [sw_w + label_gap + (len(name) * mono_char) for _c, name, _pid in items]
    legend_w = sum(widths) + (len(items) - 1) * gap_between
    lx = (total_w - legend_w) / 2.0  # center the legend row
    x_cursor = lx
    for (c, name, pid), w in zip(items, widths):
        rect_y = legend_y - sw_h + 2
        out.append(svg_rect(x_cursor, rect_y, sw_w, sw_h, c, sw=0.8, rx=2, ry=2, opacity=0.95))
        # overlay hatch to match lane block pattern (skip when pid is None)
        if pid:
            out.append(svg_rect(x_cursor, rect_y, sw_w, sw_h, f'url(#{pid})', stroke="none", sw=0, rx=2, ry=2, opacity=1.0))
        out.append(svg_text(x_cursor + sw_w + label_gap, legend_y, name, anchor='start', size=12, family=mono_family))
        x_cursor += w + gap_between

    # Lane labels
    out.append(svg_text(pad_l/2, y_cpu + lane_h*0.65, 'CPU', anchor='middle', size=12, weight='bold'))
    out.append(svg_text(pad_l/2, y_usb + lane_h*0.65, 'USB', anchor='middle', size=12, weight='bold'))
    out.append(svg_text(pad_l/2, y_tpu + lane_h*0.65, 'TPU', anchor='middle', size=12, weight='bold'))

    # Segments
    def x_of(t_ms: float) -> float:
        return pad_l + scale * (t_ms - t0)

    # CPU pre/post (outside invoke window)
    if host_pre > 1e-6:
        hx = x_of(t_cpu_pre_s)
        hw = scale*host_pre
        out.append(svg_rect(hx, y_cpu, hw, lane_h, col_cpu, rx=lane_round, ry=lane_round, opacity=0.65))
        out.append(svg_rect(hx, y_cpu, hw, lane_h, 'url(#hostHatch)', stroke="none", sw=0, rx=lane_round, ry=lane_round, opacity=1.0))
    if host_post > 1e-6:
        hx2 = x_of(t_cpu_post_s)
        hw2 = scale*host_post
        out.append(svg_rect(hx2, y_cpu, hw2, lane_h, col_cpu, rx=lane_round, ry=lane_round, opacity=0.65))
        out.append(svg_rect(hx2, y_cpu, hw2, lane_h, 'url(#hostHatch)', stroke="none", sw=0, rx=lane_round, ry=lane_round, opacity=1.0))

    # H2D burst
    out.append(svg_rect(x_of(t_h2d_s), y_usb, scale*args.h2d_ms, lane_h, col_h2d, rx=lane_round, ry=lane_round, opacity=0.95))
    out.append(svg_rect(x_of(t_h2d_s), y_usb, scale*args.h2d_ms, lane_h, 'url(#h2dHatch)', stroke="none", sw=0, rx=lane_round, ry=lane_round, opacity=1.0))

    # Compute window (gap between H2D and D2H)
    comp_x = x_of(t_comp_s)
    comp_w = scale*args.compute_ms
    comp_y = y_tpu
    comp_h = lane_h
    # Compute: pure color fill (no hatch) for contrast against patterned IO/Host
    out.append(svg_rect(comp_x, comp_y, comp_w, comp_h, col_comp, rx=lane_round, ry=lane_round, opacity=0.65))

    # D2H burst
    out.append(svg_rect(x_of(t_d2h_s), y_usb, scale*args.d2h_ms, lane_h, col_d2h, rx=lane_round, ry=lane_round, opacity=0.95))
    out.append(svg_rect(x_of(t_d2h_s), y_usb, scale*args.d2h_ms, lane_h, 'url(#d2hHatch)', stroke="none", sw=0, rx=lane_round, ry=lane_round, opacity=1.0))

    # Baseline time arrow: move down to the bottom of the compute (TPU) block
    base_x1 = pad_l
    extra_tail_px = 20.0  # extend a bit past the rightmost block
    base_x2 = pad_l + scale * (t_end - t0) + extra_tail_px
    stroke = '#4d4d4d'
    sw = 1.3
    y_base = y_tpu + lane_h + 2.0  # slightly below compute block bottom edge
    out.append(
        f'<line x1="{base_x1:.2f}" y1="{y_base:.2f}" x2="{base_x2:.2f}" y2="{y_base:.2f}" '
        f'stroke="{stroke}" stroke-width="{sw}" marker-end="url(#arrowhead)" />'
    )

    # Footer notes
    # No bottom annotation per request

    out.append('</svg>')

    # Write
    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print(str(out_path))


if __name__ == '__main__':
    main()
