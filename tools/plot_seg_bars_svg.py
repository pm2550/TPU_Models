#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a vertical stacked bar chart (SVG) for segments 1..4 with three
components per bar: H2D (out_span_ms), Compute (pure_gap_ms), D2H (in_span_ms).

Y-axis: milliseconds. X-axis: seg1..seg4. Each component uses distinct
color and hatch (Compute defaults to pure color for contrast). Numeric values
are annotated next to each stacked block.

Defaults are from user-provided numbers:
  seg1: invoke=6.998, in=0.258, out=3.740, pure_gap=2.537
  seg2: invoke=4.504, in=0.068, out=3.776, pure_gap=0.366
  seg3: invoke=4.383, in=0.007, out=3.831, pure_gap=0.318
  seg4: invoke=4.885, in=0.000, out=4.619, pure_gap=0.066

Usage:
  python tools/plot_seg_bars_svg.py [--outfile results/plots/seg_bars.svg]
"""
from pathlib import Path
import argparse


def fmt_ms(v: float) -> str:
    return f"{v:.3f}"


def svg_rect(x, y, w, h, fill, stroke="#000", sw=1, rx=2, ry=2, opacity=1.0):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx}" ry="{ry}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{sw}" />')


def svg_line(x1, y1, x2, y2, stroke="#333", sw=1.0):
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw}" />')


def svg_text(x, y, s, size=12, anchor='middle', weight='normal', family='DejaVu Sans, Arial'):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}">{s}</text>')


def main():
    ap = argparse.ArgumentParser(description='Draw stacked bars for seg1..4 (SVG).')
    ap.add_argument('--outfile', type=str, default='results/plots/seg_bars.svg')
    ap.add_argument('--height', type=float, default=360.0, help='Figure height in px (default: 360)')
    ap.add_argument('--include-full', action='store_true', help='Also draw the full aggregate bar to the right of seg4')
    args = ap.parse_args()

    # Data
    segs = [
        dict(name='seg1', invoke=6.998, d2h=0.258, h2d=3.740, comp=2.537),
        dict(name='seg2', invoke=4.504, d2h=0.068, h2d=3.776, comp=0.366),
        dict(name='seg3', invoke=4.383, d2h=0.007, h2d=3.831, comp=0.318),
        dict(name='seg4', invoke=4.885, d2h=0.000, h2d=4.619, comp=0.066),
    ]
    if args.include_full:
        segs.append(dict(name='full', invoke=17.590, d2h=0.000, h2d=14.161, comp=3.120))
    # Derive host(pre/post) time = invoke - (h2d + comp + d2h)
    for s in segs:
        s['host'] = max(0.0, s['invoke'] - (s['h2d'] + s['comp'] + s['d2h']))

    # Colors and patterns (match previous schematic):
    col_h2d = '#1f77b4'   # blue + +45° hatch
    col_comp = '#7f7f7f'  # gray (pure)
    col_d2h = '#d62728'   # red + -45° hatch
    col_host = '#9467bd'  # purple + horizontal hatch

    # Layout
    # Increase top padding to move the bar chart downward on the canvas
    pad_l, pad_r, pad_t, pad_b = 80.0, 56.0, 56.0, 44.0
    axis_gap = 26.0  # gap between Y-axis and first bar (further away)
    bar_w = 44.0
    gap = 56.0      # even larger spacing between bars
    n = len(segs)
    # Compute total width
    total_w = pad_l + axis_gap + n * bar_w + (n - 1) * gap + pad_r
    total_h = float(args.height)

    # Scale: y px per ms
    y_max = max(s['h2d'] + s['comp'] + s['d2h'] + s['host'] for s in segs)
    y_max = max(y_max, 5.0)
    y_scale = (total_h - pad_t - pad_b) / y_max

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{total_h:.0f}">')
    # Patterns for hatches
    out.append(
        '<defs>'
        # Arrowhead for axes
        '<marker id="axisArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,7 L9,3.5 z" fill="#444" />'
        '</marker>'
        # H2D hatch (+45°)
        '<pattern id="h2dHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        # D2H hatch (-45°)
        '<pattern id="d2hHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(135)">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        # Host hatch (horizontal)
        '<pattern id="hostHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 3 L 6 3" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        '</defs>'
    )

    # Axes
    x0 = pad_l
    y0 = total_h - pad_b
    # Extend axes a bit beyond plotting area and add arrowheads
    y_top = max(6.0, pad_t - 6.0)
    x_end = total_w - pad_r + 14.0
    out.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{max(6.0, pad_t-14.0):.2f}" stroke="#444" stroke-width="1.2" marker-end="url(#axisArrow)" />')
    out.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{(total_w - pad_r + 28.0):.2f}" y2="{y0:.2f}" stroke="#444" stroke-width="1.2" marker-end="url(#axisArrow)" />')

    # Y ticks
    ticks = []
    # choose step ~1ms or 2ms depending on max
    step = 1.0 if y_max <= 6 else 2.0
    t = 0.0
    while t <= y_max + 1e-6:
        ticks.append(t)
        t += step
    for v in ticks:
        yy = y0 - v * y_scale
        out.append(svg_line(x0 - 4, yy, x0, yy, stroke="#444", sw=1.0))
        out.append(svg_text(x0 - 8, yy + 4, f"{v:.0f}", size=11, anchor='end'))

    # Helper to allocate non-overlapping label positions near each bar
    def allocate_positions(preferred_list, min_y, max_y, min_sep):
        if not preferred_list:
            return []
        pairs = sorted([(py, idx) for idx, py in enumerate(preferred_list)], key=lambda x: x[0])
        pos = []
        for k, (py, _idx) in enumerate(pairs):
            if k == 0:
                pos.append(max(min_y, py))
            else:
                pos.append(max(py, pos[-1] + min_sep))
        # If top exceeds max_y, shift all down
        overflow = pos[-1] - max_y
        if overflow > 0:
            pos = [p - overflow for p in pos]
            # If this pushes bottom below min_y, clamp to evenly spaced range
            if pos[0] < min_y:
                pos = [min_y + i * min_sep for i in range(len(pos))]
        # Map back to original order
        assigned = [0.0] * len(preferred_list)
        for (py, idx), p in zip(pairs, pos):
            assigned[idx] = p
        return assigned

    # Bars
    for i, s in enumerate(segs):
        bx = pad_l + axis_gap + i * (bar_w + gap)
        by = y0
        # Collect label preferred positions and metadata for this bar
        labels = []  # list of (segment_name, value_str, pref_y)
        # Stack order: H2D at bottom, then Compute, then D2H on top
        # H2D
        zero_slot_px = 14.0  # virtual slot height used to position labels when segment value == 0
        h = s['h2d'] * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_h2d, opacity=0.95))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#h2dHatch)', stroke="none", sw=0, opacity=1.0))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("H2D", fmt_ms(s['h2d']), pref_y))
        by -= h
        # Compute
        h = s['comp'] * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_comp, opacity=0.65))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("Compute", fmt_ms(s['comp']), pref_y))
        by -= h
        # D2H
        h = s['d2h'] * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_d2h, opacity=0.95))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#d2hHatch)', stroke="none", sw=0, opacity=1.0))
        # Prefer a virtual slot centered above Compute when zero, but never above Host center
        host_h_tmp = s['host'] * y_scale
        host_center = by - ((host_h_tmp/2) if host_h_tmp > 0 else (zero_slot_px/2))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        if pref_y < host_center + 6.0:
            pref_y = host_center + 6.0
        labels.append(("D2H", fmt_ms(s['d2h']), pref_y))
        by -= h
        # Host (pre/post combined as one segment)
        h = s['host'] * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_host, opacity=0.75))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#hostHatch)', stroke="none", sw=0, opacity=1.0))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("Host", fmt_ms(s['host']), pref_y))
        by -= h

        # Now place labels to avoid overlap, allowing some space below the axis
        min_y_allowed = pad_t + 10.0
        max_y_allowed = y0 + 28.0  # can go a bit more below baseline
        min_sep = 26.0
        assigned_y = allocate_positions([p for (_t, _v, p) in labels], min_y_allowed, max_y_allowed, min_sep)
        for (seg_t, seg_v, _p), ly in zip(labels, assigned_y):
            out.append(svg_text(bx + bar_w + 6, ly - 10.0, seg_t, anchor='start', size=10))
            out.append(svg_text(bx + bar_w + 6, ly, seg_v, anchor='start', size=11))

        # X label
        out.append(svg_text(bx + bar_w/2, y0 + 18, s['name'].replace('seg','seg '), size=12))

    # Legend (top) — fixed safe position to avoid clipping when bars move
    legend_y = 26.0
    mono_family = 'DejaVu Sans Mono, Menlo, Consolas, monospace'
    # Scale legend card size with figure height so bars higher => cards larger
    scale_leg = max(0.9, min(1.8, total_h / 300.0))
    sw_w, sw_h = 22.0 * scale_leg, 14.0 * scale_leg
    label_font = max(10.0, min(16.0, 12.0 * scale_leg))
    label_gap = 6.0 * (label_font / 12.0)
    gap_between = 12.0
    items = [
        (col_h2d, 'H2D', 'h2dHatch'),
        (col_comp, 'Compute', None),
        (col_d2h, 'D2H', 'd2hHatch'),
        (col_host, 'Host(pre/post)', 'hostHatch'),
    ]
    char_px = 7.2 * (label_font / 12.0)
    widths = [sw_w + label_gap + len(name) * char_px for _c, name, _p in items]
    legend_w = sum(widths) + (len(items)-1)*gap_between
    lx = (total_w - legend_w)/2.0
    xc = lx
    for (c, name, pid), w in zip(items, widths):
        rect_y = legend_y - sw_h + 2
        out.append(svg_rect(xc, rect_y, sw_w, sw_h, c, sw=0.8, rx=2, ry=2, opacity=0.95))
        if pid:
            out.append(svg_rect(xc, rect_y, sw_w, sw_h, f'url(#{pid})', stroke="none", sw=0, rx=2, ry=2, opacity=1.0))
        out.append(svg_text(xc + sw_w + label_gap, legend_y, name, anchor='start', size=label_font, family=mono_family))
        xc += w + gap_between

    out.append('</svg>')

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print(str(out_path))


if __name__ == '__main__':
    main()
