#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produce a vertical stacked bar chart (SVG) for segments 1..4 with three
components per bar: H2D (out_span_ms), Compute (pure_gap_ms), D2H (in_span_ms).

Y-axis: milliseconds. X-axis: seg1..seg4. Each component uses distinct
color and hatch (Compute defaults to pure color for contrast). Numeric values
are annotated next to each stacked block.

Defaults are from user-provided numbers:
  seg1: invoke=6.998, in=0.318, out=3.740, pure_gap=2.537
  seg2: invoke=4.504, in=0.128, out=3.776, pure_gap=0.366
  seg3: invoke=4.383, in=0.067, out=3.831, pure_gap=0.318
  seg4: invoke=4.885, in=0.060, out=4.619, pure_gap=0.066

Usage:
  python tools/plot_seg_bars_svg.py [--outfile results/plots/seg_bars.svg]
"""
from pathlib import Path
import argparse
from decimal import Decimal, ROUND_HALF_UP


def fmt_ms(v: float) -> str:
    try:
        q = Decimal(str(v)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return f"{q:.2f}"
    except Exception:
        return f"{v:.2f}"


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

def svg_label_value_line(x, y, label, value, size=12, gap_px=12.0, family='DejaVu Sans, Arial'):
    # Render label and value in one line with a fixed pixel gap between them
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="start" '
        f'font-family="{family}" font-size="{size}">'
        f'<tspan>{label}</tspan><tspan dx="{gap_px}">{value}</tspan>'
        f'</text>'
    )


def main():
    ap = argparse.ArgumentParser(description='Draw stacked bars for seg1..4 (SVG).')
    ap.add_argument('--outfile', type=str, default='results/plots/seg_bars.svg')
    ap.add_argument('--height', type=float, default=440.0, help='Figure height in px')
    ap.add_argument('--include-full', action='store_true', help='Also draw the full aggregate bar to the right of seg4')
    # Font sizes (px)
    ap.add_argument('--tick-font-size', type=float, default=18.0, help='Y-axis tick label size')
    ap.add_argument('--seg-label-font-size', type=float, default=16.0, help='Per-segment label text size (name/value)')
    ap.add_argument('--x-label-font-size', type=float, default=18.0, help='X-axis (seg names) label size')
    ap.add_argument('--legend-font-size', type=float, default=19.5, help='Legend text size')
    ap.add_argument('--legend-swatch-height', type=float, default=26.0, help='Legend swatch height (px)')
    ap.add_argument('--legend-swatch-width', type=float, default=28.0, help='Legend swatch width (px)')
    ap.add_argument('--min-segment-ms', type=float, default=0.10, help='Minimum visible height (ms) for any non-zero segment')
    ap.add_argument('--y-unit', type=str, default='ms', help='Y-axis unit text (e.g., ms)')
    ap.add_argument('--y-unit-dy', type=float, default=8.0, help='Additional downward offset for Y-axis unit label (px)')
    ap.add_argument('--y-min-max', type=float, default=None, help='Minimum Y-axis maximum (e.g., 7 to cap ticks at 0..7)')
    ap.add_argument('--y-headroom', type=float, default=0.2, help='Extra headroom above Y max (axis units)')
    # Data overrides
    ap.add_argument('--d2h-override', type=float, default=None, help='If set, use this D2H value for all segs (ms)')
    # Spacing controls
    ap.add_argument('--label-min-sep', type=float, default=32.0, help='Minimum vertical separation between labels in one bar (px)')
    ap.add_argument('--legend-gap-between', type=float, default=24.0, help='Horizontal gap between legend cards (px)')
    ap.add_argument('--legend-y', type=float, default=26.0, help='Legend baseline Y position (px); auto-clamped to avoid clipping')
    ap.add_argument('--legend-top-pad', type=float, default=0.0, help='Minimum top padding between legend swatch and SVG top (px)')
    ap.add_argument('--bar-gap', type=float, default=120.0, help='Horizontal gap between bars (px)')
    args = ap.parse_args()

    # Data
    input_by_seg = {
        'seg1': 0.414,
        'seg2': 0.057,
        'seg3': 0.041,
        'seg4': 0.040,
    }
    segs = [
        dict(name='seg1', invoke=6.998, d2h=0.318, h2d=3.740, comp=2.522, input=input_by_seg['seg1']),
        dict(name='seg2', invoke=4.504, d2h=0.128, h2d=3.776, comp=0.351, input=input_by_seg['seg2']),
        dict(name='seg3', invoke=4.383, d2h=0.067, h2d=3.831, comp=0.303, input=input_by_seg['seg3']),
        dict(name='seg4', invoke=4.885, d2h=0.060, h2d=4.619, comp=0.051, input=input_by_seg['seg4']),
    ]
    if args.include_full:
        segs.append(dict(name='full', invoke=17.590, d2h=0.060, h2d=14.161, comp=3.120))
    # Optional override: set all D2H values to a constant
    if args.d2h_override is not None:
        for s in segs:
            s['d2h'] = float(args.d2h_override)
    # Derive host(pre/post) time = invoke - (h2d + comp + d2h)
    for s in segs:
        s['host'] = max(0.0, s['invoke'] - (s['h2d'] + s['comp'] + s['d2h']))

    # Colors and patterns (match previous schematic):
    col_h2d = '#1f77b4'   # blue + +45° hatch (Weight streaming)
    col_input = '#9ecae1' # light blue for Input streaming
    col_comp = '#7f7f7f'  # gray (pure)
    col_d2h = '#d62728'   # red + -45° hatch
    col_host = '#9467bd'  # purple + horizontal hatch

    # Layout
    # Increase top padding to move the bar chart downward on the canvas
    pad_l, pad_r, pad_t, pad_b = 90.0, 64.0, 60.0, 48.0
    axis_gap = 28.0  # gap between Y-axis and first bar
    bar_w = 46.0
    gap = float(args.bar_gap)      # spacing between bars
    n = len(segs)
    # Compute total width (bars-driven) and figure height
    total_w = pad_l + axis_gap + n * bar_w + (n - 1) * gap + pad_r
    total_h = float(args.height)

    # Scale: y px per ms
    y_max = max(s['h2d'] + s['comp'] + s['d2h'] + s['host'] for s in segs)
    # Apply user-provided minimum Y-axis maximum if set; else keep >=5
    if args.y_min_max is not None:
        try:
            y_min_max = float(args.y_min_max)
        except Exception:
            y_min_max = None
        if y_min_max is not None:
            y_max = max(y_max, y_min_max)
    else:
        y_max = max(y_max, 5.0)
    # Add headroom so bars don't touch the top (keeps last tick if < next integer)
    try:
        y_max += max(0.0, float(args.y_headroom))
    except Exception:
        pass
    y_scale = (total_h - pad_t - pad_b) / y_max
    # Ensure canvas is wide enough to fully contain the legend cards + margins
    # and the right-side labels of the last bar (avoid clipping). Legend will be
    # rendered on two rows: top three short, bottom two long. For width
    # estimation, take the max row width.
    legend_font = float(args.legend_font_size)
    legend_label_gap = 7.0 * (legend_font / 12.0)
    legend_gap_between = float(args.legend_gap_between)
    legend_swatch_w = float(args.legend_swatch_width)
    char_px = 7.8 * (legend_font / 12.0)
    legend_row1_names = ['Compute', 'D2H', 'Host(pre/post)']
    legend_row2_names = ['Input streaming (IS)', 'Weight streaming (WS)']
    row1_w = sum(legend_swatch_w + legend_label_gap + len(name)*char_px for name in legend_row1_names)              + (len(legend_row1_names)-1) * legend_gap_between
    row2_w = sum(legend_swatch_w + legend_label_gap + len(name)*char_px for name in legend_row2_names)              + (len(legend_row2_names)-1) * legend_gap_between
    legend_w_est = max(row1_w, row2_w)
    min_canvas_w = legend_w_est + 40.0  # ~20px margins on both sides
    # Also reserve vertical space so the two-row legend never overlaps bars.
    # If not enough top padding, push bars down and proportionally increase figure height.
    legend_top_pad = max(0.0, float(args.legend_top_pad))
    sw_h = float(args.legend_swatch_height)
    # Baseline Y for top row; clamp to avoid clipping at the very top
    legend_y_base = max(legend_top_pad + sw_h, float(args.legend_y))
    row_gap = max(8.0, sw_h * 0.25)
    y_row2 = legend_y_base + sw_h + row_gap
    rect_y2 = y_row2 - sw_h + legend_top_pad
    legend_block_bottom = max(rect_y2 + sw_h, y_row2)
    need_pad_top = legend_block_bottom + 28.0  # keep a larger gap below legend
    if pad_t < need_pad_top:
        extra = need_pad_top - pad_t
        pad_t += extra
        total_h += extra
        y_scale = (total_h - pad_t - pad_b) / y_max

    # Also account for the rightmost data labels next to the last bar
    n = len(segs)
    if n > 0:
        last_bx = pad_l + axis_gap + (n - 1) * (bar_w + gap)
        label_x = last_bx + bar_w + 8.0
        lbl_font = float(args.seg_label_font_size)
        lbl_char_px = 7.8 * (lbl_font / 12.0)
        # Estimate the widest combined line: "label + gap + number" among all labels/segments
        def comb_len(name, val):
            return len(name) + 1 + len(val)
        max_comb = 0
        for s in segs:
            in_ms = float(s.get('input', 0.0))
            weight_ms = max(0.0, float(s['h2d']) - in_ms)
            pairs = [
                ('WS', fmt_ms(weight_ms)),
                ('IS', fmt_ms(in_ms)),
                ('Compute', fmt_ms(s['comp'])),
                ('D2H', fmt_ms(s['d2h'])),
                ('Host', fmt_ms(s['host'])),
            ]
            for nm, vv in pairs:
                max_comb = max(max_comb, comb_len(nm, vv))
        # Leave a bit of right margin
        right_labels_w = max_comb * lbl_char_px + 16.0  # +margin
        min_canvas_w = max(min_canvas_w, label_x + right_labels_w)
    if total_w < min_canvas_w:
        total_w = min_canvas_w

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
        # Input streaming hatch (vertical)
        '<pattern id="inputHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 3 0 L 3 6" stroke="#000000" stroke-width="1.1" opacity="0.6" />'
        '</pattern>'
        '</defs>'
    )

    # Axes
    x0 = pad_l
    y0 = total_h - pad_b
    # Extend axes a bit beyond plotting area and add arrowheads
    y_top = max(6.0, pad_t - 6.0)
    x_end = total_w - pad_r + 14.0
    y_axis_top = max(6.0, pad_t-14.0)
    out.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x0:.2f}" y2="{y_axis_top:.2f}" stroke="#444" stroke-width="1.2" marker-end="url(#axisArrow)" />')
    out.append(f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{(total_w - pad_r + 28.0):.2f}" y2="{y0:.2f}" stroke="#444" stroke-width="1.2" marker-end="url(#axisArrow)" />')
    # Y-axis unit label (e.g., ms); allow a small downward offset for readability
    y_unit_y = max(12.0, pad_t - 22.0 + float(args.y_unit_dy))
    out.append(svg_text(x0 - 10, y_unit_y, args.y_unit, anchor='end', size=args.tick_font_size))

    # Y ticks
    ticks = []
    # use 1 ms step for denser ticks
    step = 1.0
    t = 0.0
    while t <= y_max + 1e-6:
        ticks.append(t)
        t += step
    for v in ticks:
        yy = y0 - v * y_scale
        out.append(svg_line(x0 - 4, yy, x0, yy, stroke="#444", sw=1.0))
        out.append(svg_text(x0 - 8, yy + 5, f"{v:.0f}", size=args.tick_font_size, anchor='end'))

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
        # Stack order: Weight (bottom) -> Input -> Compute -> D2H -> Host
        zero_slot_px = 14.0  # virtual slot height used to position labels when segment value == 0
        in_ms = float(s.get('input', 0.0))
        weight_ms = max(0.0, float(s['h2d']) - in_ms)
        # Apply minimum visible size for non-zero segments
        min_ms = max(0.0, float(args.min_segment_ms))
        disp_weight_ms = weight_ms if weight_ms == 0 else max(weight_ms, min_ms)
        h = disp_weight_ms * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_h2d, opacity=0.95))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#h2dHatch)', stroke="none", sw=0, opacity=1.0))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("WS", fmt_ms(weight_ms), pref_y))
        by -= h
        # Input streaming
        disp_in_ms = in_ms if in_ms == 0 else max(in_ms, min_ms)
        h = disp_in_ms * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_input, opacity=0.95))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#inputHatch)', stroke="none", sw=0, opacity=1.0))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("IS", fmt_ms(in_ms), pref_y))
        by -= h
        # Compute
        comp_ms = float(s['comp'])
        disp_comp_ms = comp_ms if comp_ms == 0 else max(comp_ms, min_ms)
        h = disp_comp_ms * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_comp, opacity=0.65))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("Compute", fmt_ms(s['comp']), pref_y))
        by -= h
        # D2H
        d2h_ms = float(s['d2h'])
        disp_d2h_ms = d2h_ms if d2h_ms == 0 else max(d2h_ms, min_ms)
        h = disp_d2h_ms * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_d2h, opacity=0.95))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#d2hHatch)', stroke="none", sw=0, opacity=1.0))
        # Prefer a virtual slot centered within D2H (or a small slot if zero)
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("D2H", fmt_ms(s['d2h']), pref_y))
        by -= h
        # Host (pre/post combined as one segment)
        host_ms = float(s['host'])
        disp_host_ms = host_ms if host_ms == 0 else max(host_ms, min_ms)
        h = disp_host_ms * y_scale
        if h > 0:
            out.append(svg_rect(bx, by - h, bar_w, h, col_host, opacity=0.75))
            out.append(svg_rect(bx, by - h, bar_w, h, 'url(#hostHatch)', stroke="none", sw=0, opacity=1.0))
        pref_y = by - ((h/2) if h > 0 else (zero_slot_px/2))
        labels.append(("Host", fmt_ms(s['host']), pref_y))
        by -= h

        # Now place labels to avoid overlap, allowing some space below the axis
        min_y_allowed = pad_t + 10.0
        max_y_allowed = y0 + 28.0  # can go a bit more below baseline
        min_sep = float(args.label_min_sep)
        # Render labels in fixed stack order bottom->top while avoiding overlap
        # Stack order used: H2D (bottom), Compute, D2H, Host (top)
        labels_stack = labels  # [H2D, Compute, D2H, Host] as appended above
        preferred = [p for (_t, _v, p) in labels_stack]
        # Allocate positions top->bottom so ordering is preserved
        order_top_to_bottom = sorted(range(len(preferred)), key=lambda i: preferred[i])
        preferred_sorted = [preferred[i] for i in order_top_to_bottom]
        assigned_sorted = allocate_positions(preferred_sorted, min_y_allowed, max_y_allowed, min_sep)
        # Map assigned positions back to original indices
        assigned_by_index = [0.0] * len(preferred)
        for i, y_pos in zip(order_top_to_bottom, assigned_sorted):
            assigned_by_index[i] = y_pos
        # Draw labels in bottom->top stack order, with fixed pixel gap between label and value
        for idx, (seg_t, seg_v, _p) in enumerate(labels_stack):
            ly = assigned_by_index[idx]
            base_x = bx + bar_w + 8
            y_text = ly - 5.0
            out.append(svg_label_value_line(base_x, y_text, seg_t, seg_v, size=args.seg_label_font_size, gap_px=12.0))

        # X label
        out.append(svg_text(bx + bar_w/2, y0 + 22, s['name'].replace('seg','seg '), size=args.x_label_font_size))
    # Legend (top, two rows) — fixed safe position to avoid clipping when bars move
    legend_top_pad = max(0.0, float(args.legend_top_pad))
    legend_y = max(legend_top_pad + args.legend_swatch_height, float(args.legend_y))
    mono_family = 'DejaVu Sans Mono, Menlo, Consolas, monospace'
    # Fixed legend metrics (explicit sizes)
    sw_w, sw_h = float(args.legend_swatch_width), float(args.legend_swatch_height)
    label_font = float(args.legend_font_size)
    label_gap = 7.0 * (label_font / 12.0)
    gap_between = float(args.legend_gap_between)
    # Two rows: top three short, bottom two long. Show abbreviations in legend text.
    row1 = [
        (col_comp, 'Compute', None),
        (col_d2h, 'D2H', 'd2hHatch'),
        (col_host, 'Host(pre/post)', 'hostHatch'),
    ]
    row2 = [
        (col_input, 'Input streaming (IS)', 'inputHatch'),
        (col_h2d, 'Weight streaming (WS)', 'h2dHatch'),
    ]
    char_px = 7.8 * (label_font / 12.0)
    widths1 = [sw_w + label_gap + len(name) * char_px for _c, name, _p in row1]
    widths2 = [sw_w + label_gap + len(name) * char_px for _c, name, _p in row2]
    legend_w1 = sum(widths1) + (len(row1)-1)*gap_between
    legend_w2 = sum(widths2) + (len(row2)-1)*gap_between
    # Center each row separately
    lx1 = (total_w - legend_w1)/2.0
    lx2 = (total_w - legend_w2)/2.0
    # Draw row1 at legend_y, row2 below with a small gap
    rect_y1 = legend_y - sw_h + legend_top_pad
    if rect_y1 < 0:
        dy = -rect_y1
        legend_y += dy
        rect_y1 = legend_y - sw_h + legend_top_pad
    row_gap = max(8.0, sw_h * 0.25)
    y_row1 = legend_y
    y_row2 = legend_y + sw_h + row_gap
    rect_y2 = y_row2 - sw_h + legend_top_pad
    # Render row1
    xc = lx1
    for (c, name, pid), w in zip(row1, widths1):
        out.append(svg_rect(xc, rect_y1, sw_w, sw_h, c, sw=0.8, rx=2, ry=2, opacity=0.95))
        if pid:
            out.append(svg_rect(xc, rect_y1, sw_w, sw_h, f'url(#{pid})', stroke="none", sw=0, rx=2, ry=2, opacity=1.0))
        out.append(svg_text(xc + sw_w + label_gap, y_row1, name, anchor='start', size=label_font, family=mono_family))
        xc += w + gap_between
    # Render row2
    xc = lx2
    for (c, name, pid), w in zip(row2, widths2):
        out.append(svg_rect(xc, rect_y2, sw_w, sw_h, c, sw=0.8, rx=2, ry=2, opacity=0.95))
        if pid:
            out.append(svg_rect(xc, rect_y2, sw_w, sw_h, f'url(#{pid})', stroke="none", sw=0, rx=2, ry=2, opacity=1.0))
        out.append(svg_text(xc + sw_w + label_gap, y_row2, name, anchor='start', size=label_font, family=mono_family))
        xc += w + gap_between

    out.append('</svg>')

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print(str(out_path))


if __name__ == '__main__':
    main()
