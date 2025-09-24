#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Horizontal stacked bar comparison for two models with H2D/Compute/D2H/Host segments.

Models (defaults from user):
1) Required streaming: invoke=17.590, in=0.000, out=14.161, gap=3.120
2) Fully on-chip:      invoke=5.114,  in=0.000, out=1.157,  gap=3.120

Host(pre/post) segment is computed as invoke - (out + gap + in). Labels show type and value,
zero values are still labeled and placed using a small virtual slot to keep layout consistent.

Output: results/plots/compare_two_models_hbar.svg
"""
from pathlib import Path
import argparse


def svg_rect(x, y, w, h, fill, stroke="#000", sw=1, rx=3, ry=3, opacity=1.0):
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx}" ry="{ry}" fill="{fill}" fill-opacity="{opacity}" '
            f'stroke="{stroke}" stroke-width="{sw}" />')


def svg_line(x1, y1, x2, y2, stroke="#333", sw=1.0, end=None):
    end_attr = f' marker-end="url(#{end})"' if end else ''
    return (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{end_attr} />')


def svg_text(x, y, s, size=12, anchor='middle', weight='normal', family='DejaVu Sans, Arial'):
    return (f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}">{s}</text>')


def fmt_ms(v: float) -> str:
    return f"{v:.3f}"


def allocate_positions(preferred, min_x, max_x, min_sep):
    """1D non-overlap placement near preferred x positions.
    Returns list of assigned x positions in the same order.
    """
    if not preferred:
        return []
    pairs = sorted([(px, i) for i, px in enumerate(preferred)], key=lambda t: t[0])
    xs = []
    for k, (px, _i) in enumerate(pairs):
        if k == 0:
            xs.append(max(min_x, px))
        else:
            xs.append(max(px, xs[-1] + min_sep))
    overflow = xs[-1] - max_x
    if overflow > 0:
        xs = [x - overflow for x in xs]
        if xs[0] < min_x:
            xs = [min_x + j * min_sep for j in range(len(xs))]
    assigned = [0.0] * len(preferred)
    for (px, i), x in zip(pairs, xs):
        assigned[i] = x
    return assigned


def main():
    ap = argparse.ArgumentParser(description='Compare 2 models: horizontal stacked bars (SVG).')
    ap.add_argument('--outfile', type=str, default='results/plots/compare_two_models_hbar.svg')
    ap.add_argument('--width', type=float, default=1000.0)
    ap.add_argument('--height', type=float, default=360.0)
    args = ap.parse_args()

    # Data
    # Compare: Complete model (current fully on-chip) vs Segmented model (sum of seg1..4)
    # Segmented sums provided by user:
    # seg1: invoke=6.998, in=0.258, out=3.740, pure=2.537, host≈0.463
    # seg2: invoke=4.504, in=0.068, out=3.776, pure=0.366, host≈0.294
    # seg3: invoke=4.383, in=0.007, out=3.831, pure=0.318, host≈0.227
    # seg4: invoke=4.885, in=0.000, out=4.619, pure=0.066, host≈0.200
    # Totals: invoke=20.770, in=0.333, out=15.966, pure=3.287, host≈1.184
    models = [
        dict(name='Complete Model',   invoke=17.590, d2h=0.000, h2d=14.161, comp=3.120),
        dict(name='Segmented Model',  invoke=20.770, d2h=0.333, h2d=15.966, comp=3.287),
    ]
    for m in models:
        m['host'] = max(0.0, m['invoke'] - (m['h2d'] + m['comp'] + m['d2h']))

    # Colors / patterns (consistent with other charts)
    col_h2d = '#1f77b4'   # blue + +45° hatch
    col_comp = '#7f7f7f'  # gray (pure)
    col_d2h = '#d62728'   # red + -45° hatch
    col_host = '#9467bd'  # purple + horizontal hatch

    # Layout
    total_w = float(args.width)
    total_h = float(args.height)
    # Increase left padding to keep long model labels (e.g., "Partial Off-Chip") within bounds
    pad_l, pad_r, pad_t, pad_b = 140.0, 64.0, 64.0, 56.0
    axis_gap = 16.0
    row_gap = 64.0
    bar_h = 26.0

    # X scale (ms -> px)
    xmax = max(m['h2d'] + m['comp'] + m['d2h'] + m['host'] for m in models)
    plot_w = total_w - pad_l - pad_r - axis_gap
    x_scale = plot_w / max(1e-9, xmax)

    # Y coordinates for bars
    y0 = pad_t + 50.0
    ys = [y0, y0 + bar_h + row_gap]

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" height="{total_h:.0f}">')
    # defs: arrowheads and hatches
    out.append(
        '<defs>'
        '<marker id="axisArrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,7 L9,3.5 z" fill="#444" />'
        '</marker>'
        '<pattern id="h2dHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        '<pattern id="d2hHatch" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(135)">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 0 L 0 6" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        '<pattern id="hostHatch" patternUnits="userSpaceOnUse" width="6" height="6">'
        '<rect width="6" height="6" fill="none" />'
        '<path d="M 0 3 L 6 3" stroke="#000000" stroke-width="1.2" opacity="0.6" />'
        '</pattern>'
        '</defs>'
    )

    # Axes with arrows
    x0 = pad_l
    base_y = ys[0] + row_gap/2  # baseline roughly centered across bars
    # Horizontal time axis: extend past the rightmost bar
    out.append(svg_line(x0, base_y, total_w - pad_r + 48.0, base_y, stroke="#444", sw=1.2, end='axisArrow'))
    # Vertical reference line (no arrow) — make it symmetric around base_y
    top_end = pad_t - 28.0
    # symmetric bottom end around base_y
    bottom_end = 2*base_y - top_end
    # clamp within canvas
    bottom_end = min(bottom_end, total_h - pad_b + 28.0)
    out.append(svg_line(x0, bottom_end, x0, top_end, stroke="#444", sw=1.2, end=None))

    # X ticks (ms)
    step = 2.0 if xmax <= 10 else 4.0
    t = 0.0
    while t <= xmax + 1e-6:
        xt = x0 + axis_gap + t * x_scale
        out.append(svg_line(xt, base_y - 4, xt, base_y + 4, stroke="#444", sw=1.0))
        out.append(svg_text(xt, base_y + 18, f"{t:.0f}", size=11, anchor='middle'))
        t += step

    # Legend cards (top)
    legend_y = 28.0
    mono_family = 'DejaVu Sans Mono, Menlo, Consolas, monospace'
    sw_w, sw_h = 24.0, 16.0
    label_gap = 7.0
    gap_between = 14.0
    items = [
        (col_h2d, 'H2D', 'h2dHatch'),
        (col_comp, 'Compute', None),
        (col_d2h, 'D2H', 'd2hHatch'),
        (col_host, 'Host(pre/post)', 'hostHatch'),
    ]
    char_px = 7.2
    widths = [sw_w + label_gap + len(name) * char_px for _c, name, _p in items]
    legend_w = sum(widths) + (len(items)-1)*gap_between
    lx = (total_w - legend_w)/2.0
    xc = lx
    for (c, name, pid), w in zip(items, widths):
        rect_y = legend_y - sw_h + 2
        out.append(svg_rect(xc, rect_y, sw_w, sw_h, c, sw=0.8, rx=2, ry=2, opacity=0.95))
        if pid:
            out.append(svg_rect(xc, rect_y, sw_w, sw_h, f'url(#{pid})', stroke="none", sw=0, rx=2, ry=2, opacity=1.0))
        out.append(svg_text(xc + sw_w + label_gap, legend_y, name, anchor='start', size=12, family=mono_family))
        xc += w + gap_between

    # Draw bars and labels (labels rendered above each bar; order consistent with legend, Host at outermost)
    zero_slot_px = 20.0  # virtual slot for zero-length segments
    # Horizontal separation between adjacent labels to avoid Host/D2H crowding
    min_sep = 44.0       # min separation between label centers within the same bar (px)
    for idx, m in enumerate(models):
        y_center = ys[idx]
        y_top = y_center - bar_h/2
        x_start = x0 + axis_gap
        segments = [
            ('H2D', col_h2d, 'h2dHatch', m['h2d']),
            ('Compute', col_comp, None, m['comp']),
            ('D2H', col_d2h, 'd2hHatch', m['d2h']),
            ('Host', col_host, 'hostHatch', m['host']),
        ]

        # Draw stacked rectangles and collect preferred label x positions
        prefs = []
        x = x_start
        for seg_name, color, pid, val in segments:
            w = val * x_scale
            if w > 0:
                out.append(svg_rect(x, y_top, w, bar_h, color, opacity=0.95 if seg_name != 'Compute' else 0.65))
                if pid:
                    out.append(svg_rect(x, y_top, w, bar_h, f'url(#{pid})', stroke='none', sw=0, rx=3, ry=3, opacity=1.0))
                cx = x + w/2
            else:
                # zero width: place a virtual slot right after x (without advancing stack)
                cx = x + zero_slot_px/2
            # Apply small bias for visual balance
            if seg_name == 'Compute':
                # default slight left; bottom row move back right (closer to its segment)
                cx -= 12.0
                if idx == 1:
                    cx += 12.0  # net ~0 shift for bottom
            elif seg_name == 'D2H':
                # default slight left; bottom row move right a bit (closer to its segment)
                cx -= 14.0
                if idx == 1:
                    cx += 10.0  # net ~-4 px
            elif seg_name == 'Host':
                cx -= 10.0
            prefs.append((seg_name, val, cx))
            x += w

        # Bias Host label towards the outermost right edge (end of invoke),
        # so it will not appear inside D2H when Host is small. Allow a small overflow.
        min_x = x0 + axis_gap + 6.0
        overflow_margin = 48.0
        max_x = x0 + axis_gap + xmax * x_scale + overflow_margin
        full_end_x = x0 + axis_gap + m['invoke'] * x_scale
        for idx_p, (seg_name, val, cx) in enumerate(prefs):
            if seg_name == 'Host':
                # Bring host inward a bit from the absolute end
                biased_cx = min(max_x, full_end_x - 8.0)
                prefs[idx_p] = (seg_name, val, max(min_x, biased_cx))
                break
        # Allocate non-overlapping label x positions
        pref_xs = [cx for (_n, _v, cx) in prefs]
        xs_assigned = allocate_positions(pref_xs, min_x, max_x, min_sep)
        # Ensure Host label appears to the right of D2H by at least min_sep,
        # without pushing D2H left; let Host extend within overflow margin.
        try:
            host_idx = next(i for i,(n,_,_) in enumerate(prefs) if n == 'Host')
            d2h_idx  = next(i for i,(n,_,_) in enumerate(prefs) if n == 'D2H')
            if xs_assigned[host_idx] <= xs_assigned[d2h_idx]:
                desired = xs_assigned[d2h_idx] + min_sep
                xs_assigned[host_idx] = min(max_x, desired)
        except StopIteration:
            pass

        # Render labels for both bars above the time axis for visual consistency
        for (seg_name, val, _cx), lx in zip(prefs, xs_assigned):
            label_y1 = y_top - 14.0  # type above bar
            label_y2 = y_top - 2.0   # value just above type
            out.append(svg_text(lx, label_y1, seg_name, anchor='middle', size=11))
            out.append(svg_text(lx, label_y2, fmt_ms(val), anchor='middle', size=12))

        # Y category labels (model names) to the left
        out.append(svg_text(x0 - 10.0, y_center + 4.0, m['name'], anchor='end', size=13, weight='bold'))

    out.append('</svg>')

    out_path = Path(args.outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(out), encoding='utf-8')
    print(str(out_path))


if __name__ == '__main__':
    main()
