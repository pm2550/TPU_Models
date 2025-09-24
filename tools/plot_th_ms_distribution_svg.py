#!/usr/bin/env python3
import csv
import math
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
SRC = BASE/'five_models/results/theory_chain_source_data.csv'
OUT_DIR = BASE/'five_models/results/plots'
OUT_SVG = OUT_DIR/'th_ms_mean_distribution.svg'

def load_th_values_unique(path: Path):
    """Load Th_ms_mean but count each segment once.
    Rule: keep only rows where group_name is a single segment (e.g., 'seg1'),
    prefer K=8 if duplicates exist. Key is (model, group_name).
    """
    by_key = {}
    with path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            g = (r.get('group_name') or '').strip()
            if not g.startswith('seg') or 'to8' in g:
                continue
            model = (r.get('model') or '').strip()
            # prefer K=8
            try:
                K = int(r.get('K') or 0)
            except Exception:
                K = 0
            key = (model, g)
            cur = by_key.get(key)
            if cur is None or (K == 8) or (cur.get('K_val', 0) != 8 and K > cur.get('K_val', 0)):
                # parse value
                s = (r.get('Th_ms_mean') or '').strip()
                try:
                    v = float(s)
                except Exception:
                    continue
                by_key[key] = {'val': v, 'K_val': K}
    return [x['val'] for x in by_key.values()]

def svg_escape(text: str) -> str:
    return (text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))

def main():
    vals = load_th_values_unique(SRC)
    if not vals:
        print('No numeric Th_ms_mean values found in', SRC)
        return 1
    bw = 0.1
    vmin = min(vals)
    vmax = max(vals)
    start = math.floor(vmin / bw) * bw
    end = math.ceil(vmax / bw) * bw
    nbins = max(1, int(round((end - start) / bw)))
    counts = [0] * nbins
    for v in vals:
        idx = int((v - start) / bw)
        if idx < 0:
            idx = 0
        elif idx >= nbins:
            idx = nbins - 1
        counts[idx] += 1
    xs = [start + (i + 0.5) * bw for i in range(nbins)]
    ymax = max(counts) if counts else 1

    # SVG canvas
    W, H = 1000, 420
    L, R, T, B = 70, 20, 20, 60  # margins
    plot_w = W - L - R
    plot_h = H - T - B

    def x_to_px(x):
        return L + (x - start) / (end - start or 1.0) * plot_w

    def y_to_px(y):
        return T + (1.0 - (y / (ymax or 1.0))) * plot_h

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parts = []
    parts.append(f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}'>")
    parts.append("<style> .axis{stroke:#333;stroke-width:1} .grid{stroke:#ccc;stroke-width:0.5} .pt{fill:#1f77b4} text{font-family:sans-serif;font-size:12px;fill:#222} </style>")

    # Axes
    parts.append(f"<line class='axis' x1='{L}' y1='{T}' x2='{L}' y2='{H-B}'/>")
    parts.append(f"<line class='axis' x1='{L}' y1='{H-B}' x2='{W-R}' y2='{H-B}'/>")

    # X ticks every 0.1 ms
    tick = bw
    tick_count = int(round((end - start) / tick))
    for i in range(tick_count + 1):
        xv = start + i * tick
        xp = x_to_px(xv)
        parts.append(f"<line class='grid' x1='{xp}' y1='{T}' x2='{xp}' y2='{H-B}'/>")
        parts.append(f"<line class='axis' x1='{xp}' y1='{H-B}' x2='{xp}' y2='{H-B+5}'/>")
        parts.append(f"<text x='{xp}' y='{H-B+18}' text-anchor='middle'>{xv:.1f}</text>")

    # Y ticks at integer counts
    for yv in range(0, ymax + 1):
        yp = y_to_px(yv)
        parts.append(f"<line class='grid' x1='{L}' y1='{yp}' x2='{W-R}' y2='{yp}'/>")
        parts.append(f"<text x='{L-8}' y='{yp+4}' text-anchor='end'>{yv}</text>")

    # Points
    for xv, c in zip(xs, counts):
        if c <= 0:
            continue
        xp = x_to_px(xv)
        yp = y_to_px(c)
        parts.append(f"<circle class='pt' cx='{xp:.2f}' cy='{yp:.2f}' r='3' />")

    # Labels
    parts.append(f"<text x='{(L+W-R)/2}' y='{T-2}' text-anchor='middle'>Th_ms_mean distribution (0.1 ms bins)</text>")
    parts.append(f"<text x='{(L+W-R)/2}' y='{H-10}' text-anchor='middle'>Time (ms)</text>")
    # Rotated Y label
    parts.append(f"<text x='12' y='{(T+H-B)/2}' transform='rotate(-90 12 {(T+H-B)/2})' text-anchor='middle'>Frequency (count)</text>")
    # Stats annotation
    import statistics as _st
    mean_v = sum(vals)/len(vals)
    median_v = _st.median(vals)
    parts.append(f"<text x='{W-R-5}' y='{T+15}' text-anchor='end'>N={len(vals)}, mean={mean_v:.3f}, median={median_v:.3f}</text>")

    parts.append("</svg>")
    OUT_SVG.write_text("\n".join(parts))
    print('Saved:', OUT_SVG)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
