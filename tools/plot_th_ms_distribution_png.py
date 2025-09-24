#!/usr/bin/env python3
import csv
import math
from pathlib import Path
from statistics import median
from PIL import Image, ImageDraw, ImageFont

BASE = Path('/home/10210/Desktop/OS')
SRC = BASE/'five_models/results/theory_chain_source_data.csv'
OUT_DIR = BASE/'five_models/results/plots'
OUT_PNG = OUT_DIR/'th_ms_mean_distribution.png'

def load_th_values_unique(path: Path):
    """Load Th_ms_mean but count each segment once.
    Rule: only single segments ('segX'), exclude 'segYto8', dedupe by (model, group_name),
    prefer K=8 if available.
    """
    by_key = {}
    with path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            g = (r.get('group_name') or '').strip()
            if not g.startswith('seg') or 'to8' in g:
                continue
            model = (r.get('model') or '').strip()
            try:
                K = int(r.get('K') or 0)
            except Exception:
                K = 0
            key = (model, g)
            cur = by_key.get(key)
            if cur is None or (K == 8) or (cur.get('K_val', 0) != 8 and K > cur.get('K_val', 0)):
                s = (r.get('Th_ms_mean') or '').strip()
                try:
                    v = float(s)
                except Exception:
                    continue
                by_key[key] = {'val': v, 'K_val': K}
    return [x['val'] for x in by_key.values()]

def main():
    vals = load_th_values_unique(SRC)
    if not vals:
        print('No numeric Th_ms_mean values found in', SRC)
        return 1

    bw = 0.1  # bin width in ms
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
    ymax = max(counts)

    # Canvas
    W, H = 1200, 520
    L, R, T, B = 80, 20, 30, 80
    plot_w = W - L - R
    plot_h = H - T - B
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (W, H), 'white')
    dr = ImageDraw.Draw(img)

    def x_to_px(x):
        return L + (x - start) / (end - start or 1.0) * plot_w

    def y_to_px(y):
        return T + (1.0 - (y / (ymax or 1.0))) * plot_h

    # Fonts (fallback to default)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 14)
        font_small = ImageFont.truetype('DejaVuSans.ttf', 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Axes
    dr.line([(L, T), (L, H - B)], fill=(51, 51, 51), width=1)
    dr.line([(L, H - B), (W - R, H - B)], fill=(51, 51, 51), width=1)

    # X ticks every 0.1 ms
    tick = bw
    tick_count = int(round((end - start) / tick))
    # If too many ticks, step them (draw every nth label)
    step = 1
    if tick_count > 120:
        step = max(1, tick_count // 120)
    for i in range(tick_count + 1):
        xv = start + i * tick
        xp = x_to_px(xv)
        # grid
        dr.line([(xp, T), (xp, H - B)], fill=(220, 220, 220), width=1)
        # tick
        dr.line([(xp, H - B), (xp, H - B + 6)], fill=(51, 51, 51), width=1)
        # labels
        if i % step == 0:
            txt = f"{xv:.1f}"
            tw, th = dr.textsize(txt, font=font_small)
            dr.text((xp - tw / 2, H - B + 10), txt, fill=(34, 34, 34), font=font_small)

    # Y ticks at integers
    for yv in range(0, ymax + 1):
        yp = y_to_px(yv)
        dr.line([(L, yp), (W - R, yp)], fill=(220, 220, 220), width=1)
        dr.text((L - 12 - dr.textsize(str(yv), font=font_small)[0], yp - 7), str(yv), fill=(34, 34, 34), font=font_small)

    # Points
    for xv, c in zip(xs, counts):
        if c <= 0:
            continue
        xp = x_to_px(xv)
        yp = y_to_px(c)
        # draw small circle
        r = 3
        dr.ellipse((xp - r, yp - r, xp + r, yp + r), fill=(31, 119, 180), outline=None)

    # Labels and title
    title = 'Th_ms_mean distribution (0.1 ms bins)'
    tw, th = dr.textsize(title, font=font)
    dr.text(((W - tw) / 2, 6), title, fill=(34, 34, 34), font=font)

    xlab = 'Time (ms)'
    xw, xh = dr.textsize(xlab, font=font)
    dr.text((L + (plot_w - xw) / 2, H - B + 40), xlab, fill=(34, 34, 34), font=font)

    ylab = 'Frequency (count)'
    # Render y label rotated by drawing each char (simple approach)
    # or skip rotation and place near axis
    dr.text((10, T + (plot_h - dr.textsize(ylab, font=font)[1]) / 2), ylab, fill=(34, 34, 34), font=font)

    # Stats
    N = len(vals)
    mean_v = sum(vals) / N
    med_v = median(vals)
    stats = f"N={N}, mean={mean_v:.3f}, median={med_v:.3f}"
    sw, sh = dr.textsize(stats, font=font)
    dr.text((W - R - sw, T + 8), stats, fill=(34, 34, 34), font=font)

    img.save(OUT_PNG)
    print('Saved:', OUT_PNG)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
