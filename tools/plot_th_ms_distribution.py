#!/usr/bin/env python3
import csv
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

BASE = Path('/home/10210/Desktop/OS')
SRC = BASE/'five_models/results/theory_chain_source_data.csv'
OUT_DIR = BASE/'five_models/results/plots'
OUT_PNG = OUT_DIR/'th_ms_mean_distribution.png'

def load_th_values(path: Path):
    vals = []
    with path.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            s = (r.get('Th_ms_mean') or '').strip()
            if not s:
                continue
            try:
                v = float(s)
            except Exception:
                continue
            vals.append(v)
    return vals

def main():
    vals = load_th_values(SRC)
    if not vals:
        print('No numeric Th_ms_mean values found in', SRC)
        return 1
    bw = 0.1  # bin width (ms)
    vmin = min(vals)
    vmax = max(vals)
    start = math.floor(vmin / bw) * bw
    end = math.ceil(vmax / bw) * bw
    nbins = max(1, int(round((end - start) / bw)))
    # build counts per bin
    counts = [0] * nbins
    for v in vals:
        idx = int((v - start) / bw)
        if idx < 0:
            idx = 0
        elif idx >= nbins:
            idx = nbins - 1
        counts[idx] += 1
    xs = [start + (i + 0.5) * bw for i in range(nbins)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5), dpi=140)
    plt.scatter(xs, counts, s=18, c='#1f77b4', alpha=0.9)
    plt.xlabel('Th_ms_mean (ms)')
    plt.ylabel('频率（数量）')
    plt.title('Th_ms_mean 分布（0.1 ms 分箱）')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(bw))
    ax.grid(True, axis='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG)
    print('Saved:', OUT_PNG)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

