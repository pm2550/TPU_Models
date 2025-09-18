#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path('/home/10210/Desktop/OS')
RES_DIR = BASE/'five_models/results'
COMBO_CSV = RES_DIR/'combo_cycle_times.csv'
THEORY_CSV = RES_DIR/'theory_chain_times.csv'
OUT_DIR = RES_DIR/'plots'
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    'densenet201_8seg_uniform_local',
    'inceptionv3_8seg_uniform_local',
    'resnet101_8seg_uniform_local',
    'resnet50_8seg_uniform_local',
    'xception_8seg_uniform_local',
]

# Helper to collect measured total cycle time (ms) per (model, K)
def load_measured_times():
    rows = list(csv.DictReader(open(COMBO_CSV)))
    data = defaultdict(list)  # (model,K) -> list of time_ms
    for r in rows:
        model = r['model']
        K = int(r['K'])
        try:
            t_ms = float(r['total_cycle_ms'])
        except Exception:
            continue
        if t_ms <= 0:
            continue
        data[(model, K)].append(t_ms)
    return data


def load_theory_times():
    rows = list(csv.DictReader(open(THEORY_CSV)))
    # Only TOTAL rows; keep ms domain
    lb = defaultdict(lambda: math.nan)  # lower-bound time (ms)
    ub = defaultdict(lambda: math.nan)  # upper-bound time (ms)
    for r in rows:
        if r.get('group_index') != 'TOTAL':
            continue
        model = r['model']
        try:
            K = int(r['K'])
            # Use hosted bounds so plots reflect host-side overhead
            Wi_lb = float(r['Wi_lb_ms_hosted'])
            Wi_ub = float(r['Wi_ub_ms_hosted'])
        except Exception:
            continue
        lb[(model, K)] = Wi_lb
        ub[(model, K)] = Wi_ub
    return lb, ub


def plot_model(model, measured, lb, ub):
    Ks = list(range(2,9))
    fig, ax = plt.subplots(figsize=(8,4.5), dpi=130)
    # Scatter measured: jitter x slightly to reduce overplot
    for K in Ks:
        arr = measured.get((model,K), [])
        if not arr:
            continue
        x = [K + (i - len(arr)/2) * (0.02 / max(len(arr),1)) for i in range(len(arr))]
        colors = plt.cm.Blues([0.3 + 0.7*(i/(max(1,len(arr)-1))) for i in range(len(arr))])
        ax.scatter(x, arr, s=10, c=colors, alpha=0.6, edgecolors='none', label='_nolegend_')
    # Lines for theory bounds
    y_lb = [lb.get((model,K), math.nan) for K in Ks]
    y_ub = [ub.get((model,K), math.nan) for K in Ks]
    ax.plot(Ks, y_lb, '-o', color='red', label='theory LB', linewidth=1.5, markersize=3)
    ax.plot(Ks, y_ub, '-o', color='orange', label='theory UB', linewidth=1.5, markersize=3)

    ax.set_title(f"{model} — time per cycle vs K (ms)")
    ax.set_xlabel('K (segments)')
    ax.set_ylabel('Time per cycle (ms)')
    ax.set_xticks(Ks)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    out = OUT_DIR/f"{model}_time_vs_K.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    measured = load_measured_times()
    lb, ub = load_theory_times()
    outs = []
    for m in MODELS:
        outs.append(plot_model(m, measured, lb, ub))
    print("Saved:")
    for p in outs:
        print(p)

if __name__ == '__main__':
    main()
