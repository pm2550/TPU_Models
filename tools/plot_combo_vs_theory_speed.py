#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict
import math
import matplotlib.pyplot as plt

BASE = Path('/home/10210/Desktop/OS')
THEORY_CSV = BASE/'five_models/results/theory_chain_times.csv'
MEASURED_CSV = BASE/'five_models/results/combo_cycle_times.csv'
OUT_DIR = BASE/'five_models/results/plots'


def read_theory_totals():
    rows = list(csv.DictReader(open(THEORY_CSV)))
    # {model: {K: (Wi_lb_ms, Wi_ub_ms)}}
    out = defaultdict(dict)
    for r in rows:
        if r.get('group_index') != 'TOTAL':
            continue
        try:
            K = int(r['K'])
        except Exception:
            continue
        m = r['model']
        try:
            lb = float(r['Wi_lb_ms'])
            ub = float(r['Wi_ub_ms'])
        except Exception:
            continue
        out[m][K] = (lb, ub)
    return out


def read_measured_means():
    rows = list(csv.DictReader(open(MEASURED_CSV)))
    # accumulate total_cycle_ms per (model,K)
    acc = defaultdict(list)
    for r in rows:
        try:
            K = int(r['K'])
            t = float(r['total_cycle_ms'])
        except Exception:
            continue
        acc[(r['model'], K)].append(t)
    # compute mean per (model,K)
    means = defaultdict(dict)
    for (m, K), arr in acc.items():
        if not arr:
            continue
        mean_ms = sum(arr) / len(arr)
        means[m][K] = mean_ms
    return means


def ms_to_speed_inf_per_s(ms):
    if not ms or ms <= 0:
        return float('nan')
    return 1000.0 / ms


def plot_for_model(model, theory, measured_means):
    ks = sorted(set(list(theory.get(model, {}).keys()) + list(measured_means.get(model, {}).keys())))
    if not ks:
        return None

    # Build curves
    speed_lb = []  # lower speed bound = 1000 / time_UB
    speed_ub = []  # upper speed bound = 1000 / time_LB
    speed_mean = []  # measured mean speed
    for K in ks:
        lb_ub = theory.get(model, {}).get(K)
        if lb_ub:
            lb_ms, ub_ms = lb_ub
            speed_lb.append(ms_to_speed_inf_per_s(ub_ms))
            speed_ub.append(ms_to_speed_inf_per_s(lb_ms))
        else:
            speed_lb.append(float('nan'))
            speed_ub.append(float('nan'))
        m_ms = measured_means.get(model, {}).get(K)
        speed_mean.append(ms_to_speed_inf_per_s(m_ms) if m_ms else float('nan'))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, speed_lb, '-o', color='#1f77b4', label='theory speed lower')
    ax.plot(ks, speed_ub, '-o', color='#ff7f0e', label='theory speed upper')
    ax.scatter(ks, speed_mean, color='#2ca02c', marker='s', s=40, label='measured mean')

    ax.set_title(f'Speed vs K (segments) — {model}')
    ax.set_xlabel('K (number of segments)')
    ax.set_ylabel('Speed (inferences/s)')
    ax.set_xticks(ks)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f'{model}_speed_vs_K.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main():
    theory = read_theory_totals()
    measured_means = read_measured_means()
    outs = []
    models = sorted(set(list(theory.keys()) + list(measured_means.keys())))
    for m in models:
        p = plot_for_model(m, theory, measured_means)
        if p:
            outs.append(p)
    if outs:
        print('Saved plots:')
        for p in outs:
            print(' -', p)
    else:
        print('No plots generated (missing data).')


if __name__ == '__main__':
    main()
