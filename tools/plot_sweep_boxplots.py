#!/usr/bin/env python3
"""
Boxplot visualization for measured cycle times vs theory sweep bounds.

Inputs:
- Aggregated sweep CSV from tools/run_model_theory_sweeps.py (default: five_models/results/theory_sweeps/all_models.csv)
- Measured combo cycle times CSV (default: five_models/results/combo_cycle_times.csv)

Behavior:
- For each model, for K=2..8, build a boxplot of filtered measured cycle times
  (filter matches the sweep script: MAD k=5 with IQR fallback 1.5x fence).
- Overlay three lines (per K):
  - LB (green): sum of per-group LB_ms_total_hosted
  - expected_UB (yellow): sum of per-group UB_expected_ms_total_hosted
  - UB (red): sum of per-group UB_ms_total_hosted
- Save per-model figures to five_models/results/theory_sweeps/plots.
- Also output metrics CSVs summarizing coverage/violations and severity.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

BASE = Path('/home/10210/Desktop/OS')
SWEEP_CSV = BASE / 'five_models/results/theory_sweeps/all_models.csv'
MEAS_CSV = BASE / 'five_models/results/combo_cycle_times.csv'
OUT_DIR = BASE / 'five_models/results/theory_sweeps/plots'

# Display names for model prefixes in plot titles
DISPLAY_NAMES = {
    'resnet50': 'ResNet-50',
    'inceptionv3': 'InceptionV3',
    'resnet101': 'ResNet-101',
    'densenet201': 'DenseNet-201',
    'xception': 'Xception',
}

# Outlier filtering configuration (must match run_model_theory_sweeps.py)
MAD_K = 3.0  # default |x - median| <= MAD_K * (1.4826 * MAD)
IQR_K = 1.5  # default IQR fence


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Boxplot measured cycle times vs. theory sweep bounds')
    ap.add_argument('--sweep', type=Path, default=SWEEP_CSV, help='Aggregated sweep CSV path')
    ap.add_argument('--measured', type=Path, default=MEAS_CSV, help='Measured combo cycle CSV path')
    ap.add_argument('--out', type=Path, default=OUT_DIR, help='Output directory for plots and metrics')
    ap.add_argument('--models', type=str, default='', help='Comma-separated model names to include (default: all in sweep CSV)')
    ap.add_argument('--whis', type=str, default='range', help='Whiskers: "range" for min/max, or e.g. "5,95"')
    ap.add_argument('--filter-mode', type=str, default='mad', choices=['mad','iqr','none'],
                    help='Outlier filter mode for input points (default: mad)')
    ap.add_argument('--mad-k', type=float, default=MAD_K, help='MAD filter k (default: 5.0)')
    ap.add_argument('--iqr-k', type=float, default=IQR_K, help='IQR fence k (default: 1.5)')
    # Expected UB line visibility (default: hidden)
    ap.add_argument('--show-expected', dest='show_expected', action='store_true',
                    help='Show yellow Expected UB line (and legend entry)')
    ap.add_argument('--hide-expected', dest='show_expected', action='store_false', default=False,
                    help='Hide yellow Expected UB line (default)')
    return ap.parse_args()


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float('nan')
    m = n // 2
    if n % 2:
        return s[m]
    return 0.5 * (s[m-1] + s[m])


def mad_filter(values, *, mode: str='mad', mad_k: float=MAD_K, iqr_k: float=IQR_K):
    """Return (inliers, outliers) with MAD k and IQR fallback (matches sweep script).
    - Primary: keep |x - median| <= MAD_K * (1.4826 * MAD)
    - Fallback: IQR fence with k=IQR_K when MAD ~ 0
    """
    vals = list(values)
    if len(vals) <= 3:
        return vals, []
    med = median(vals)
    abs_dev = [abs(v - med) for v in vals]
    mad = median(abs_dev)
    if mode == 'none':
        lo, hi = -float('inf'), float('inf')
    elif mode == 'iqr':
        s = sorted(vals)
        q1 = s[int(0.25 * (len(s) - 1))]
        q3 = s[int(0.75 * (len(s) - 1))]
        iqr = q3 - q1
        if iqr > 0:
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
        else:
            lo, hi = -float('inf'), float('inf')
    elif mad > 1e-9:
        sigma = 1.4826 * mad
        lo = med - mad_k * sigma
        hi = med + mad_k * sigma
    else:
        s = sorted(vals)
        q1 = s[int(0.25 * (len(s) - 1))]
        q3 = s[int(0.75 * (len(s) - 1))]
        iqr = q3 - q1
        if iqr > 0:
            lo = q1 - iqr_k * iqr
            hi = q3 + iqr_k * iqr
        else:
            lo, hi = -float('inf'), float('inf')
    inliers = [x for x in vals if lo <= x <= hi]
    outliers = [x for x in vals if not (lo <= x <= hi)]
    return inliers, outliers


def load_measured(meas_csv: Path):
    data = defaultdict(list)  # (model, K) -> list[ms]
    with meas_csv.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            model = (r.get('model') or '').strip()
            try:
                K = int(r.get('K') or 0)
                ms = float(r.get('total_cycle_ms') or 0.0)
            except Exception:
                continue
            if K <= 0 or ms <= 0:
                continue
            data[(model, K)].append(ms)
    return data


def load_sweep_totals(sweep_csv: Path):
    """Load per-(model,K) sweep totals by summing per-group rows.
    Returns:
      models: sorted list of models
      lb[(model,K)], ub[(model,K)], exp[(model,K)] -> floats (ms)
      meta[model] -> dict with f_expected_gt_99, expected_coverage_rate, cycles_used, cycles_total, cycles_outliers_dropped
    """
    lb = defaultdict(lambda: math.nan)
    ub = defaultdict(lambda: math.nan)
    exp = defaultdict(lambda: math.nan)
    meta = {}
    sums_lb = defaultdict(float)
    sums_ub = defaultdict(float)
    sums_exp = defaultdict(float)
    models_set = set()
    with sweep_csv.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            model = (r.get('model') or '').strip()
            if not model:
                continue
            models_set.add(model)
            try:
                K = int(r.get('K') or 0)
            except Exception:
                continue
            # Sum group totals per (model, K)
            try:
                sums_lb[(model, K)] += float(r.get('LB_ms_total_hosted') or 0.0)
            except Exception:
                pass
            try:
                sums_ub[(model, K)] += float(r.get('UB_ms_total_hosted') or 0.0)
            except Exception:
                pass
            try:
                sums_exp[(model, K)] += float(r.get('UB_expected_ms_total_hosted') or 0.0)
            except Exception:
                pass
            # Capture meta once per model (same across rows)
            if model not in meta:
                try:
                    meta[model] = {
                        'f_expected_gt_99': float(r.get('f_expected_gt_99') or 'nan'),
                        'expected_coverage_rate': float(r.get('expected_coverage_rate') or 'nan'),
                        'cycles_used': int(r.get('cycles_used') or 0),
                        'cycles_total': int(r.get('cycles_total') or 0),
                        'cycles_outliers_dropped': int(r.get('cycles_outliers_dropped') or 0),
                    }
                except Exception:
                    meta[model] = {}
    # Finalize
    for key, v in sums_lb.items():
        lb[key] = v
    for key, v in sums_ub.items():
        ub[key] = v
    for key, v in sums_exp.items():
        exp[key] = v
    models = sorted(models_set)
    return models, lb, ub, exp, meta


def compute_metrics_for_model(model: str, Ks: list[int], measured, lb, ub, exp, *, mode: str, mad_k: float, iqr_k: float):
    """Compute coverage/violation metrics for a model (aggregated across Ks).
    Returns a dict with rates and severities for UB and expected_UB, and per-K details list.
    """
    # Aggregated counters
    agg = {
        'n_total': 0,
        'within_LB_UB': 0,
        'within_LB_Exp': 0,
        'above_UB': 0,
        'above_Exp': 0,
        'below_LB': 0,
        # severity accumulators (only above UB/Exp), ratio and ms
        'sev_UB_sum_ratio': 0.0,
        'sev_UB_sum_ms': 0.0,
        'sev_UB_count': 0,
        'sev_Exp_sum_ratio': 0.0,
        'sev_Exp_sum_ms': 0.0,
        'sev_Exp_count': 0,
        'sev_below_LB_sum_ratio': 0.0,
        'sev_below_LB_sum_ms': 0.0,
        'sev_below_LB_count': 0,
    }
    per_k = []
    for K in Ks:
        arr_all = measured.get((model, K), [])
        if not arr_all:
            continue
        arr_in, _ = mad_filter(arr_all, mode=mode, mad_k=mad_k, iqr_k=iqr_k)
        l = lb.get((model, K), math.nan)
        u = ub.get((model, K), math.nan)
        e = exp.get((model, K), math.nan)
        if not (math.isfinite(l) and math.isfinite(u) and math.isfinite(e)):
            continue
        lo = min(l, u)
        hi = max(l, u)
        n_total = len(arr_in)
        within_LB_UB = 0
        within_LB_Exp = 0
        above_UB = 0
        above_Exp = 0
        below_LB = 0
        sev_UB_sum_ratio = sev_UB_sum_ms = 0.0
        sev_UB_count = 0
        sev_Exp_sum_ratio = sev_Exp_sum_ms = 0.0
        sev_Exp_count = 0
        sev_below_LB_sum_ratio = sev_below_LB_sum_ms = 0.0
        sev_below_LB_count = 0
        for t in arr_in:
            # coverage wrt [LB, UB]
            if lo <= t <= hi:
                within_LB_UB += 1
            elif t < lo:
                below_LB += 1
                if lo > 0:
                    sev_below_LB_sum_ratio += (lo - t) / lo
                sev_below_LB_sum_ms += (lo - t)
                sev_below_LB_count += 1
            else:  # t > hi
                above_UB += 1
                if hi > 0:
                    sev_UB_sum_ratio += (t - hi) / hi
                sev_UB_sum_ms += (t - hi)
                sev_UB_count += 1
            # coverage wrt [LB, Exp]
            hi2 = max(l, e)
            lo2 = min(l, e)
            if lo2 <= t <= hi2:
                within_LB_Exp += 1
            elif t > hi2:
                above_Exp += 1
                if hi2 > 0:
                    sev_Exp_sum_ratio += (t - hi2) / hi2
                sev_Exp_sum_ms += (t - hi2)
                sev_Exp_count += 1
            # If t < lo2, it's below LB and counted already via below_LB
        agg['n_total'] += n_total
        agg['within_LB_UB'] += within_LB_UB
        agg['within_LB_Exp'] += within_LB_Exp
        agg['above_UB'] += above_UB
        agg['above_Exp'] += above_Exp
        agg['below_LB'] += below_LB
        agg['sev_UB_sum_ratio'] += sev_UB_sum_ratio
        agg['sev_UB_sum_ms'] += sev_UB_sum_ms
        agg['sev_UB_count'] += sev_UB_count
        agg['sev_Exp_sum_ratio'] += sev_Exp_sum_ratio
        agg['sev_Exp_sum_ms'] += sev_Exp_sum_ms
        agg['sev_Exp_count'] += sev_Exp_count
        agg['sev_below_LB_sum_ratio'] += sev_below_LB_sum_ratio
        agg['sev_below_LB_sum_ms'] += sev_below_LB_sum_ms
        agg['sev_below_LB_count'] += sev_below_LB_count
        per_k.append({
            'model': model,
            'K': K,
            'n_total': n_total,
            'coverage_LB_UB': (within_LB_UB / n_total) if n_total else 0.0,
            'coverage_LB_Exp': (within_LB_Exp / n_total) if n_total else 0.0,
            'violation_above_UB_rate': (above_UB / n_total) if n_total else 0.0,
            'violation_above_Exp_rate': (above_Exp / n_total) if n_total else 0.0,
            'violation_below_LB_rate': (below_LB / n_total) if n_total else 0.0,
            'sev_above_UB_ratio_avg': (sev_UB_sum_ratio / sev_UB_count) if sev_UB_count else 0.0,
            'sev_above_Exp_ratio_avg': (sev_Exp_sum_ratio / sev_Exp_count) if sev_Exp_count else 0.0,
            'sev_below_LB_ratio_avg': (sev_below_LB_sum_ratio / sev_below_LB_count) if sev_below_LB_count else 0.0,
            'sev_overall_LB_UB_ratio_avg': ((sev_UB_sum_ratio + sev_below_LB_sum_ratio) / (sev_UB_count + sev_below_LB_count)) if (sev_UB_count + sev_below_LB_count) else 0.0,
            'sev_overall_LB_Exp_ratio_avg': ((sev_Exp_sum_ratio + sev_below_LB_sum_ratio) / (sev_Exp_count + sev_below_LB_count)) if (sev_Exp_count + sev_below_LB_count) else 0.0,
        })
    # Aggregate metrics
    N = agg['n_total']
    model_metrics = {
        'model': model,
        'n_total': N,
        'coverage_LB_UB': (agg['within_LB_UB'] / N) if N else 0.0,
        'coverage_LB_Exp': (agg['within_LB_Exp'] / N) if N else 0.0,
        'violation_above_UB_rate': (agg['above_UB'] / N) if N else 0.0,
        'violation_above_Exp_rate': (agg['above_Exp'] / N) if N else 0.0,
        'violation_below_LB_rate': (agg['below_LB'] / N) if N else 0.0,
        'sev_above_UB_ratio_avg': (agg['sev_UB_sum_ratio'] / agg['sev_UB_count']) if agg['sev_UB_count'] else 0.0,
        'sev_above_Exp_ratio_avg': (agg['sev_Exp_sum_ratio'] / agg['sev_Exp_count']) if agg['sev_Exp_count'] else 0.0,
        'sev_below_LB_ratio_avg': (agg['sev_below_LB_sum_ratio'] / agg['sev_below_LB_count']) if agg['sev_below_LB_count'] else 0.0,
        'sev_overall_LB_UB_ratio_avg': ((agg['sev_UB_sum_ratio'] + agg['sev_below_LB_sum_ratio']) / (agg['sev_UB_count'] + agg['sev_below_LB_count'])) if (agg['sev_UB_count'] + agg['sev_below_LB_count']) else 0.0,
        'sev_overall_LB_Exp_ratio_avg': ((agg['sev_Exp_sum_ratio'] + agg['sev_below_LB_sum_ratio']) / (agg['sev_Exp_count'] + agg['sev_below_LB_count'])) if (agg['sev_Exp_count'] + agg['sev_below_LB_count']) else 0.0,
    }
    return model_metrics, per_k


def plot_model_boxplot(model: str, Ks: list[int], measured, lb, ub, exp, out_dir: Path, whis=(5, 95), *, mode: str, mad_k: float, iqr_k: float, f_expected: float | None = None, show_expected: bool = False):
    short = model.split('_')[0]
    disp_name = DISPLAY_NAMES.get(short.lower(), short)
    # Prepare data for K=2..8 (skip empty groups to avoid matplotlib issue)
    xs = []
    data = []
    y_lb = []
    y_ub = []
    y_exp = []
    for K in Ks:
        arr = measured.get((model, K), [])
        if not arr:
            continue
        inliers, _ = mad_filter(arr, mode=mode, mad_k=mad_k, iqr_k=iqr_k)
        if not inliers:
            continue
        xs.append(K)
        data.append(inliers)
        y_lb.append(lb.get((model, K), math.nan))
        y_ub.append(ub.get((model, K), math.nan))
        y_exp.append(exp.get((model, K), math.nan))

    if not data:
        return None

    # Use a 4:3 aspect ratio for the figure and enable constrained layout
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140, constrained_layout=True)
    # Bump all font sizes by +2 points (title, labels, ticks, legends)
    try:
        base_fs = float(matplotlib.rcParams.get('font.size', 10))
    except Exception:
        base_fs = 10.0
    big_fs = base_fs + 8.0  # 再放大两号（总+8pt）
    label_fs = big_fs + 2.0  # 仅坐标轴标签再放大两号
    tick_fs = big_fs + 2.0   # 刻度文字再放大两号
    bp = ax.boxplot(
        data,
        positions=list(range(len(xs))),
        widths=0.6,
        showmeans=True,
        showfliers=False,  # 去掉黑色圆点（异常点）
        meanprops={
            "marker": "^",
            "markersize": 6,
            "markerfacecolor": "#1f77b4",  # 蓝色三角 = 均值
            "markeredgecolor": "#1f77b4",
        },
        medianprops={
            "color": "#ff7f0e",  # 橙色线 = 中位数
            "linewidth": 1.4,
        },
        whis=whis,
        patch_artist=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    # 统一样式（以防 matplotlib 版本对 props 支持差异）
    for box in bp.get('boxes', []):
        box.set(facecolor='white', edgecolor='black', linewidth=1.2)
    for whisker in bp.get('whiskers', []):
        whisker.set(color='black', linewidth=1.0)
    for cap in bp.get('caps', []):
        cap.set(color='black', linewidth=1.0)
    for med in bp.get('medians', []):
        med.set(color='#ff7f0e', linewidth=1.4)

    # Overlay lines for bounds aligned to positions
    pos_to_K = {i: k for i, k in enumerate(xs)}
    pos = list(range(len(xs)))
    def y_for(series):
        return [series.get((model, pos_to_K[i]), math.nan) for i in pos]
    # We already collected y_* arrays in the same order; use those
    ax.plot(pos, y_lb, '-o', color='green', linewidth=1.8, markersize=6, label='LB')
    if show_expected:
        label_exp = 'Expected UB'
        try:
            if f_expected is not None and math.isfinite(float(f_expected)):
                label_exp = f'Expected UB (f={float(f_expected):.2f})'
        except Exception:
            pass
        ax.plot(pos, y_exp, '-o', color='gold', linewidth=1.8, markersize=3, label=label_exp)
    ax.plot(pos, y_ub, '-o', color='red', linewidth=1.8, markersize=6, label='UB')

    # X-axis labels as actual K values
    ax.set_xticks(pos)
    ax.set_xticklabels(xs)
    ax.set_xlabel('K (segments)', fontsize=label_fs, fontweight='bold')
    ax.set_ylabel('Time per cycle (ms)', fontsize=label_fs, fontweight='bold')
    # Fix y-axis ticks: 10 units per major tick
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.tick_params(axis='both', labelsize=tick_fs)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.6)
    # Top-left legend: explain box semantics
    whisk_h = Line2D([0], [0], color='black', linewidth=1.2, label='Full range')
    box_h = Patch(facecolor='white', edgecolor='black', linewidth=1.2, label='P25–P75')
    mean_h = Line2D([0], [0], marker='^', linestyle='None', markersize=6,
                    markerfacecolor='#1f77b4', markeredgecolor='#1f77b4', label='Mean')
    med_h = Line2D([0], [0], color='#ff7f0e', linewidth=1.4, label='Median')
    legend_box = ax.legend(handles=[whisk_h, box_h, mean_h, med_h],
                           loc='upper left', fontsize=big_fs, framealpha=0.9)

    # Bottom-right legend: theory lines
    legend_lines = ax.legend(loc='lower right', fontsize=big_fs, framealpha=0.85)
    ax.add_artist(legend_box)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{short}_box_vs_K.png'
    # Save with tight bounding box to avoid title clipping on the right
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    # Also save a vector PDF alongside PNG (lossless)
    out_pdf = out_dir / f'{short}_box_vs_K.pdf'
    try:
        fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.1)
    except Exception:
        pass
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    out_dir = args.out
    models, lb, ub, exp, meta = load_sweep_totals(args.sweep)
    measured = load_measured(args.measured)
    if args.models.strip():
        include = set(x.strip() for x in args.models.split(',') if x.strip())
        models = [m for m in models if m in include]
    Ks = list(range(2, 9))
    # 解析 whis：默认使用 'range'（最小到最大）。也支持 "5,95" 百分位或单一倍数(浮点数)
    if isinstance(args.whis, str) and args.whis.strip().lower() == 'range':
        whis = (0, 100)  # 使用最小/最大（以百分位表示）
    else:
        try:
            parts = [p.strip() for p in (args.whis.split(',') if isinstance(args.whis, str) else list(args.whis))]
            if len(parts) == 1:
                whis = float(parts[0])
            else:
                whis = tuple(float(x) for x in parts)
        except Exception:
            whis = 'range'

    # Plot per model and collect metrics
    out_paths = []
    metrics_rows = []
    per_k_rows = []
    for model in models:
        meta_m = meta.get(model, {})
        f_expected = meta_m.get('f_expected_gt_99') if isinstance(meta_m, dict) else None
        p = plot_model_boxplot(model, Ks, measured, lb, ub, exp, out_dir, whis=whis,
                               mode=args.filter_mode, mad_k=args.mad_k, iqr_k=args.iqr_k,
                               f_expected=f_expected, show_expected=args.show_expected)
        if p:
            out_paths.append(p)
        mrow, krows = compute_metrics_for_model(model, Ks, measured, lb, ub, exp,
                                               mode=args.filter_mode, mad_k=args.mad_k, iqr_k=args.iqr_k)
        # Attach sweep meta
        mrow.update({
            'f_expected_gt_99': meta_m.get('f_expected_gt_99'),
            'expected_coverage_rate': meta_m.get('expected_coverage_rate'),
            'cycles_used': meta_m.get('cycles_used'),
            'cycles_total': meta_m.get('cycles_total'),
            'cycles_outliers_dropped': meta_m.get('cycles_outliers_dropped'),
        })
        metrics_rows.append(mrow)
        per_k_rows.extend(krows)

    # Write metrics CSVs
    out_dir.mkdir(parents=True, exist_ok=True)
    if metrics_rows:
        cols = [
            'model','n_total','coverage_LB_UB','coverage_LB_Exp',
            'violation_above_UB_rate','violation_above_Exp_rate','violation_below_LB_rate',
            'sev_above_UB_ratio_avg','sev_above_Exp_ratio_avg','sev_below_LB_ratio_avg',
            'sev_overall_LB_UB_ratio_avg','sev_overall_LB_Exp_ratio_avg',
            'f_expected_gt_99','expected_coverage_rate','cycles_used','cycles_total','cycles_outliers_dropped'
        ]
        with (out_dir / 'metrics_per_model.csv').open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader(); wr.writerows(metrics_rows)
    if per_k_rows:
        cols_k = [
            'model','K','n_total','coverage_LB_UB','coverage_LB_Exp',
            'violation_above_UB_rate','violation_above_Exp_rate','violation_below_LB_rate',
            'sev_above_UB_ratio_avg','sev_above_Exp_ratio_avg','sev_below_LB_ratio_avg',
            'sev_overall_LB_UB_ratio_avg','sev_overall_LB_Exp_ratio_avg'
        ]
        with (out_dir / 'metrics_per_model_K.csv').open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=cols_k)
            wr.writeheader(); wr.writerows(per_k_rows)

    # Print brief summary
    if out_paths:
        print('Saved plots (PNG):')
        for p in out_paths:
            print(p)
        # Also print corresponding PDF paths
        print('Saved plots (PDF):')
        for p in out_paths:
            try:
                print(Path(p).with_suffix('.pdf'))
            except Exception:
                pass
    if metrics_rows:
        print('Saved metrics:', out_dir / 'metrics_per_model.csv')
    if per_k_rows:
        print('Saved metrics (per-K):', out_dir / 'metrics_per_model_K.csv')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
