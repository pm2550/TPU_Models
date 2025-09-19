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
OUT_DIR = RES_DIR/'plots_newBound'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Theory bandwidths used for bounds annotation
# Green line uses B_IN as LB; Red line uses UB (formerly B_IN2) as the second bound.
B_IN = float(__import__('os').environ.get('B_IN', '368.5'))   # MiB/s (lower bound reference)
UB = float(__import__('os').environ.get('UB', __import__('os').environ.get('B_IN2', '338.5')))     # MiB/s (upper bound reference)
# Note: we intentionally do NOT draw the previous yellow line anymore.

MODELS = [
    'densenet201_8seg_uniform_local',
    'inceptionv3_8seg_uniform_local',
    'resnet101_8seg_uniform_local',
    'resnet50_8seg_uniform_local',
    'xception_8seg_uniform_local',
]

# Optional: restrict to a subset of models via env FILTER_MODEL (comma-separated full model names)
import os as _os
_filter_env = (_os.environ.get('FILTER_MODEL') or '').strip()
if _filter_env:
    _allow = {s.strip() for s in _filter_env.split(',') if s.strip()}
    MODELS = [m for m in MODELS if m in _allow]

# Outlier filtering configuration (applies to scatter points per (model, K))
OUTLIER_FILTER = True
# Method priority: use MAD (median absolute deviation); if degenerate, fallback to IQR fence
OUTLIER_MAD_K = 5.0   # inliers within median ± K * (1.4826 * MAD)
OUTLIER_SHOW_GREY = False  # if True, plot excluded outliers as faint grey points

def _median(xs):
    n = len(xs)
    if n == 0:
        return float('nan')
    ys = sorted(xs)
    m = n // 2
    if n % 2 == 1:
        return ys[m]
    return 0.5 * (ys[m-1] + ys[m])

def _iqr_bounds(xs, k=1.5):
    if not xs:
        return (-float('inf'), float('inf'))
    ys = sorted(xs)
    n = len(ys)
    # simple quantile indices (approximate, no interpolation)
    q1 = ys[max(0, n//4 - 1)]
    q3 = ys[min(n-1, (3*n)//4)]
    iqr = max(q3 - q1, 0.0)
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return (lo, hi)

def split_inliers_outliers(values):
    """Return (inliers, outliers) after robust filtering.
    - Primary: MAD filter with K=OUTLIER_MAD_K
    - Fallback: IQR 1.5 fence when MAD ~ 0
    """
    vals = list(values)
    if len(vals) <= 3:
        return vals, []
    med = _median(vals)
    abs_dev = [abs(v - med) for v in vals]
    mad = _median(abs_dev)
    if mad > 1e-9:
        sigma = 1.4826 * mad
        lo = med - OUTLIER_MAD_K * sigma
        hi = med + OUTLIER_MAD_K * sigma
    else:
        lo, hi = _iqr_bounds(vals, k=1.5)
    inliers = [v for v in vals if (v >= lo and v <= hi)]
    outliers = [v for v in vals if not (v >= lo and v <= hi)]
    return inliers, outliers

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
    lb = defaultdict(lambda: math.nan)   # lower-bound time (ms)
    ub2 = defaultdict(lambda: math.nan)  # second bound time (formerly LB2, now labeled UB) (ms)
    for r in rows:
        if r.get('group_index') != 'TOTAL':
            continue
        model = r['model']
        try:
            K = int(r['K'])
            # Use hosted bounds so plots reflect host-side overhead
            Wi_lb = float(r['Wi_lb_ms_hosted'])
            Wi_lb2 = float(r.get('Wi_lb_ms_hosted_in2') or 'nan')
        except Exception:
            continue
        lb[(model, K)] = Wi_lb
        ub2[(model, K)] = Wi_lb2
    return lb, ub2


def plot_model(model, measured, lb, ub2):
    # derive a short display name, e.g., 'resnet101' from 'resnet101_8seg_uniform_local'
    short = model.split('_')[0]
    Ks = list(range(2,9))
    fig, ax = plt.subplots(figsize=(8,4.5), dpi=130)
    # Scatter measured: jitter x slightly to reduce overplot
    for K in Ks:
        arr_all = measured.get((model,K), [])
        if not arr_all:
            continue
        arr_in, arr_out = (arr_all, [])
        if OUTLIER_FILTER:
            arr_in, arr_out = split_inliers_outliers(arr_all)
            # Print a small note to stdout if we filtered points
            if arr_out:
                print(f"Filtered {len(arr_out)} outliers for {model} K={K} (kept {len(arr_in)}/{len(arr_all)})")
        # Plot inliers
        n = len(arr_in)
        if n:
            x = [K + (i - n/2) * (0.02 / max(n,1)) for i in range(n)]
            colors = plt.cm.Blues([0.3 + 0.7*(i/(max(1,n-1))) for i in range(n)])
            ax.scatter(x, arr_in, s=10, c=colors, alpha=0.6, edgecolors='none', label='_nolegend_')
        # Optionally show outliers as faint grey dots
        if OUTLIER_SHOW_GREY and arr_out:
            n2 = len(arr_out)
            x2 = [K + (i - n2/2) * (0.02 / max(n2,1)) for i in range(n2)]
            ax.scatter(x2, arr_out, s=8, c='lightgrey', alpha=0.5, edgecolors='none', label='_nolegend_')
    # Lines for theory bounds
    y_lb = [lb.get((model,K), math.nan) for K in Ks]
    y_ub2 = [ub2.get((model,K), math.nan) for K in Ks]
    ax.plot(Ks, y_lb, '-o', color='green', label=f'LB (B_in={B_IN:.1f} MiB/s)', linewidth=1.8, markersize=3)
    ax.plot(Ks, y_ub2, '-o', color='red', label=f'UB (UB={UB:.1f} MiB/s)', linewidth=1.8, markersize=3)

    ax.set_title(f"{short} — time per cycle vs K (ms) [{UB:.1f}-{B_IN:.1f} MiB/s]")
    ax.set_xlabel('K (segments)')
    ax.set_ylabel('Time per cycle (ms)')
    ax.set_xticks(Ks)
    ax.grid(True, linestyle='--', alpha=0.3)
    # Removed top-left annotation card per request

    # Place legend away from the top-left annotation card to avoid overlap
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    out = OUT_DIR/f"{short}_time_vs_K {UB:.1f}-{B_IN:.1f}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def summarize_bounds(measured):
    """
    Build a JSON-serializable dict with bound coverage and error stats per model and per K
    after filtering outliers. Error is distance to nearest bound if outside band, else 0.
    """
    rows = {}
    # Reload theory times (LB and UB2) for reference
    lb, ub2 = load_theory_times()
    for model in MODELS:
        short = model.split('_')[0]
        model_entry = {
            'short_name': short,
            'B_in_MiBps': B_IN,
            'UB_MiBps': UB,
            'by_K': {},
            'totals': {},
        }
        total_kept = total_all = 0
        total_in = total_below = total_above = 0
        err_list = []
        for K in range(2,9):
            arr_all = measured.get((model, K), [])
            arr_in, arr_out = split_inliers_outliers(arr_all) if OUTLIER_FILTER else (arr_all, [])
            total_all += len(arr_all)
            total_kept += len(arr_in)
            LB = lb.get((model, K), math.nan)
            UBv = ub2.get((model, K), math.nan)
            in_cnt = below_cnt = above_cnt = 0
            errs = []
            for v in arr_in:
                if math.isnan(LB) or math.isnan(UBv):
                    # can't classify without both bounds
                    continue
                if v < LB:
                    below_cnt += 1
                    errs.append(LB - v)
                elif v > UBv:
                    above_cnt += 1
                    errs.append(v - UBv)
                else:
                    in_cnt += 1
                    errs.append(0.0)
            total_in += in_cnt; total_below += below_cnt; total_above += above_cnt
            err_list.extend(errs)
            kept = len(arr_in)
            miss = kept - in_cnt
            miss_rate = (miss/kept) if kept else 0.0
            # basic error stats
            errs_sorted = sorted(errs)
            def pctl(p):
                if not errs_sorted:
                    return 0.0
                i = min(len(errs_sorted)-1, max(0, int(round(p*(len(errs_sorted)-1)))))
                return errs_sorted[i]
            model_entry['by_K'][K] = {
                'points_total': len(arr_all),
                'points_kept': kept,
                'points_filtered': len(arr_all) - kept,
                'LB_ms': None if math.isnan(LB) else round(LB, 3),
                'UB_ms': None if math.isnan(UBv) else round(UBv, 3),
                'inside_cnt': in_cnt,
                'below_cnt': below_cnt,
                'above_cnt': above_cnt,
                'miss_rate': round(miss_rate, 4),
                'err_mean_ms': round(sum(errs)/len(errs), 3) if errs else 0.0,
                'err_p50_ms': round(pctl(0.5), 3),
                'err_p95_ms': round(pctl(0.95), 3),
                'err_max_ms': round(max(errs_sorted), 3) if errs_sorted else 0.0,
            }
        kept_total = total_kept
        miss_total = total_below + total_above
        miss_rate_total = (miss_total/kept_total) if kept_total else 0.0
        err_list_sorted = sorted(err_list)
        def pctl2(p):
            if not err_list_sorted:
                return 0.0
            i = min(len(err_list_sorted)-1, max(0, int(round(p*(len(err_list_sorted)-1)))))
            return err_list_sorted[i]
        model_entry['totals'] = {
            'points_total': total_all,
            'points_kept': kept_total,
            'points_filtered': total_all - kept_total,
            'inside_cnt': total_in,
            'below_cnt': total_below,
            'above_cnt': total_above,
            'miss_rate': round(miss_rate_total, 4),
            'err_mean_ms': round(sum(err_list)/len(err_list), 3) if err_list else 0.0,
            'err_p50_ms': round(pctl2(0.5), 3),
            'err_p95_ms': round(pctl2(0.95), 3),
            'err_max_ms': round(max(err_list_sorted), 3) if err_list_sorted else 0.0,
        }
        rows[model] = model_entry
    return rows


def main():
    measured = load_measured_times()
    lb, ub2 = load_theory_times()
    outs = []
    for m in MODELS:
        outs.append(plot_model(m, measured, lb, ub2))
    # Write coverage and error stats JSON
    report = summarize_bounds(measured)
    out_json = OUT_DIR/"combo_bounds_report.json"
    import json
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("Saved figures:")
    for p in outs:
        print(p)
    print("Saved report:", out_json)

if __name__ == '__main__':
    main()
