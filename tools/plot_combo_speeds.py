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

# Theory bandwidths used for LOWER-BOUND (LB) variants
# Green line uses B_IN; Red line uses B_IN2/B_OUT2 (variant). These are only for annotation.
B_IN = 344 # MiB/s (original lower-bound)
B_IN2 = 344 # MiB/s (variant lower-bound)
# You can also annotate B_OUT if desired (used by the upper bound derivation)
B_OUT = 87.0   # MiB/s
B_OUT2 = 35.0  # MiB/s (variant lower-bound Cout)

MODELS = [
    'densenet201_8seg_uniform_local',
    'inceptionv3_8seg_uniform_local',
    'resnet101_8seg_uniform_local',
    'resnet50_8seg_uniform_local',
    'xception_8seg_uniform_local',
]

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
    lb = defaultdict(lambda: math.nan)  # lower-bound time (ms)
    lb2 = defaultdict(lambda: math.nan) # variant lower-bound using B_IN2 (ms)
    ub = defaultdict(lambda: math.nan)  # upper-bound time (ms)
    for r in rows:
        if r.get('group_index') != 'TOTAL':
            continue
        model = r['model']
        try:
            K = int(r['K'])
            # Use hosted bounds so plots reflect host-side overhead
            Wi_lb = float(r['Wi_lb_ms_hosted'])
            Wi_lb2 = float(r.get('Wi_lb_ms_hosted_in2') or 'nan')
            Wi_ub = float(r['Wi_ub_ms_hosted'])
        except Exception:
            continue
        lb[(model, K)] = Wi_lb
        lb2[(model, K)] = Wi_lb2
        ub[(model, K)] = Wi_ub
    return lb, lb2, ub


def plot_model(model, measured, lb, lb2, ub):
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
    y_lb2 = [lb2.get((model,K), math.nan) for K in Ks]
    y_ub = [ub.get((model,K), math.nan) for K in Ks]
    # Green: show simple label without parentheses details
    ax.plot(Ks, y_lb, '-o', color='green', label='theory LB', linewidth=1.8, markersize=3)
    # Draw LB2 only if it is not identical to LB (numerically) or if any variant bandwidth differs
    same_series = True
    for a,b in zip(y_lb, y_lb2):
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        if abs(a-b) > 1e-6:
            same_series = False
            break
    if (not same_series) or (B_IN2 != B_IN or B_OUT2 != B_OUT):
        # Red: rename to match the former yellow line name "theory UB"
        ax.plot(Ks, y_lb2, '-o', color='red', label='theory UB', linewidth=1.8, markersize=3)
    # Yellow line hidden per request (previously: theory UB)
    # ax.plot(Ks, y_ub, '-o', color='yellow', label='theory UB', linewidth=1.8, markersize=3)

    ax.set_title(f"{short} — time per cycle vs K (ms)")
    ax.set_xlabel('K (segments)')
    ax.set_ylabel('Time per cycle (ms)')
    ax.set_xticks(Ks)
    ax.grid(True, linestyle='--', alpha=0.3)
    # Removed top-left annotation card per request

    # Place legend away from the top-left annotation card to avoid overlap
    ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    out = OUT_DIR/f"{short}_time_vs_K.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main():
    measured = load_measured_times()
    lb, lb2, ub = load_theory_times()
    outs = []
    for m in MODELS:
        outs.append(plot_model(m, measured, lb, lb2, ub))
    print("Saved:")
    for p in outs:
        print(p)

    # After plotting, compute coverage metrics for filtered (inlier) points
    print("\nCoverage metrics (filtered points, per model; UB = 红线LB2)")
    Ks = list(range(2,9))
    overall = {
        'n_total': 0,
        'n_within': 0,
        'n_above': 0,
        'n_below': 0,
        # Above-only severity accumulators
        'sev_sum_ratio': 0.0,
        'sev_sum_ms': 0.0,
        'sev_count': 0,
        # Overall OOB severity accumulators (both below and above)
        'sev_oob_sum_ratio': 0.0,
        'sev_oob_sum_ms': 0.0,
        'sev_oob_count': 0,
    }
    for model in MODELS:
        n_total = n_within = n_above = n_below = 0
        # Above-only severity (ratio, ms, count)
        sev_sum_ratio = sev_sum_ms = 0.0
        sev_count = 0
        # Overall OOB severity (ratio, ms, count)
        sev_oob_sum_ratio = sev_oob_sum_ms = 0.0
        sev_oob_count = 0
        for K in Ks:
            arr_all = measured.get((model, K), [])
            if not arr_all:
                continue
            arr_in, _ = (arr_all, [])
            if OUTLIER_FILTER:
                arr_in, _ = split_inliers_outliers(arr_all)
            l = lb.get((model, K), math.nan)
            u = lb2.get((model, K), math.nan)  # use red line (LB2) as UB
            if not (math.isfinite(l) and math.isfinite(u)):
                continue
            # ensure proper ordering for interval checks
            lo = min(l, u)
            hi = max(l, u)
            for t in arr_in:
                n_total += 1
                if t < lo:
                    n_below += 1
                    # OOB overall severity relative to lower bound
                    if lo > 0:
                        sev_oob_sum_ratio += (lo - t) / lo
                    sev_oob_sum_ms += (lo - t)
                    sev_oob_count += 1
                elif t > hi:
                    n_above += 1
                    if hi > 0:
                        sev_sum_ratio += (t - hi) / hi
                    sev_sum_ms += (t - hi)
                    sev_count += 1
                    # Also accumulate into overall OOB
                    if hi > 0:
                        sev_oob_sum_ratio += (t - hi) / hi
                    sev_oob_sum_ms += (t - hi)
                    sev_oob_count += 1
                else:
                    n_within += 1
        # Aggregate overall
        overall['n_total'] += n_total
        overall['n_within'] += n_within
        overall['n_above'] += n_above
        overall['n_below'] += n_below
        overall['sev_sum_ratio'] += sev_sum_ratio
        overall['sev_sum_ms'] += sev_sum_ms
        overall['sev_count'] += sev_count
        overall['sev_oob_sum_ratio'] += sev_oob_sum_ratio
        overall['sev_oob_sum_ms'] += sev_oob_sum_ms
        overall['sev_oob_count'] += sev_oob_count
        # Rates for this model
        if n_total > 0:
            cover_rate = n_within / n_total
            oob_rate = (n_above + n_below) / n_total
            above_rate = n_above / n_total
            sev_ratio_avg = (sev_sum_ratio / sev_count) if sev_count > 0 else 0.0
            sev_oob_ratio_avg = (sev_oob_sum_ratio / sev_oob_count) if sev_oob_count > 0 else 0.0
            # Print concise per-model metrics
            short = model.split('_')[0]
            print(f"- {short}: 覆盖率={cover_rate:.1%}, 超界率={oob_rate:.1%}, 超上界率={above_rate:.1%}, 越界严重度(超上界,均值比例)={sev_ratio_avg:.2%}, 越界严重度(总体,均值比例)={sev_oob_ratio_avg:.2%} | 样本数={n_total}")
        else:
            short = model.split('_')[0]
            print(f"- {short}: 无可用样本（缺少上下界或测量点）")

    # Overall summary
    n_total = overall['n_total']
    if n_total > 0:
        cover_rate = overall['n_within'] / n_total
        oob_rate = (overall['n_above'] + overall['n_below']) / n_total
        above_rate = overall['n_above'] / n_total
        sev_ratio_avg = (overall['sev_sum_ratio'] / overall['sev_count']) if overall['sev_count'] > 0 else 0.0
        sev_oob_ratio_avg = (overall['sev_oob_sum_ratio'] / overall['sev_oob_count']) if overall['sev_oob_count'] > 0 else 0.0
        print(f"总体: 覆盖率={cover_rate:.1%}, 超界率={oob_rate:.1%}, 超上界率={above_rate:.1%}, 越界严重度(超上界,均值比例)={sev_ratio_avg:.2%}, 越界严重度(总体,均值比例)={sev_oob_ratio_avg:.2%} | 样本数={n_total}")
    else:
        print("总体: 无可用样本（缺少上下界或测量点）")

if __name__ == '__main__':
    main()
