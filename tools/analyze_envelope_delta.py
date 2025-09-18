#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyze the relationship between envelope delta (cur_pre - bak_pre) and per-segment IO sizes/spans.

Inputs:
- results/pure_pre_compare_detailed.csv (from compare_pure_pre_csvs.py)
- results/models_local_batch_usbmon/single/combined_summary_span.json (bytes_in/out, span ms)

Outputs:
- results/envelope_delta_analysis.txt (human-readable summary)
- results/envelope_delta_merged.csv (row-wise merged features)
"""

import csv
import json
from pathlib import Path
from math import isnan

BASE = Path('/home/10210/Desktop/OS')
DET = BASE / 'results/pure_pre_compare_detailed.csv'
SPAN = BASE / 'results/models_local_batch_usbmon/single/combined_summary_span.json'
OUT_TXT = BASE / 'results/envelope_delta_analysis.txt'
OUT_CSV = BASE / 'results/envelope_delta_merged.csv'


def to_float(v):
    try:
        return None if v in (None, '') else float(v)
    except Exception:
        return None


def pearson(xs, ys):
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        n += 1
        sx += x; sy += y
        sxx += x*x; syy += y*y
        sxy += x*y
    if n < 2:
        return None
    num = n*sxy - sx*sy
    den = ((n*sxx - sx*sx) * (n*syy - sy*sy)) ** 0.5
    if den == 0:
        return None
    return num / den


def linreg_2(X1, X2, y):
    """
    Fit y ~ c + a*X1 + b*X2 using normal equations.
    Returns (c, a, b, R2, n)
    """
    rows = []
    for x1, x2, t in zip(X1, X2, y):
        if x1 is None or x2 is None or t is None:
            continue
        rows.append((x1, x2, t))
    n = len(rows)
    if n < 3:
        return None, None, None, None, n
    sum1 = sum(x1 for x1,_,_ in rows)
    sum2 = sum(x2 for _,x2,_ in rows)
    sumy = sum(t for *_,t in rows)
    s11 = sum(x1*x1 for x1,_,_ in rows)
    s22 = sum(x2*x2 for _,x2,_ in rows)
    s12 = sum(x1*x2 for x1,x2,_ in rows)
    s1y = sum(x1*t for x1,_,t in rows)
    s2y = sum(x2*t for _,x2,t in rows)
    # Solve for [c,a,b] in normal equations: [n sum1 sum2; sum1 s11 s12; sum2 s12 s22] [c,a,b]^T = [sumy, s1y, s2y]
    import numpy as np
    A = np.array([[n,    sum1, sum2],
                  [sum1, s11,  s12 ],
                  [sum2, s12,  s22 ]], dtype=float)
    b = np.array([sumy, s1y, s2y], dtype=float)
    try:
        coef = np.linalg.solve(A, b)
    except Exception:
        return None, None, None, None, n
    c, a, b1 = coef.tolist()
    # R^2
    ybar = sumy / n
    ss_tot = sum((t - ybar)**2 for *_, t in rows)
    ss_res = sum((t - (c + a*x1 + b1*x2))**2 for x1,x2,t in rows)
    R2 = None if ss_tot == 0 else 1 - ss_res/ss_tot
    return c, a, b1, R2, n


def main():
    if not DET.exists() or not SPAN.exists():
        raise SystemExit('Missing inputs')
    # Load detailed compare
    with DET.open() as f:
        rd = csv.DictReader(f)
        det_rows = list(rd)
    # Load span features
    span = json.loads(SPAN.read_text())
    # Merge features
    merged = []
    for r in det_rows:
        m, s = r['model'], r['segment']
        sp = (span.get(m) or {}).get(s) or {}
        bytes_in = to_float(sp.get('bytes_in_mean'))
        bytes_out = to_float(sp.get('bytes_out_mean'))
        in_ms = to_float(sp.get('in_span_ms_mean'))
        out_ms = to_float(sp.get('out_span_ms_mean'))
        pre_delta = to_float(r.get('pre_delta'))
        merged.append({
            'model': m,
            'segment': s,
            'pre_delta': pre_delta,
            'MiB_in': None if bytes_in is None else bytes_in/(1024.0*1024.0),
            'MiB_out': None if bytes_out is None else bytes_out/(1024.0*1024.0),
            'in_span_ms': in_ms,
            'out_span_ms': out_ms,
        })

    # Write merged CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        fn = ['model','segment','pre_delta','MiB_in','MiB_out','in_span_ms','out_span_ms']
        wr = csv.DictWriter(f, fieldnames=fn)
        wr.writeheader(); wr.writerows(merged)

    # Compute correlations
    xs_in = [r['MiB_in'] for r in merged]
    xs_out = [r['MiB_out'] for r in merged]
    xs_inms = [r['in_span_ms'] for r in merged]
    xs_outms = [r['out_span_ms'] for r in merged]
    ys = [r['pre_delta'] for r in merged]
    r_in = pearson(xs_in, ys)
    r_out = pearson(xs_out, ys)
    r_inms = pearson(xs_inms, ys)
    r_outms = pearson(xs_outms, ys)

    # Simple linear regression with MiB_in, MiB_out
    try:
        import numpy  # noqa: F401
        c, a, b1, R2, n = linreg_2(xs_in, xs_out, ys)
    except Exception:
        c = a = b1 = R2 = None; n = 0

    # Per-model stats
    from collections import defaultdict
    grp = defaultdict(list)
    for r in merged:
        if r['pre_delta'] is not None:
            grp[r['model']].append(r['pre_delta'])
    per_model_stats = []
    for m, vals in grp.items():
        import statistics as stats
        mean = stats.mean(vals)
        stdev = stats.pstdev(vals) if len(vals)>1 else 0.0
        per_model_stats.append((m, len(vals), mean, stdev))
    per_model_stats.sort()

    # Output summary
    with OUT_TXT.open('w') as f:
        f.write('Envelope delta analysis (cur_pre - bak_pre)\n')
        f.write(f'Rows: {len(merged)}\n')
        import statistics as stats
        valid = [v for v in ys if v is not None]
        if valid:
            f.write(f'Overall mean delta (ms): {stats.mean(valid):.6f}\n')
            f.write(f'Overall stdev (ms): {stats.pstdev(valid):.6f}\n')
            f.write(f'Min/Max delta (ms): {min(valid):.6f} / {max(valid):.6f}\n')
        f.write('\nCorrelations (Pearson r):\n')
        f.write(f'  delta vs MiB_in:  {"na" if r_in is None else f"{r_in:.4f}"}\n')
        f.write(f'  delta vs MiB_out: {"na" if r_out is None else f"{r_out:.4f}"}\n')
        f.write(f'  delta vs in_ms:   {"na" if r_inms is None else f"{r_inms:.4f}"}\n')
        f.write(f'  delta vs out_ms:  {"na" if r_outms is None else f"{r_outms:.4f}"}\n')
        f.write('\nLinear fit: delta ~ c + a*MiB_in + b*MiB_out\n')
        f.write(f'  n={n}, c={c}, a={a}, b={b1}, R2={R2}\n')
        f.write('\nPer-model mean±stdev (ms):\n')
        for m, cnt, mean, sd in per_model_stats:
            f.write(f'  {m}: n={cnt}, mean={mean:.6f}, stdev={sd:.6f}\n')

    print(f'Saved: {OUT_TXT}\nSaved: {OUT_CSV}')


if __name__ == '__main__':
    main()
