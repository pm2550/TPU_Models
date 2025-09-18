#!/usr/bin/env python3
import csv
from pathlib import Path
import math

BASE = Path(__file__).resolve().parent.parent
MERGED = BASE / 'results/envelope_delta_merged.csv'
OUT = BASE / 'results/delta_fit_error_stats.txt'

# Coefficients from results/delta_in_out_fit.txt
C_IN = 0.5527747236073199
K_IN = 0.2992134815732149

C_INOUT = 0.5410365185578128
K_INOUT = 0.2928623213396811
M_OUT = 0.018170333443736905

def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def pct(vs, p):
    if not vs:
        return float('nan')
    vs = sorted(vs)
    k = (len(vs)-1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vs[int(k)]
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def main():
    rows = read_rows(MERGED)
    y = []
    yhat_in = []
    yhat_inout = []
    for r in rows:
        delta = float(r['pre_delta'])
        in_ms = float(r['in_span_ms'])
        out_ms = float(r['out_span_ms'])
        y.append(delta)
        yhat_in.append(C_IN + K_IN * in_ms)
        yhat_inout.append(C_INOUT + K_INOUT * in_ms + M_OUT * out_ms)

    def stats(y, yhat):
        errs = [yh - yt for yh, yt in zip(yhat, y)]
        abs_errs = [abs(e) for e in errs]
        mae = sum(abs_errs)/len(abs_errs)
        rmse = math.sqrt(sum(e*e for e in errs)/len(errs))
        max_abs = max(abs_errs)
        p90 = pct(abs_errs, 0.90)
        p95 = pct(abs_errs, 0.95)
        return mae, rmse, p90, p95, max_abs

    mae1, rmse1, p90_1, p95_1, max1 = stats(y, yhat_in)
    mae2, rmse2, p90_2, p95_2, max2 = stats(y, yhat_inout)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w') as f:
        f.write('Error stats for delta fit (ms)\n')
        f.write(f'Rows: {len(y)}\n\n')
        f.write('Model: delta ~ c + k*in_span_ms\n')
        f.write(f'  MAE={mae1:.4f}, RMSE={rmse1:.4f}, p90_abs_err={p90_1:.4f}, p95_abs_err={p95_1:.4f}, max_abs_err={max1:.4f}\n\n')
        f.write('Model: delta ~ c + k*in_span_ms + m*out_span_ms\n')
        f.write(f'  MAE={mae2:.4f}, RMSE={rmse2:.4f}, p90_abs_err={p90_2:.4f}, p95_abs_err={p95_2:.4f}, max_abs_err={max2:.4f}\n')

    print(f'Saved: {OUT}')

if __name__ == '__main__':
    main()
