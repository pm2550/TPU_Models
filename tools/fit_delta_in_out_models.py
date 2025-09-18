#!/usr/bin/env python3
import csv
from pathlib import Path
import math

BASE = Path(__file__).resolve().parent.parent
MERGED = BASE / 'results/envelope_delta_merged.csv'
OUT = BASE / 'results/delta_in_out_fit.txt'


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def fit_linear(y, X):
    # X: list of lists (each row features, with leading 1 for intercept if desired)
    # Solve via normal equation: beta = (X^T X)^{-1} X^T y (2x2 or 3x3 small cases)
    import numpy as np
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat)**2).sum())
    ss_tot = float(((y - y.mean())**2).sum())
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float('nan')
    return beta.tolist(), r2


def main():
    rows = read_rows(MERGED)
    y = [float(r['pre_delta']) for r in rows]

    # Features
    in_mib = [float(r['MiB_in']) for r in rows]
    out_mib = [float(r['MiB_out']) for r in rows]
    in_ms = [float(r['in_span_ms']) for r in rows]
    out_ms = [float(r['out_span_ms']) for r in rows]

    X_mib_in = [[1.0, xi] for xi in in_mib]
    X_mib_inout = [[1.0, xi, xo] for xi, xo in zip(in_mib, out_mib)]
    X_ms_in = [[1.0, xi] for xi in in_ms]
    X_ms_inout = [[1.0, xi, xo] for xi, xo in zip(in_ms, out_ms)]
    # New: combined in_span_ms + MiB_in
    X_ms_in__mib_in = [[1.0, x_ms, x_mib] for x_ms, x_mib in zip(in_ms, in_mib)]

    b1, r2_1 = fit_linear(y, X_mib_in)
    b2, r2_2 = fit_linear(y, X_mib_inout)
    b3, r2_3 = fit_linear(y, X_ms_in)
    b4, r2_4 = fit_linear(y, X_ms_inout)
    b5, r2_5 = fit_linear(y, X_ms_in__mib_in)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w') as f:
        f.write('Linear fits for delta (sizes and spans)\n')
        f.write(f'Rows: {len(rows)}\n\n')
        f.write('delta ~ c + a*MiB_in\n')
        f.write(f'  beta={b1}, R2={r2_1:.4f}\n')
        f.write('delta ~ c + a*MiB_in + b*MiB_out\n')
        f.write(f'  beta={b2}, R2={r2_2:.4f}\n\n')
        f.write('delta ~ c + k*in_span_ms\n')
        f.write(f'  beta={b3}, R2={r2_3:.4f}\n')
        f.write('delta ~ c + k*in_span_ms + m*out_span_ms\n')
        f.write(f'  beta={b4}, R2={r2_4:.4f}\n\n')
        f.write('delta ~ c + k*in_span_ms + a*MiB_in\n')
        f.write(f'  beta={b5}, R2={r2_5:.4f}\n')

if __name__ == '__main__':
    main()
