#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fit on-chip IO span time vs bytes (per-invoke) across all models/segments.

Data source:
- results/models_local_batch_usbmon/single/<model>/seg*/active_analysis_strict.json
- five_models/baselines/theory_io_seg.json to classify on-chip (off_used_MiB == 0)

Method:
- For each on-chip segment, skip the first frame, take per_invoke records with bytes_in/bytes_out and in_span_sc_ms/out_span_sc_ms.
- Build pooled datasets: IN: (MiB_in, in_span_ms); OUT: (MiB_out, out_span_ms).
- Fit two simple models per direction:
  1) With intercept: time_ms = a + b * MiB
  2) Through origin: time_ms = b0 * MiB
- Report slope b (ms/MiB), intercept a (ms), R^2, and estimated bandwidth B_est = 1000 / b (MiB/s).
- Save a JSON summary and optional CSV of raw points.

Outputs:
- five_models/results/onchip_io_fit_summary.json
- five_models/results/onchip_io_points.csv (optional raw points)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

WORKDIR = Path('/home/10210/Desktop/OS')
RESULTS_BASE = WORKDIR / 'results' / 'models_local_batch_usbmon' / 'single'
THEORY_SEG = WORKDIR / 'five_models' / 'baselines' / 'theory_io_seg.json'
OUT_DIR = WORKDIR / 'five_models' / 'results'

MiB = 1024.0 * 1024.0

def _load_json(p: Path):
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def _is_onchip(model: str, seg: int, theory: dict) -> bool:
    try:
        t = ((theory.get(model) or {}).get('segments') or {}).get(f'seg{seg}') or {}
        return float(t.get('off_used_MiB', 0) or 0) <= 0.0
    except Exception:
        return True

def _collect_points() -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    theory = _load_json(THEORY_SEG) or {}
    in_pts: List[Tuple[float, float]] = []   # (MiB, ms)
    out_pts: List[Tuple[float, float]] = []  # (MiB, ms)
    if not RESULTS_BASE.exists():
        return in_pts, out_pts
    for model_dir in RESULTS_BASE.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for seg in range(1, 9):
            if not _is_onchip(model, seg, theory):
                continue
            ana = model_dir / f'seg{seg}' / 'active_analysis_strict.json'
            if not ana.exists():
                continue
            data = _load_json(ana)
            if not isinstance(data, dict) or 'per_invoke' not in data:
                continue
            per = data['per_invoke']
            if len(per) <= 1:
                continue
            per = per[1:]  # skip first frame
            for rec in per:
                try:
                    bin_b = int(rec.get('bytes_in', 0) or 0)
                    bout_b = int(rec.get('bytes_out', 0) or 0)
                    in_ms = float(rec.get('in_span_sc_ms') or 0.0)
                    out_ms = float(rec.get('out_span_sc_ms') or 0.0)
                except Exception:
                    continue
                if bin_b > 0 and in_ms > 0:
                    in_pts.append((bin_b / MiB, in_ms))
                if bout_b > 0 and out_ms > 0:
                    out_pts.append((bout_b / MiB, out_ms))
    return in_pts, out_pts

def _fit_with_intercept(x: List[float], y: List[float]) -> Dict[str, float]:
    # Ordinary least squares for y = a + b x
    n = len(x)
    if n == 0:
        return {'a': None, 'b': None, 'r2': None}
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    sxx = sum((xi - mean_x) ** 2 for xi in x)
    sxy = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    if sxx == 0:
        return {'a': mean_y, 'b': 0.0, 'r2': 0.0}
    b = sxy / sxx
    a = mean_y - b * mean_x
    ss_tot = sum((yi - mean_y) ** 2 for yi in y)
    ss_res = sum((yi - (a + b * xi)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return {'a': a, 'b': b, 'r2': r2}

def _fit_through_origin(x: List[float], y: List[float]) -> Dict[str, float]:
    # OLS for y = b x (no intercept)
    n = len(x)
    if n == 0:
        return {'b': None, 'r2': None}
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    if sxx == 0:
        return {'b': 0.0, 'r2': 0.0}
    b = sxy / sxx
    # R^2 for through-origin can be computed as 1 - SSE/SST0 where SST0 = sum(yi^2)
    ss_res = sum((yi - b * xi) ** 2 for xi, yi in zip(x, y))
    ss_tot0 = sum(yi * yi for yi in y)
    r2 = 1.0 - (ss_res / ss_tot0) if ss_tot0 > 0 else 1.0
    return {'b': b, 'r2': r2}

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    in_pts, out_pts = _collect_points()
    in_x = [p[0] for p in in_pts]
    in_y = [p[1] for p in in_pts]
    out_x = [p[0] for p in out_pts]
    out_y = [p[1] for p in out_pts]

    fit_in_int = _fit_with_intercept(in_x, in_y)
    fit_in_org = _fit_through_origin(in_x, in_y)
    fit_out_int = _fit_with_intercept(out_x, out_y)
    fit_out_org = _fit_through_origin(out_x, out_y)

    def est_bandwidth_ms_per_mib(b: float | None) -> Dict[str, float | None]:
        if b is None or b <= 0:
            return {'ms_per_mib': b, 'mib_per_s': None}
        return {'ms_per_mib': b, 'mib_per_s': 1000.0 / b}

    summary = {
        'in': {
            'with_intercept': {**fit_in_int, **est_bandwidth_ms_per_mib(fit_in_int.get('b'))},
            'through_origin': {**fit_in_org, **est_bandwidth_ms_per_mib(fit_in_org.get('b'))},
            'n_points': len(in_pts)
        },
        'out': {
            'with_intercept': {**fit_out_int, **est_bandwidth_ms_per_mib(fit_out_int.get('b'))},
            'through_origin': {**fit_out_org, **est_bandwidth_ms_per_mib(fit_out_org.get('b'))},
            'n_points': len(out_pts)
        }
    }

    with open(OUT_DIR / 'onchip_io_fit_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Also save raw points for optional plotting elsewhere
    try:
        import csv
        with open(OUT_DIR / 'onchip_io_points.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['direction', 'MiB', 'span_ms'])
            for m, t in in_pts:
                w.writerow(['in', f'{m:.6f}', f'{t:.6f}'])
            for m, t in out_pts:
                w.writerow(['out', f'{m:.6f}', f'{t:.6f}'])
    except Exception:
        pass

    # Optional function and distribution plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # IN: scatter + fit lines, residuals, speed histogram
        if in_pts:
            plt.figure(figsize=(6,4))
            plt.scatter(in_x, in_y, s=6, alpha=0.4, label='points')
            # Fit lines over range
            x_min, x_max = (min(in_x), max(in_x))
            xs = [x_min, x_max]
            if fit_in_int.get('b') is not None and fit_in_int.get('a') is not None:
                ys_int = [fit_in_int['a'] + fit_in_int['b'] * v for v in xs]
                plt.plot(xs, ys_int, 'r-', label=f"int: a={fit_in_int['a']:.2f}, b={fit_in_int['b']:.2f}, R²={fit_in_int['r2']:.3f}")
            if fit_in_org.get('b') is not None:
                ys_org = [fit_in_org['b'] * v for v in xs]
                plt.plot(xs, ys_org, 'g--', label=f"orig: b={fit_in_org['b']:.2f}, R²={fit_in_org['r2']:.3f}")
            plt.xlabel('MiB')
            plt.ylabel('in span time (ms)')
            plt.title('On-chip IN IO: span time vs MiB')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(OUT_DIR / 'onchip_io_fit_in.png', dpi=150)
            plt.close()

            # Residuals w.r.t through-origin fit
            if fit_in_org.get('b') is not None:
                resid = [yi - fit_in_org['b'] * xi for xi, yi in zip(in_x, in_y)]
                plt.figure(figsize=(6,3.5))
                plt.scatter(in_x, resid, s=6, alpha=0.4)
                plt.axhline(0, color='k', lw=1)
                plt.xlabel('MiB')
                plt.ylabel('residual (ms)')
                plt.title('IN residuals (to origin-fit)')
                plt.tight_layout()
                plt.savefig(OUT_DIR / 'onchip_io_in_residuals.png', dpi=150)
                plt.close()

                # Residual histogram
                plt.figure(figsize=(6,3.5))
                plt.hist(resid, bins=40, alpha=0.8)
                plt.xlabel('residual (ms)')
                plt.ylabel('count')
                plt.title('IN residual distribution')
                plt.tight_layout()
                plt.savefig(OUT_DIR / 'onchip_io_in_residual_hist.png', dpi=150)
                plt.close()

            # Speed histogram (MiB/s) from points
            speeds = [1000.0 * (xi/yi) for xi, yi in zip(in_x, in_y) if yi > 0]
            if speeds:
                plt.figure(figsize=(6,3.5))
                plt.hist(speeds, bins=40, alpha=0.8)
                plt.xlabel('effective IN speed (MiB/s)')
                plt.ylabel('count')
                plt.title('IN effective speed distribution')
                plt.tight_layout()
                plt.savefig(OUT_DIR / 'onchip_io_in_speed_hist.png', dpi=150)
                plt.close()

        # OUT: scatter + fit lines, residuals, speed histogram
        if out_pts:
            plt.figure(figsize=(6,4))
            plt.scatter(out_x, out_y, s=6, alpha=0.4, label='points')
            x_min, x_max = (min(out_x), max(out_x))
            xs = [x_min, x_max]
            if fit_out_int.get('b') is not None and fit_out_int.get('a') is not None:
                ys_int = [fit_out_int['a'] + fit_out_int['b'] * v for v in xs]
                plt.plot(xs, ys_int, 'r-', label=f"int: a={fit_out_int['a']:.2f}, b={fit_out_int['b']:.2f}, R²={fit_out_int['r2']:.3f}")
            if fit_out_org.get('b') is not None:
                ys_org = [fit_out_org['b'] * v for v in xs]
                plt.plot(xs, ys_org, 'g--', label=f"orig: b={fit_out_org['b']:.2f}, R²={fit_out_org['r2']:.3f}")
            plt.xlabel('MiB')
            plt.ylabel('out span time (ms)')
            plt.title('On-chip OUT IO: span time vs MiB')
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(OUT_DIR / 'onchip_io_fit_out.png', dpi=150)
            plt.close()

            if fit_out_org.get('b') is not None:
                resid = [yi - fit_out_org['b'] * xi for xi, yi in zip(out_x, out_y)]
                plt.figure(figsize=(6,3.5))
                plt.scatter(out_x, resid, s=6, alpha=0.4)
                plt.axhline(0, color='k', lw=1)
                plt.xlabel('MiB')
                plt.ylabel('residual (ms)')
                plt.title('OUT residuals (to origin-fit)')
                plt.tight_layout()
                plt.savefig(OUT_DIR / 'onchip_io_out_residuals.png', dpi=150)
                plt.close()

                plt.figure(figsize=(6,3.5))
                plt.hist(resid, bins=40, alpha=0.8)
                plt.xlabel('residual (ms)')
                plt.ylabel('count')
                plt.title('OUT residual distribution')
                plt.tight_layout()
                plt.savefig(OUT_DIR / 'onchip_io_out_residual_hist.png', dpi=150)
                plt.close()

            speeds = [1000.0 * (xi/yi) for xi, yi in zip(out_x, out_y) if yi > 0]
            if speeds:
                plt.figure(figsize=(6,3.5))
                plt.hist(speeds, bins=40, alpha=0.8)
                plt.xlabel('effective OUT speed (MiB/s)')
                plt.ylabel('count')
                plt.title('OUT effective speed distribution')
                plt.tight_layout()
                plt.savefig(OUT_DIR / 'onchip_io_out_speed_hist.png', dpi=150)
                plt.close()
    except Exception:
        # plotting is optional; ignore failures
        pass

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
