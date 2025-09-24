#!/usr/bin/env python3
"""
Run compute_theory for each model with a specific H2D (Cin) speed and export a
per-model CSV that includes LB, UB (slow D2H), and a scaled-UB using a factor f
chosen to cover >=99% of measured invoke means in single-mode summaries.

Details:
- Th_ms fixed at 1.2 ms; host slope kappa=0.
- LB uses fast D2H (B_OUT), UB uses slow D2H (B_OUT2). H2D speed fixed per model.
- t_rem (offchip) per group:
  * LB: max(off_ms - Ce_ms, 0)
  * UB: off_ms
  * UB_f: max(f*off_ms - Ce_ms, 0)
- Cin column order is size, speed, then result (ms).

Outputs: five CSVs under five_models/results/theory_sweeps/<model>.csv
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from statistics import median

BASE = Path('/home/10210/Desktop/OS')
TOOLS = BASE/'tools'
THEORY_CSV = BASE/'five_models/results/theory_chain_times.csv'
COMB_SUM = BASE/'results/models_local_batch_usbmon/single/combined_summary.json'
COMBOS_JSON = BASE/'five_models/baselines/theory_io_combos.json'
SEGS_JSON = BASE/'five_models/baselines/theory_io_seg.json'
COMBO_CYCLE_CSV = BASE/'five_models/results/combo_cycle_times.csv'
OUT_DIR = BASE/'five_models/results/theory_sweeps'

# Model-specific H2D speeds (MiB/s)
MODEL_H2D = {
    'resnet101_8seg_uniform_local': 344.0,
    'resnet50_8seg_uniform_local': 339.5,
    'inceptionv3_8seg_uniform_local': 338.6,
    'xception_8seg_uniform_local': 287.0,
    'densenet201_8seg_uniform_local': 325.0,
}

# D2H speeds: fast vs slow (MiB/s). Use compute_theory defaults unless overridden via env.
DEFAULT_B_OUT = 87.0
DEFAULT_B_OUT2 = 35.0


def call_compute(B_IN: float, B_OUT: float, B_OUT2: float) -> None:
    """Invoke compute_theory with env overrides.
    - Fixed host Th=1.2ms and kappa=0.
    - Allow EXTRA_CIN_BYTES passthrough if set (previous behavior).
    """
    os.environ['USE_CODE_DEFAULTS'] = '0'
    os.environ['B_IN'] = str(B_IN)
    os.environ['B_OUT'] = str(B_OUT)
    os.environ['B_OUT2'] = str(B_OUT2)
    os.environ['HOST_C_MS'] = '1.2'
    os.environ['KAPPA_MS_PER_MS'] = '0.0'
    # import and run
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location('compute_theory_chain_times', str(TOOLS/'compute_theory_chain_times.py'))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.compute_chain_times()


def load_group_io_bytes(model: str, K: int, group: str) -> tuple[float, float]:
    """Return (base_input_bytes, base_output_bytes) for a model/group/K."""
    # K=8 → segments JSON
    if K == 8:
        J = json.loads(SEGS_JSON.read_text())
        gd = ((J.get(model) or {}).get('segments') or {}).get(group) or {}
        return float(gd.get('base_input_bytes') or 0.0), float(gd.get('base_output_bytes') or 0.0)
    # else from combos
    J = json.loads(COMBOS_JSON.read_text())
    Kkey = f'K{K}'
    gd = (((J.get(model) or {}).get('combos') or {}).get(Kkey) or {}).get(group) or {}
    return float(gd.get('base_input_bytes') or 0.0), float(gd.get('base_output_bytes') or 0.0)


def load_weight_split_mib(model: str, K: int, group: str) -> tuple[float, float]:
    """Return (segment_model_MiB, off_used_MiB) for model/K/group from baselines JSON.
    Falls back to (0.0, 0.0) when unavailable.
    """
    seg_mib = 0.0
    off_mib = 0.0
    try:
        if K == 8:
            J = json.loads(SEGS_JSON.read_text())
            gd = (((J.get(model) or {}).get('segments') or {}).get(group)) or {}
            seg_mib = float(gd.get('segment_model_MiB') or 0.0)
            off_mib = float(gd.get('off_used_MiB') or gd.get('off_chip_MiB') or 0.0)
        else:
            J = json.loads(COMBOS_JSON.read_text())
            Kkey = f'K{K}'
            gd = ((((J.get(model) or {}).get('combos') or {}).get(Kkey) or {}).get(group)) or {}
            seg_mib = float(gd.get('segment_model_MiB') or 0.0)
            off_mib = float(gd.get('off_used_MiB') or gd.get('off_chip_MiB') or 0.0)
    except Exception:
        pass
    return seg_mib, off_mib


def parse_off_mib(notes: str, group: str) -> float:
    if not notes:
        return 0.0
    # find pattern '<group>:off_used_MiB=val'
    target = f"{group}:off_used_MiB="
    for part in (notes or '').split(';'):
        part = part.strip()
        if part.startswith(target):
            try:
                return float(part.split('=')[1])
            except Exception:
                return 0.0
    return 0.0


def mib_to_ms(mib: float, mibps: float) -> float:
    return (mib / (mibps or 1.0)) * 1000.0


def bytes_to_ms(nbytes: float, mibps: float) -> float:
    return (nbytes / (1024.0 * 1024.0) / (mibps or 1.0)) * 1000.0


def choose_factor_99(model: str, enriched_rows: list[dict], th_ms: float = 1.2,
                     coverage_quantile: float = 0.99) -> tuple[float, float, int, int, int]:
    """Pick minimal factor f (single per-model) so that across ALL cycles
    (K=2..8, all rounds), UB_f(K) covers at least the given coverage_quantile
    (default 99%) of measured total_cycle_ms.

    Returns (f, coverage_rate, used_cycles, total_cycles, outlier_dropped).

    Steps:
      1) Load all cycles (model,K,total_ms) from combo_cycle_times.csv.
      2) Robust outlier filtering on total_ms using Median/MAD fallback to IQR.
      3) For each remaining cycle, compute minimal f_req in [0,1] s.t.
         total_for_k(f_req) >= total_ms (expected UB with partial overlap).
      4) f = coverage_quantile percentile of f_req list (i.e., just enough to cover that fraction).
      5) Compute achieved coverage rate with this f across the filtered cycles.
    """
    # Build per-K group lists from enriched rows
    byK: dict[int, list[dict]] = {}
    for e in enriched_rows:
        byK.setdefault(int(e['K']), []).append(e)

    # Load measured cycle times for this model
    if not COMBO_CYCLE_CSV.exists():
        return 1.0
    meas: list[tuple[int, float]] = []  # (K, total_ms)
    with COMBO_CYCLE_CSV.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            if r.get('model') != model:
                continue
            try:
                K = int(r.get('K') or 0)
                ms = float(r.get('total_cycle_ms') or 0.0)
            except Exception:
                continue
            if K in byK and ms > 0:
                meas.append((K, ms))
    total_cycles = len(meas)
    if not meas:
        return 1.0, 1.0, 0, 0, 0

    # Robust outlier filtering on total_ms
    vals = [ms for _, ms in meas]
    med = median(vals)
    abs_dev = [abs(x - med) for x in vals]
    mad = median(abs_dev)
    keep_idx = set(range(len(vals)))
    if mad > 0:
        # Robust z-score threshold
        for i, x in enumerate(vals):
            z = 0.6745 * (x - med) / mad
            if abs(z) > 3.5:
                if i in keep_idx:
                    keep_idx.remove(i)
    else:
        # Fallback to IQR
        s = sorted(vals)
        q1 = s[int(0.25 * (len(s)-1))]
        q3 = s[int(0.75 * (len(s)-1))]
        iqr = q3 - q1
        if iqr > 0:
            lo = q1 - 3.0 * iqr
            hi = q3 + 3.0 * iqr
            for i, x in enumerate(vals):
                if x < lo or x > hi:
                    if i in keep_idx:
                        keep_idx.remove(i)
    meas_filt = [meas[i] for i in sorted(list(keep_idx))]
    dropped = total_cycles - len(meas_filt)
    if not meas_filt:
        return 1.0, 1.0, 0, total_cycles, dropped

    def total_for_k(f: float, rows: list[dict]) -> float:
        # Sum per-group for "expected" bound using UB Cout (slow)
        # Only fraction f can overlap Ce; the rest (1-f) is extra outside overlap.
        # total = Cin + Cout_slow + Ce + [max(f*off - Ce, 0) + (1-f)*off] + t_warm + Th
        const = 0.0
        sum_rem = 0.0
        n = 0
        for e in rows:
            const += float(e['Cin_ms']) + float(e['Cout_ms_ub']) + float(e['t_warm_ms']) + float(e['Ce_ms'])
            ce = float(e['Ce_ms'])
            off_ms = float(e['t_remaining_ms'])
            sum_rem += max(f * off_ms - ce, 0.0) + (1.0 - f) * off_ms
            n += 1
        return const + sum_rem + n * th_ms

    # Compute required f per measured cycle (global across K)
    f_reqs: list[float] = []
    for K, target in meas_filt:
        rowsK = byK.get(K)
        if not rowsK:
            continue
        # If no off_ms, f has no effect
        if sum(float(e['t_remaining_ms']) for e in rowsK) <= 1e-12:
            f_reqs.append(0.0 if total_for_k(0.0, rowsK) >= target else 1.0)
            continue
        if total_for_k(0.0, rowsK) >= target:
            f_reqs.append(0.0)
            continue
        # Binary search f in [0,1]
        f_lo, f_hi = 0.0, 1.0
        if total_for_k(f_hi, rowsK) < target:
            f_reqs.append(1.0)
            continue
        for _ in range(40):
            f_mid = 0.5 * (f_lo + f_hi)
            if total_for_k(f_mid, rowsK) >= target:
                f_hi = f_mid
            else:
                f_lo = f_mid
        f_reqs.append(max(0.0, f_hi))

    if not f_reqs:
        return 1.0, 1.0, len(meas_filt), total_cycles, dropped
    # coverage_quantile percentile over all cycles' f requirements
    q = sorted(f_reqs)
    idx = min(len(q) - 1, int(math.ceil(coverage_quantile * len(q)) - 1))
    f_model = min(1.0, max(0.0, q[idx]))
    # Compute achieved coverage with this f
    covered = 0
    for K, target in meas_filt:
        rowsK = byK.get(K)
        if not rowsK:
            continue
        if total_for_k(f_model, rowsK) >= target:
            covered += 1
    coverage_rate = covered / float(len(meas_filt)) if meas_filt else 1.0
    return f_model, coverage_rate, len(meas_filt), total_cycles, dropped


def build_per_model_csv(model: str, b_in: float, b_out_hi: float, b_out_lo: float) -> None:
    # Run compute to refresh theory CSV with requested speeds and host params
    call_compute(b_in, b_out_hi, b_out_lo)
    # Load rows for model
    with THEORY_CSV.open() as f:
        rd = csv.DictReader(f)
        rows = [r for r in rd if r.get('model') == model and r.get('group_index') != 'TOTAL']
    # Enrich rows
    enriched = []
    for r in rows:
        K = int(r.get('K') or 0)
        g = r.get('group_name')
        d_in, d_out = load_group_io_bytes(model, K, g)
        cin_ms = bytes_to_ms(d_in, b_in)
        cout_ms_hi = bytes_to_ms(d_out, b_out_hi)
        cout_ms_lo = bytes_to_ms(d_out, b_out_lo)
        ce = float(r.get('Ce_ms') or 0.0)
        seg_total_mib, off_mib = load_weight_split_mib(model, K, g)
        off_ms = mib_to_ms(off_mib, b_in)
        # LB / UB hosted totals (Th fixed 1.2)
        th = 1.2
        warm_mib = max(seg_total_mib - off_mib, 0.0)
        t_warm = mib_to_ms(warm_mib, b_in)
        t_over = max(off_ms - ce, 0.0)
        lb_const = cin_ms + cout_ms_hi + ce + t_warm
        ub_const = cin_ms + cout_ms_lo + ce + t_warm
        lb = lb_const + t_over
        ub = ub_const + off_ms
        # Warm_MiB already from model def (not back-calculated)
        enriched.append({
            'model': model,
            'K': K,
            'group_name': g,
            'group_segs': r.get('group_segs'),
            'Cin_bytes': int(d_in),
            'B_IN_mibps': b_in,
            'Cin_ms': round(cin_ms, 3),
            'Cout_bytes': int(d_out),
            'B_OUT_mibps_lb': b_out_hi,
            'Cout_ms_lb': round(cout_ms_hi, 3),
            'B_OUT_mibps_ub': b_out_lo,
            'Cout_ms_ub': round(cout_ms_lo, 3),
            'Ce_ms': round(ce, 3),
            'Remaining_MiB': round(off_mib, 6),
            't_remaining_ms': round(off_ms, 3),
            't_overlapped_remaining_ms': round(t_over, 3),
            'Warm_MiB': round(warm_mib, 6),
            't_warm_ms': round(t_warm, 3),
            'Th_ms': th,
            'LB_ms_total_hosted': round(lb + th, 3),
            'UB_ms_total_hosted': round(ub + th, 3),
        })
    # Choose factor using combo_cycle_times coverage at 99%
    f99, cov_rate, used_n, total_n, dropped = choose_factor_99(model, enriched, th_ms=1.2)
    for e in enriched:
        ce = float(e['Ce_ms'])
        off_ms = float(e['t_remaining_ms'])
        cin_ms = float(e['Cin_ms'])
        cout_ms_lo = float(e['Cout_ms_ub'])
        t_warm = float(e['t_warm_ms'])
        th = float(e['Th_ms'])
        # expected total uses UB Cout (slow), with only fraction f overlappable; add (1-f)*off_ms outside
        ubf_total = cin_ms + cout_ms_lo + ce + max(f99 * off_ms - ce, 0.0) + (1.0 - f99) * off_ms + t_warm + th
        e['f_expected_gt_99'] = round(f99, 6)
        e['UB_expected_ms_total_hosted'] = round(ubf_total, 3)
        e['expected_coverage_rate'] = round(cov_rate, 4)
        e['cycles_used'] = used_n
        e['cycles_total'] = total_n
        e['cycles_outliers_dropped'] = dropped
    # Write per-model CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR/f'{model}.csv'
    cols = [
        'model','K','group_name','group_segs',
        # Cin breakdown (size, speed, result)
        'Cin_bytes','B_IN_mibps','Cin_ms',
        # Cout breakdown (LB/UB use same size with different speeds)
        'Cout_bytes','B_OUT_mibps_lb','Cout_ms_lb','B_OUT_mibps_ub','Cout_ms_ub',
        # Compute and remaining
        'Ce_ms','Remaining_MiB','t_remaining_ms','t_overlapped_remaining_ms','Warm_MiB','t_warm_ms','Th_ms',
        # Final totals
        'LB_ms_total_hosted','UB_ms_total_hosted',
        # Expected upper bound (with f)
        'f_expected_gt_99','UB_expected_ms_total_hosted','expected_coverage_rate','cycles_used','cycles_total','cycles_outliers_dropped'
    ]
    with out_path.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader(); wr.writerows(enriched)
    print('Wrote', out_path)


def main() -> int:
    b_out = float(os.environ.get('B_OUT', DEFAULT_B_OUT))
    b_out2 = float(os.environ.get('B_OUT2', DEFAULT_B_OUT2))
    # Generate per-model CSVs
    for model, b_in in MODEL_H2D.items():
        build_per_model_csv(model, b_in, b_out, b_out2)
    # Combine into a single CSV
    all_rows = []
    out_cols = None
    for model in MODEL_H2D.keys():
        p = OUT_DIR / f"{model}.csv"
        if not p.exists():
            continue
        with p.open() as f:
            rd = csv.DictReader(f)
            if out_cols is None:
                out_cols = rd.fieldnames
            for r in rd:
                all_rows.append(r)
    if all_rows and out_cols:
        combined = OUT_DIR / 'all_models.csv'
        with combined.open('w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=out_cols)
            wr.writeheader(); wr.writerows(all_rows)
        print('Wrote', combined)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
