#!/usr/bin/env python3
"""
Scan results/models_local_batch_usbmon/single/*_8seg_uniform_local/seg*/active_analysis_strict.json
and compute warm-only mean/stdev of pure_ms per segment, then update
five_models/results/single_pure_invoke_times.csv by inserting/updating rows.

Skips off-chip segments using five_models/baselines/theory_io_seg.json.
Optionally restrict to specific models via ONLY_MODELS env (comma-separated).
"""
import csv
import json
import os
from pathlib import Path
import statistics

ROOT = Path('/home/10210/Desktop/OS')
RESULTS_BASE = ROOT / 'results' / 'models_local_batch_usbmon' / 'single'
PURE_CSV = ROOT / 'five_models' / 'results' / 'single_pure_invoke_times.csv'
SEG_META_CSV = ROOT / 'five_models' / 'results' / 'single_segment_metrics.csv'
THEORY_SEG_JSON = ROOT / 'five_models' / 'baselines' / 'theory_io_seg.json'

def load_cut_map():
    cut = {}
    try:
        with SEG_META_CSV.open(newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                model = row['model']
                seg = row['segment']
                cut[(model, seg)] = (row.get('cut_start_layer', ''), row.get('cut_end_layer', ''))
    except FileNotFoundError:
        pass
    return cut


def load_existing_rows():
    rows = []
    if PURE_CSV.exists():
        with PURE_CSV.open(newline='', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
    return rows


def write_rows(rows):
    fieldnames = ['model', 'segment', 'cut_start_layer', 'cut_end_layer', 'pure_invoke_mean_ms', 'pure_invoke_stdev_ms', 'count']
    with PURE_CSV.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_pure_stats(an_file: Path):
    try:
        data = json.loads(an_file.read_text(encoding='utf-8'))
        pv = data.get('per_invoke', [])
        if not pv:
            return None
        pure = []
        for rec in pv:
            v = rec.get('pure_ms')
            if v is None:
                v = rec.get('pure_compute_ms')
            if v is not None:
                pure.append(float(v))
        if not pure:
            return None
        vals = pure[1:] if len(pure) > 1 else pure  # drop warm
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return mean, stdev, len(vals)
    except Exception:
        return None


def main():
    # Filters
    only_models_env = os.environ.get('ONLY_MODELS')
    only_models = None
    if only_models_env:
        only_models = {m.strip() for m in only_models_env.split(',') if m.strip()}

    # Load off-chip map
    offchip = {}
    try:
        seg_def = json.loads(THEORY_SEG_JSON.read_text(encoding='utf-8'))
        for m, info in seg_def.items():
            segs = info.get('segments', {})
            for k, v in segs.items():
                offchip[(m, k)] = float(v.get('off_chip_MiB', 0.0) or 0.0)
    except Exception:
        offchip = {}

    cut_map = load_cut_map()
    existing = load_existing_rows()
    idx = {(r['model'], r['segment']): i for i, r in enumerate(existing)}

    # Always MERGE into existing rows. When ONLY_MODELS is set,
    # we still update/append only those models, but keep all others.
    found_updates = 0

    for model_dir in RESULTS_BASE.iterdir():
        if not model_dir.is_dir():
            continue
        name = model_dir.name
        if not name.endswith('_8seg_uniform_local'):
            continue
        if only_models is not None and name not in only_models:
            continue
        for seg_dir in sorted(model_dir.glob('seg*')):
            if not seg_dir.is_dir():
                continue
            an_file = seg_dir / 'active_analysis_strict.json'
            if not an_file.exists():
                continue
            seg = seg_dir.name
            if offchip.get((name, seg), 0.0) > 0.0:
                continue
            stats = compute_pure_stats(an_file)
            if not stats:
                continue
            mean, stdev, count = stats
            start, end = cut_map.get((name, seg), ('', ''))
            row = {
                'model': name,
                'segment': seg,
                'cut_start_layer': start,
                'cut_end_layer': end,
                'pure_invoke_mean_ms': f"{mean:.3f}",
                'pure_invoke_stdev_ms': f"{stdev:.3f}",
                'count': str(count),
            }
            if (name, seg) in idx:
                existing[idx[(name, seg)]] = row
            else:
                existing.append(row)
                idx[(name, seg)] = len(existing) - 1
            found_updates += 1

    out_rows = existing
    if found_updates:
        def seg_key(seg_name: str):
            try:
                return int(seg_name.replace('seg', ''))
            except Exception:
                return 0
        out_rows.sort(key=lambda r: (r['model'], seg_key(r['segment'])))
        write_rows(out_rows)
        print(f"Updated {found_updates} rows in {PURE_CSV}")
    else:
        print("No updates found.")


if __name__ == '__main__':
    main()
