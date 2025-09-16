#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
import csv

WORKDIR = Path('/home/10210/Desktop/OS')
COMBINED = WORKDIR / 'results' / 'models_local_batch_usbmon' / 'single' / 'combined_summary_span.json'
THEORY = WORKDIR / 'five_models' / 'baselines' / 'theory_io_seg.json'
OUT_CSV = WORKDIR / 'five_models' / 'results' / 'single_pure_invoke_times.csv'

def load_json(p: Path):
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def main():
    combined = load_json(COMBINED)
    theory = load_json(THEORY)

    # Build rows
    rows = []
    header = [
        'model','segment','type','count',
        'pure_ms_final','pure_ms_pre','pure_ms_post','adjustment_applied','theory_out_mibps_used'
    ]

    for model, segs in combined.items():
        tsegs = ((theory.get(model) or {}).get('segments')) or {}
        for seg_key, m in segs.items():
            # Determine off-chip
            tseg = tsegs.get(seg_key) or {}
            try:
                off_used = float(tseg.get('off_used_MiB', 0) or 0)
            except Exception:
                off_used = 0.0
            is_off = off_used > 0
            typ = 'off-chip' if is_off else 'on-chip'

            count = int(m.get('count', 0) or 0)
            pre = m.get('pure_span_ms_mean')
            post = m.get('pure_span_offchip_adjusted_ms_mean', pre)
            # Final choice: adjusted for off-chip, pre for on-chip
            final = post if is_off else pre
            theory_out = m.get('theory_out_mibps_used')
            adj_applied = (is_off and pre is not None and post is not None and abs(float(post) - float(pre)) > 1e-9)
            rows.append([
                model, seg_key, typ, count,
                None if final is None else float(final),
                None if pre is None else float(pre),
                None if post is None else float(post),
                1 if adj_applied else 0,
                None if theory_out is None else float(theory_out),
            ])

    # Write CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)

    print(f'Wrote {OUT_CSV} with {len(rows)} rows')

if __name__ == '__main__':
    main()
