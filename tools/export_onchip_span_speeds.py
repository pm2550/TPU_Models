#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export on-chip per-segment span in/out speeds to a CSV for quick inspection.

Inputs:
- results/models_local_batch_usbmon/single/onchip_summary_span.json

Outputs:
- results/onchip_span_speeds.csv with columns:
  model,segment,count,in_span_ms_mean,out_span_ms_mean,
  in_speed_span_mibps_mean,out_speed_span_mibps_mean,
  bytes_in_mean,bytes_out_mean
"""

import csv
import json
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
IN_JSON = BASE / 'results/models_local_batch_usbmon/single/onchip_summary_span.json'
OUT_CSV = BASE / 'results/onchip_span_speeds.csv'


def to_float(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def compute_speed_mibps(bytes_val, span_ms):
    b = to_float(bytes_val)
    ms = to_float(span_ms)
    if b is None or ms is None or ms <= 0:
        return None
    mib = b / (1024.0 * 1024.0)
    sec = ms / 1000.0
    if sec <= 0:
        return None
    return mib / sec


def main():
    if not IN_JSON.exists():
        raise SystemExit(f'Missing input JSON: {IN_JSON}')
    data = json.loads(IN_JSON.read_text())

    rows = []
    header = [
        'model', 'segment', 'count',
        'in_span_ms_mean', 'out_span_ms_mean',
        'in_speed_span_mibps_mean', 'out_speed_span_mibps_mean',
        'bytes_in_mean', 'bytes_out_mean',
    ]

    for model, segs in (data or {}).items():
        for seg, rec in (segs or {}).items():
            count = int(rec.get('count') or 0)
            in_span_ms = to_float(rec.get('in_span_ms_mean'))
            out_span_ms = to_float(rec.get('out_span_ms_mean'))
            in_speed = to_float(rec.get('in_speed_span_mibps_mean'))
            out_speed = to_float(rec.get('out_speed_span_mibps_mean'))
            bytes_in = to_float(rec.get('bytes_in_mean'))
            bytes_out = to_float(rec.get('bytes_out_mean'))

            # If speeds are missing, compute from bytes and span
            if in_speed is None:
                in_speed = compute_speed_mibps(bytes_in, in_span_ms)
            if out_speed is None:
                out_speed = compute_speed_mibps(bytes_out, out_span_ms)

            rows.append([
                model, seg, count,
                None if in_span_ms is None else f'{in_span_ms}',
                None if out_span_ms is None else f'{out_span_ms}',
                None if in_speed is None else f'{in_speed}',
                None if out_speed is None else f'{out_speed}',
                None if bytes_in is None else f'{bytes_in}',
                None if bytes_out is None else f'{bytes_out}',
            ])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for r in rows:
            wr.writerow(r)

    print(f'Saved: {OUT_CSV} ({len(rows)} rows)')


if __name__ == '__main__':
    main()
