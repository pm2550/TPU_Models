#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarize on-chip per-model total in/out bytes and span times, and compute
average speeds as total MiB / total seconds.

Input:
- results/models_local_batch_usbmon/single/onchip_summary_span.json

Output:
- results/onchip_total_speeds.csv with columns:
  model, seg_count, total_in_span_ms, total_out_span_ms,
  total_bytes_in, total_bytes_out, avg_in_speed_mibps, avg_out_speed_mibps
"""

import csv
import json
from pathlib import Path

BASE = Path('/home/10210/Desktop/OS')
IN_JSON = BASE / 'results/models_local_batch_usbmon/single/onchip_summary_span.json'
OUT_CSV = BASE / 'results/onchip_total_speeds.csv'


def to_float(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None


def main():
    if not IN_JSON.exists():
        raise SystemExit(f'Missing input JSON: {IN_JSON}')
    data = json.loads(IN_JSON.read_text())

    rows = []
    per_model = []  # keep per-model aggregates for exclusion scenarios
    header = [
        'model', 'seg_count',
        'total_in_span_ms', 'total_out_span_ms',
        'total_bytes_in', 'total_bytes_out',
        'avg_in_speed_mibps', 'avg_out_speed_mibps',
    ]

    grand_in_ms = 0.0
    grand_out_ms = 0.0
    grand_bytes_in = 0.0
    grand_bytes_out = 0.0
    grand_seg_count = 0

    for model, segs in (data or {}).items():
        total_in_span_ms = 0.0
        total_out_span_ms = 0.0
        total_bytes_in = 0.0
        total_bytes_out = 0.0
        seg_count = 0
        for seg, rec in (segs or {}).items():
            in_ms = to_float(rec.get('in_span_ms_mean'))
            out_ms = to_float(rec.get('out_span_ms_mean'))
            bin_bytes = to_float(rec.get('bytes_in_mean'))
            bout_bytes = to_float(rec.get('bytes_out_mean'))
            # Sum present values; treat None as 0
            if in_ms is not None:
                total_in_span_ms += in_ms
            if out_ms is not None:
                total_out_span_ms += out_ms
            if bin_bytes is not None:
                total_bytes_in += bin_bytes
            if bout_bytes is not None:
                total_bytes_out += bout_bytes
            seg_count += 1

        # Compute average speeds (MiB/s)
        avg_in = None
        if total_in_span_ms > 0 and total_bytes_in > 0:
            avg_in = (total_bytes_in / (1024.0 * 1024.0)) / (total_in_span_ms / 1000.0)
        avg_out = None
        if total_out_span_ms > 0 and total_bytes_out > 0:
            avg_out = (total_bytes_out / (1024.0 * 1024.0)) / (total_out_span_ms / 1000.0)

        rows.append([
            model, seg_count,
            f'{total_in_span_ms}', f'{total_out_span_ms}',
            f'{total_bytes_in}', f'{total_bytes_out}',
            '' if avg_in is None else f'{avg_in}',
            '' if avg_out is None else f'{avg_out}',
        ])

        # Accumulate into grand totals
        grand_in_ms += total_in_span_ms
        grand_out_ms += total_out_span_ms
        grand_bytes_in += total_bytes_in
        grand_bytes_out += total_bytes_out
        grand_seg_count += seg_count

        per_model.append({
            'model': model,
            'seg_count': seg_count,
            'in_ms': total_in_span_ms,
            'out_ms': total_out_span_ms,
            'bytes_in': total_bytes_in,
            'bytes_out': total_bytes_out,
            'avg_in': avg_in,
            'avg_out': avg_out,
        })

    # Append overall totals row
    grand_avg_in = None
    if grand_in_ms > 0 and grand_bytes_in > 0:
        grand_avg_in = (grand_bytes_in / (1024.0 * 1024.0)) / (grand_in_ms / 1000.0)
    grand_avg_out = None
    if grand_out_ms > 0 and grand_bytes_out > 0:
        grand_avg_out = (grand_bytes_out / (1024.0 * 1024.0)) / (grand_out_ms / 1000.0)

    rows.append([
        'ALL', grand_seg_count,
        f'{grand_in_ms}', f'{grand_out_ms}',
        f'{grand_bytes_in}', f'{grand_bytes_out}',
        '' if grand_avg_in is None else f'{grand_avg_in}',
        '' if grand_avg_out is None else f'{grand_avg_out}',
    ])

    # Find lowest avg_out among models with a valid avg_out and exclude it for an alternate ALL
    valid_out = [m for m in per_model if m.get('avg_out') is not None]
    if valid_out:
        lowest = min(valid_out, key=lambda m: m['avg_out'])
        ex_in_ms = grand_in_ms - lowest['in_ms']
        ex_out_ms = grand_out_ms - lowest['out_ms']
        ex_bytes_in = grand_bytes_in - lowest['bytes_in']
        ex_bytes_out = grand_bytes_out - lowest['bytes_out']
        ex_avg_in = None
        if ex_in_ms > 0 and ex_bytes_in > 0:
            ex_avg_in = (ex_bytes_in / (1024.0 * 1024.0)) / (ex_in_ms / 1000.0)
        ex_avg_out = None
        if ex_out_ms > 0 and ex_bytes_out > 0:
            ex_avg_out = (ex_bytes_out / (1024.0 * 1024.0)) / (ex_out_ms / 1000.0)
        rows.append([
            f"ALL_EXCL_LOWEST_OUT:{lowest['model']}", grand_seg_count - lowest['seg_count'],
            f'{ex_in_ms}', f'{ex_out_ms}',
            f'{ex_bytes_in}', f'{ex_bytes_out}',
            '' if ex_avg_in is None else f'{ex_avg_in}',
            '' if ex_avg_out is None else f'{ex_avg_out}',
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
