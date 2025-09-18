#!/usr/bin/env python3
"""
Update five_models/results/single_pure_invoke_times.csv post/final values based on:
- Base pure time comes from gap p50 (results/.../combined_pure_gap_seg1-8_summary.csv) when available;
  otherwise falls back to existing pure_ms_pre in CSV. The script DOES NOT overwrite pure_ms_pre.
- For rows with type == 'off-chip':
    pure_ms_post  = base_pure + max(0, out_span_ms_mean - (bytes_out_MiB / Bin_speed)*1000)
    pure_ms_final = pure_ms_post
- For rows with type != 'off-chip':
    pure_ms_post  = base_pure
    pure_ms_final = base_pure

Bin_speed (MiB/s) is configurable via environment variable BIN_SPEED; default is 325.0.
Preserves CSV schema and column order; does not alter count/theory fields to avoid clobbering manual edits.
Updates 'source' to 'gap' or f'gap+adj{Bin_speed}'.
"""
import csv
import json
from pathlib import Path
import os
from shutil import copyfile

BASE = Path('/home/10210/Desktop/OS')
COMBINED_SPAN = BASE / 'results/models_local_batch_usbmon/single/combined_summary_span.json'
GAP_CSV = BASE / 'results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv'
OUT_CSV = BASE / 'five_models/results/single_pure_invoke_times.csv'
THEORY_SEG = BASE / 'five_models/baselines/theory_io_seg.json'


Bin_speed = 330.0

def load_combined_span():
    """Load combined span stats: returns {model:{seg:{pre:float, post:float, count:int, theory:float}}}."""
    if not COMBINED_SPAN.exists():
        return {}
    J = json.loads(COMBINED_SPAN.read_text())
    M = {}
    for model, segs in (J or {}).items():
        for seg, rec in (segs or {}).items():
            pre = rec.get('pure_span_ms_mean')
            post = rec.get('pure_span_offchip_adjusted_ms_mean', pre)
            cnt = rec.get('count')
            theo = rec.get('theory_out_mibps_used')
            try:
                pre = None if pre is None else float(pre)
            except Exception:
                pre = None
            try:
                post = None if post is None else float(post)
            except Exception:
                post = pre
            try:
                cnt = 0 if cnt is None else int(cnt)
            except Exception:
                cnt = 0
            try:
                theo = None if theo is None else float(theo)
            except Exception:
                theo = None
            M.setdefault(model, {})[seg] = {
                'pre': pre,
                'post': post,
                'count': cnt,
                'theory_out_mibps_used': theo,
                # extras for recomputing delta with 330 MiB/s
                'out_span_ms': rec.get('out_span_ms_mean'),
                'bytes_out_mean': rec.get('bytes_out_mean'),
            }
    return M

def load_gap_p50():
    """Load gap-based pure p50 from CSV: returns {(model,seg): p50_ms}."""
    if not GAP_CSV.exists():
        return {}
    with GAP_CSV.open() as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    m = {}
    for r in rows:
        key = (r.get('model'), r.get('segment'))
        if not key[0] or not key[1]:
            continue
        try:
            p50 = float(r.get('p50_ms') or r.get('mean_ms') or 0.0)
        except Exception:
            p50 = 0.0
        m[key] = p50
    return m


def update_csv(span_map, gap_map):
    rows = []
    with OUT_CSV.open() as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            r_type = (r.get('type') or '').strip()
            # Determine base pure time to use (do NOT overwrite pure_ms_pre)
            pre_to_use = None
            if (m, seg) in gap_map:
                try:
                    pre_to_use = float(gap_map[(m, seg)])
                except Exception:
                    pre_to_use = None
            if pre_to_use is None:
                # fallback to existing pure_ms_pre in CSV
                try:
                    pre_to_use = float(r.get('pure_ms_pre') or 0.0)
                except Exception:
                    pre_to_use = None

            # Populate post/final from span map and chosen pre
            span = (span_map.get(m) or {}).get(seg)
            if span is not None:
                # For off-chip rows use adjusted post; for on-chip, post==pre_to_use
                post_from_span = span.get('post')
                if r_type == 'off-chip':
                    # Compute delta using Bin_speed (MiB/s):
                    # delta = max(0, out_span_ms - (bytes_out_MiB/Bin_speed)*1000)
                    out_span_ms = span.get('out_span_ms')
                    bytes_out = span.get('bytes_out_mean')
                    use_post = pre_to_use
                    try:
                        out_span_ms = None if out_span_ms is None else float(out_span_ms)
                    except Exception:
                        out_span_ms = None
                    try:
                        bytes_out = None if bytes_out is None else float(bytes_out)
                    except Exception:
                        bytes_out = None
                    if pre_to_use is not None and out_span_ms is not None and bytes_out is not None and bytes_out > 0:
                        bytes_out_mib = bytes_out / (1024.0 * 1024.0)
                        theo_ms = (bytes_out_mib / Bin_speed) * 1000.0
                        delta = out_span_ms - theo_ms
                        if delta < 0:
                            delta = 0.0
                        use_post = float(pre_to_use) + float(delta)
                    elif post_from_span is not None:
                        use_post = post_from_span
                    if use_post is not None:
                        r['pure_ms_post'] = f"{use_post}"
                        r['pure_ms_final'] = f"{use_post}"
                        # adjustment flag based on whether adjusted differs from pre
                        try:
                            adj = (use_post is not None and pre_to_use is not None and abs(float(use_post) - float(pre_to_use)) > 1e-9)
                        except Exception:
                            adj = False
                        r['adjustment_applied'] = '1' if adj else '0'
                else:
                    # on-chip: post/final equals pre_to_use (leave pre column untouched)
                    if pre_to_use is not None:
                        r['pure_ms_post'] = f"{pre_to_use}"
                        r['pure_ms_final'] = f"{pre_to_use}"
                        r['adjustment_applied'] = '0'
                # Optionally update source to reflect basis; do not change count/theory to avoid clobbering
                if 'source' in r:
                    # tag includes actual Bin_speed used
                    tag = f"gap+adj{int(Bin_speed) if float(Bin_speed).is_integer() else Bin_speed}"
                    r['source'] = tag if r_type == 'off-chip' else 'gap'
            else:
                # No span info; keep existing values as-is
                pass

            rows.append(r)

    # Backup then write
    try:
        copyfile(OUT_CSV, OUT_CSV.with_suffix('.csv.measured.bak'))
    except Exception:
        pass
    with OUT_CSV.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader(); wr.writerows(rows)


def main():
    if not OUT_CSV.exists():
        raise SystemExit(f"Missing CSV to update: {OUT_CSV}")
    if not COMBINED_SPAN.exists():
        raise SystemExit(f"Missing span summary JSON: {COMBINED_SPAN}")
    span_map = load_combined_span()
    gap_map = load_gap_p50()
    update_csv(span_map, gap_map)
    print(f"Updated: {OUT_CSV}")


if __name__ == '__main__':
    main()
