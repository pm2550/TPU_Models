#!/usr/bin/env python3
"""
Rewrite five_models/results/single_pure_invoke_times.csv values from span summaries:
- pure_ms_pre  = pure_span_ms_mean (from combined_summary_span.json) for both on-chip and off-chip
- For rows with type == 'off-chip':
    pure_ms_post  = pure_span_offchip_adjusted_ms_mean
    pure_ms_final = pure_ms_post
- For rows with type != 'off-chip':
    pure_ms_post  = pure_ms_pre
    pure_ms_final = pure_ms_pre

Additionally, if available, refresh 'count' and 'theory_out_mibps_used' from span JSON.
Preserve CSV schema and column order; do not add or remove columns. Update 'source' to
'span+adj330' for off-chip rows and 'span' for on-chip rows.
"""
import csv
import json
from pathlib import Path
from shutil import copyfile

BASE = Path('/home/10210/Desktop/OS')
COMBINED_SPAN = BASE / 'results/models_local_batch_usbmon/single/combined_summary_span.json'
OUT_CSV = BASE / 'five_models/results/single_pure_invoke_times.csv'
THEORY_SEG = BASE / 'five_models/baselines/theory_io_seg.json'

Bin_speed = 325.0  # MiB/s for off-chip adjustment if recomputing delta

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


def update_csv(span_map):
    rows = []
    with OUT_CSV.open() as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            r_type = (r.get('type') or '').strip()
            # Populate from span map if available
            span = (span_map.get(m) or {}).get(seg)
            if span is not None:
                pre = span.get('pre')
                # keep the adjusted from span as fallback
                post_from_span = span.get('post')
                if pre is not None:
                    r['pure_ms_pre'] = f"{pre}"
                # For off-chip rows use adjusted post; for on-chip, post==pre
                if r_type == 'off-chip':
                    # Compute delta with a single standard speed 330 MiB/s:
                    # delta = max(0, out_span_ms - (bytes_out_MiB/330)*1000)
                    out_span_ms = span.get('out_span_ms')
                    bytes_out = span.get('bytes_out_mean')
                    use_post = pre
                    try:
                        out_span_ms = None if out_span_ms is None else float(out_span_ms)
                    except Exception:
                        out_span_ms = None
                    try:
                        bytes_out = None if bytes_out is None else float(bytes_out)
                    except Exception:
                        bytes_out = None
                    if pre is not None and out_span_ms is not None and bytes_out is not None and bytes_out > 0:
                        bytes_out_mib = bytes_out / (1024.0 * 1024.0)
                        theo_ms = (bytes_out_mib / Bin_speed) * 1000.0
                        delta = out_span_ms - theo_ms
                        if delta < 0:
                            delta = 0.0
                        use_post = float(pre) + float(delta)
                    elif post_from_span is not None:
                        use_post = post_from_span
                    if use_post is not None:
                        r['pure_ms_post'] = f"{use_post}"
                        r['pure_ms_final'] = f"{use_post}"
                        # adjustment flag based on whether adjusted differs from pre
                        try:
                            adj = (use_post is not None and pre is not None and abs(float(use_post) - float(pre)) > 1e-9)
                        except Exception:
                            adj = False
                        r['adjustment_applied'] = '1' if adj else '0'
                else:
                    # on-chip
                    if pre is not None:
                        r['pure_ms_post'] = f"{pre}"
                        r['pure_ms_final'] = f"{pre}"
                        r['adjustment_applied'] = '0'
                # Refresh count and theory if available
                if span.get('count') is not None:
                    r['count'] = str(int(span['count']))
                if span.get('theory_out_mibps_used') is not None:
                    r['theory_out_mibps_used'] = f"{float(span['theory_out_mibps_used'])}"
                # Update source if such a column exists
                if 'source' in r:
                    r['source'] = 'span+adj330' if r_type == 'off-chip' else 'span'
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
    update_csv(span_map)
    print(f"Updated: {OUT_CSV}")


if __name__ == '__main__':
    main()
