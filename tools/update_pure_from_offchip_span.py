#!/usr/bin/env python3
"""
Update five_models/results/single_pure_invoke_times.csv using measured values from
results/models_local_batch_usbmon/single/offchip_summary_span.json.

Mapping per (model, segX):
- pure_ms_pre  <- pure_span_ms_mean
- pure_ms_post <- pure_span_offchip_adjusted_ms_mean
- pure_ms_final <- pure_ms_post
- adjustment_applied: 1 if post > pre else 0
- theory_out_mibps_used: from json field theory_out_mibps_used when present
- count: from json 'count'
- source: 'measured_span'

Non-listed segments/models are left unchanged.
"""
import csv
import json
from pathlib import Path
from shutil import copyfile

BASE = Path('/home/10210/Desktop/OS')
IN_JSON = BASE / 'results/models_local_batch_usbmon/single/offchip_summary_span.json'
COMBINED_GAP = BASE / 'results/models_local_batch_usbmon/single/combined_pure_gap_seg1-8_summary.csv'
OUT_CSV = BASE / 'five_models/results/single_pure_invoke_times.csv'
THEORY_SEG = BASE / 'five_models/baselines/theory_io_seg.json'

# Standard inbound bandwidth (MiB/s)
B_IN_STD = 330.0


def load_span_adjusted():
    """Load off-chip span summary; returns {model:{seg:{adj: ms, span_pure: ms, theory: str, count:int}}}."""
    J = json.loads(IN_JSON.read_text())
    M = {}
    for model, segs in J.items():
        for seg, rec in (segs or {}).items():
            M.setdefault(model, {})[seg] = {
                'adj': float(rec.get('pure_span_offchip_adjusted_ms_mean') or 0.0),
                'span_pure': float(rec.get('pure_span_ms_mean') or 0.0),
                'in_speed': float(rec.get('in_speed_span_mibps_mean') or 0.0),
                'theory_out_mibps_used': str(rec.get('theory_out_mibps_used') or ''),
                'count': int(rec.get('count') or 0),
            }
    return M

def load_gap_p50():
    """Load combined gap p50 per segment; returns {model:{seg: p50_ms}}."""
    G = {}
    if not COMBINED_GAP.exists():
        return G
    import csv as _csv
    with COMBINED_GAP.open() as f:
        rd = _csv.DictReader(f)
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            if not m or not seg:
                continue
            try:
                p50 = float(r.get('p50_ms') or 0.0)
            except Exception:
                p50 = 0.0
            G.setdefault(m, {})[seg] = p50
    return G


def load_off_used_mib():
    """Load off_used_MiB per segment from theory_io_seg.json; returns {model:{seg: off_used_MiB}}."""
    if not THEORY_SEG.exists():
        return {}
    J = json.loads(THEORY_SEG.read_text())
    out = {}
    for model, obj in (J or {}).items():
        segs = (obj or {}).get('segments', {})
        for seg, gd in (segs or {}).items():
            try:
                off_mib = float(gd.get('off_used_MiB') or gd.get('off_chip_MiB') or 0.0)
            except Exception:
                off_mib = 0.0
            out.setdefault(model, {})[seg] = off_mib
    return out


def update_csv(span_adj, gap_p50):
    rows = []
    with OUT_CSV.open() as f:
        rd = csv.DictReader(f)
        fieldnames = rd.fieldnames or []
        off_used = load_off_used_mib()
        for r in rd:
            m = r.get('model'); seg = r.get('segment')
            # 1) pre 用 gap p50（若有）
            if m in gap_p50 and seg in gap_p50[m]:
                pre = float(gap_p50[m][seg])
                r['pure_ms_pre'] = f"{pre}"
            else:
                try:
                    pre = float(r.get('pure_ms_pre') or 0.0)
                except Exception:
                    pre = 0.0
            # 2) post/final = pre + delta
            #    其中 delta = max(0, off_used_MiB * (1/actual_in - 1/B_IN_STD)) * 1000
            #    actual_in 取自 span 的 in_speed_span_mibps_mean
            if m in span_adj and seg in span_adj[m]:
                actual_in = float(span_adj[m][seg].get('in_speed') or 0.0)
                off_mib = float((off_used.get(m, {}) or {}).get(seg, 0.0))
                delta = 0.0
                if off_mib > 0.0 and actual_in > 0.0:
                    delta = (1.0/actual_in - 1.0/B_IN_STD) * off_mib * 1000.0
                    if delta < 0.0:
                        delta = 0.0
                post = pre + delta
                r['pure_ms_post'] = f"{post}"
                r['pure_ms_final'] = f"{post}"
                r['adjustment_applied'] = '1' if post > pre else '0'
                # 仅当 JSON 明确提供理论带宽值时才覆盖
                if span_adj[m][seg]['theory_out_mibps_used']:
                    r['theory_out_mibps_used'] = span_adj[m][seg]['theory_out_mibps_used']
                if span_adj[m][seg]['count']:
                    r['count'] = str(span_adj[m][seg]['count'])
            else:
                # 无调整信息则保持 post=pre
                r['pure_ms_post'] = f"{pre}"
                r['pure_ms_final'] = f"{pre}"
                r['adjustment_applied'] = '0'
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
    # span 调整信息可选，有则用于 post 的 max
    span_adj = load_span_adjusted() if IN_JSON.exists() else {}
    gap_p50 = load_gap_p50()
    update_csv(span_adj, gap_p50)
    print(f"Updated: {OUT_CSV}")


if __name__ == '__main__':
    main()
