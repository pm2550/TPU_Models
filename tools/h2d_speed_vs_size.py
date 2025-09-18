#!/usr/bin/env python3
import os
import json
import csv
from typing import Dict, Any, List, Optional, Tuple

BASE = "/home/10210/Desktop/OS"
SINGLE_SPAN_JSON = os.path.join(BASE, "results", "models_local_batch_usbmon", "single", "combined_summary_span.json")
CHAIN_ROOT = os.path.join(BASE, "results", "models_local_combo_chain")
OUT_CSV = os.path.join(BASE, "five_models", "results", "h2d_speed_vs_size.csv")

MiB = 1024.0 * 1024.0

def safe_load_json(p: str) -> Optional[Dict[str, Any]]:
    try:
        with open(p, 'r') as f:
            return json.load(f)
    except Exception:
        return None

def collect_single() -> List[List[Any]]:
    rows: List[List[Any]] = []
    J = safe_load_json(SINGLE_SPAN_JSON) or {}
    for model, segs in sorted(J.items()):
        if not isinstance(segs, dict):
            continue
        for seg_name, stats in sorted(segs.items()):
            if not isinstance(stats, dict):
                continue
            # H2D is OUT (Bo/Co) in this repo's convention
            bytes_out = float(stats.get('bytes_out_mean') or 0.0)
            span_ms = float(stats.get('out_span_ms_mean') or 0.0)
            if bytes_out <= 0 or span_ms <= 0:
                continue
            span_s = span_ms / 1000.0
            speed_mibps = (bytes_out / MiB) / span_s if span_s > 0 else 0.0
            rows.append([
                'single', model, None, seg_name, int(bytes_out), bytes_out / MiB, span_s, speed_mibps
            ])
    return rows

def find_perf_summary(dir_path: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(dir_path, 'performance_summary.json')
    return safe_load_json(p)

def collect_chain() -> List[List[Any]]:
    rows: List[List[Any]] = []
    if not os.path.isdir(CHAIN_ROOT):
        return rows
    for model in sorted(os.listdir(CHAIN_ROOT)):
        model_dir = os.path.join(CHAIN_ROOT, model)
        if not os.path.isdir(model_dir):
            continue
        for k in range(2, 9):
            k_dir = os.path.join(model_dir, f"K{k}")
            if not os.path.isdir(k_dir):
                continue
            for seg in sorted(d for d in os.listdir(k_dir) if d.startswith('seg') and os.path.isdir(os.path.join(k_dir, d))):
                seg_dir = os.path.join(k_dir, seg)
                P = find_perf_summary(seg_dir)
                if not P:
                    continue
                overall = (((P.get('io_performance') or {}).get('strict_window') or {}).get('overall_avg') or {})
                bytes_out = float(overall.get('avg_bytes_out_per_invoke') or 0.0)
                span_s = float(overall.get('span_s') or 0.0)
                if bytes_out <= 0 or span_s <= 0:
                    continue
                speed_mibps = (bytes_out / MiB) / span_s if span_s > 0 else 0.0
                rows.append([
                    'chain', model, k, seg, int(bytes_out), bytes_out / MiB, span_s, speed_mibps
                ])
    return rows

def main():
    rows: List[List[Any]] = []
    rows += collect_single()
    rows += collect_chain()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['mode','model','K','segment','bytes_out','size_MiB','span_s','speed_MiBps'])
        for r in rows:
            w.writerow(r)
    # Print a brief summary per mode
    def summary(mode: str) -> Tuple[int, float, float]:
        import statistics as st
        xs = [r for r in rows if r[0] == mode]
        speeds = [r[7] for r in xs]
        sizes = [r[5] for r in xs]
        n = len(xs)
        med_speed = st.median(speeds) if speeds else 0.0
        med_size = st.median(sizes) if sizes else 0.0
        return n, med_speed, med_size
    n1, msp1, msz1 = summary('single')
    n2, msp2, msz2 = summary('chain')
    print(f"saved: {OUT_CSV}")
    print(f"single: n={n1}, median_speed={msp1:.1f} MiB/s, median_size={msz1:.2f} MiB")
    print(f"chain : n={n2}, median_speed={msp2:.1f} MiB/s, median_size={msz2:.2f} MiB")

if __name__ == '__main__':
    main()
