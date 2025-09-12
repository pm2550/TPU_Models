#!/usr/bin/env python3
import os
import re
import csv
import json
import argparse
import statistics as st
import subprocess

VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"
CORRECT = "/home/10210/Desktop/OS/tools/correct_per_invoke_stats.py"
RESULTS_BASE = "/home/10210/Desktop/OS/results/models_local_combo_chain"
OUT_CSV = "/home/10210/Desktop/OS/five_models/results/combo_real_metrics.csv"

def run_correct(usb_path: str, merged_path: str, time_map_path: str, extra_s: float) -> list:
    """Run correct_per_invoke_stats and return JSON_PER_INVOKE list"""
    if not (os.path.isfile(usb_path) and os.path.isfile(merged_path) and os.path.isfile(time_map_path)):
        return []
    cmd = [
        VENV_PY, CORRECT,
        usb_path, merged_path, time_map_path,
        "--mode", "bulk_complete",
        "--include", "overlap",
        "--extra", f"{extra_s:.3f}",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout or ""
    m = re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", out, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(1))
    except Exception:
        return []

def rebuild_merged_spans(k_dir: str) -> str:
    """Rebuild merged_invokes.json by concatenating seg*/invokes.json, sorted by begin"""
    merged_path = os.path.join(k_dir, 'merged_invokes.json')
    spans = []
    for name in sorted(os.listdir(k_dir)):
        p = os.path.join(k_dir, name)
        if not (os.path.isdir(p) and name.startswith('seg')):
            continue
        invp = os.path.join(p, 'invokes.json')
        if not os.path.isfile(invp):
            continue
        try:
            J = json.load(open(invp))
            for sp in (J.get('spans') or []):
                b = float(sp.get('begin'))
                e = float(sp.get('end'))
                if e > b:
                    spans.append({'begin': b, 'end': e, 'seg_label': name})
        except Exception:
            continue
    spans.sort(key=lambda x: x['begin'])
    try:
        with open(merged_path, 'w') as f:
            json.dump({'spans': spans}, f)
    except Exception:
        pass
    return merged_path

def list_segments(k_dir: str) -> list:
    try:
        return [d for d in sorted(os.listdir(k_dir)) if d.startswith('seg') and os.path.isdir(os.path.join(k_dir, d))]
    except Exception:
        return []

def load_merged_spans(k_dir: str) -> list:
    p = os.path.join(k_dir, 'merged_invokes.json')
    try:
        j = json.load(open(p))
        return j.get('spans', [])
    except Exception:
        return []

def load_invoke_mean_ms(seg_dir: str) -> float:
    p = os.path.join(seg_dir, 'performance_summary.json')
    try:
        j = json.load(open(p))
        inv = ((j.get('inference_performance') or {}).get('invoke_times') or {})
        return float(inv.get('mean_ms') or 0.0)
    except Exception:
        return 0.0

def avg(vals):
    return st.mean(vals) if vals else 0.0

def summarize_model(model_name: str, extra_s: float) -> list:
    rows = []
    model_dir = os.path.join(RESULTS_BASE, model_name)
    if not os.path.isdir(model_dir):
        return rows
    for k in range(2, 9):
        k_dir = os.path.join(model_dir, f"K{k}")
        usb = os.path.join(k_dir, 'usbmon.txt')
        # 强制重建 merged_invokes，避免使用过期窗口
        merged = rebuild_merged_spans(k_dir)
        tm = os.path.join(k_dir, 'time_map.json')
        per = run_correct(usb, merged, tm, extra_s)
        if not per:
            continue
        spans = load_merged_spans(k_dir)
        seg2idx = {}
        for i, s in enumerate(spans):
            seg2idx.setdefault(s.get('seg_label'), []).append(i)
        for seg in list_segments(k_dir):
            idxs = seg2idx.get(seg, [])
            xs = [per[i] for i in idxs if 0 <= i < len(per)]
            xs = [x for x in xs if (x.get('bytes_in', 0) > 0 or x.get('bytes_out', 0) > 0)]
            if not xs:
                continue
            in_ms = avg([x.get('in_active_s', 0.0) * 1000.0 for x in xs])
            out_ms = avg([x.get('out_active_s', 0.0) * 1000.0 for x in xs])
            uni_ms = avg([max(x.get('in_active_s', 0.0), x.get('out_active_s', 0.0)) * 1000.0 for x in xs])
            bin_avg = int(avg([x.get('bytes_in', 0) for x in xs]))
            bout_avg = int(avg([x.get('bytes_out', 0) for x in xs]))
            cover = avg([(min(x.get('in_active_s', 0.0), x.get('out_active_s', 0.0)) / (x.get('in_active_s', 0.0) or 1e-9)) for x in xs])
            frac = sum(1 for x in xs if x.get('out_active_s', 0.0) >= x.get('in_active_s', 0.0)) / len(xs)
            inv_mean = load_invoke_mean_ms(os.path.join(k_dir, seg))
            rows.append([
                model_name, k, seg, inv_mean, bin_avg, bout_avg,
                in_ms, out_ms, uni_ms, cover, frac, len(xs)
            ])
    return rows

def main():
    ap = argparse.ArgumentParser(description='Summarize real I/O metrics for combos')
    ap.add_argument('--models', nargs='*', help='limit to specific model names')
    ap.add_argument('--extra', type=float, default=0.020, help='window expansion seconds for parser')
    args = ap.parse_args()

    models = args.models if args.models else [d for d in sorted(os.listdir(RESULTS_BASE)) if os.path.isdir(os.path.join(RESULTS_BASE, d))]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model','K','segment','invoke_mean_ms','avg_in_bytes','avg_out_bytes','avg_in_ms','avg_out_ms','avg_union_ms','avg_in_covered_by_out','frac_windows_out>=in','num_samples'])
        total = 0
        for m in models:
            rows = summarize_model(m, args.extra)
            for r in rows:
                w.writerow(r)
            total += len(rows)
    print(f"saved: {OUT_CSV}, rows={total}")

if __name__ == '__main__':
    main()


