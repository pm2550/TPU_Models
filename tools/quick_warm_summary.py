#!/usr/bin/env python3
import os
import json
import csv
import statistics
import subprocess

ROOT = "/home/10210/Desktop/OS"
RES = os.path.join(ROOT, "results", "layered_usbmon")
OUT = os.path.join(ROOT, "results", "layered_usbmon_warm_quick_summary.csv")


def read_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def summarize_model(name: str):
    d = os.path.join(RES, name, 'warm')
    invp = os.path.join(d, 'invokes.json')
    tmap = os.path.join(d, 'time_map.json')
    cap = os.path.join(d, 'usbmon.txt')
    iv_ms_mean = ''
    union_ms_mean = ''

    J = read_json(invp)
    if J:
        spans = J.get('spans') or []
        if len(spans) > 1:
            iv_ms = [max(0.0, (s['end'] - s['begin']) * 1000.0) for s in spans[1:]]
            if iv_ms:
                iv_ms_mean = f"{statistics.mean(iv_ms):.3f}"

    if os.path.isfile(cap) and os.path.isfile(tmap) and J:
        cmd = [
            os.path.join(ROOT, '.venv', 'bin', 'python'),
            os.path.join(ROOT, 'tools', 'correct_per_invoke_stats.py'),
            cap, invp, tmap, '--extra', '0.010', '--mode', 'bulk_complete', '--include', 'overlap'
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        payload = None
        for ln in (r.stdout or '').splitlines():
            if ln.startswith('JSON_PER_INVOKE:'):
                try:
                    payload = json.loads(ln.split(':', 1)[1].strip())
                except Exception:
                    payload = None
                break
        if payload and len(payload) > 1:
            union = []
            for x in payload[1:]:  # 跳过首个
                u = float(x.get('union_active_s', 0.0) or 0.0)
                union.append(u * 1000.0)
            if union:
                union_ms_mean = f"{statistics.mean(union):.3f}"

    return iv_ms_mean, union_ms_mean


def main():
    models = [d for d in sorted(os.listdir(RES)) if os.path.isdir(os.path.join(RES, d))]
    rows = []
    for m in models:
        iv, ua = summarize_model(m)
        rows.append([m, iv, ua])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'warm_avg_invoke_ms', 'warm_avg_union_active_ms'])
        w.writerows(rows)
    print(f"saved: {OUT}, rows={len(rows)}")


if __name__ == '__main__':
    main()


