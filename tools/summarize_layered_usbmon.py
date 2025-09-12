#!/usr/bin/env python3
import os
import json
import csv
from typing import List, Dict

ROOT = "/home/10210/Desktop/OS"
RESULTS = os.path.join(ROOT, "results", "layered_usbmon")
OUT_CSV = os.path.join(ROOT, "results", "layered_usbmon_summary.csv")


def list_models() -> List[str]:
    if not os.path.isdir(RESULTS):
        return []
    names = []
    for d in sorted(os.listdir(RESULTS)):
        p = os.path.join(RESULTS, d)
        if os.path.isdir(p):
            names.append(d)
    return names


def read_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def collect_invoke_windows(dir_path: str) -> List[Dict]:
    inv = read_json(os.path.join(dir_path, 'invokes.json'))
    if not inv:
        return []
    return inv.get('spans') or []


def calc_union_active_per_invoke(dir_path: str) -> List[float]:
    cap = os.path.join(dir_path, 'usbmon.txt')
    inv = os.path.join(dir_path, 'invokes.json')
    tmap = os.path.join(dir_path, 'time_map.json')
    if not (os.path.isfile(cap) and os.path.isfile(inv) and os.path.isfile(tmap)):
        return []
    # 直接调用已校准的统计脚本，采用 overlap + bulk_complete 组合口径
    import subprocess, json as _json
    cmd = [
        os.path.join(ROOT, '.venv', 'bin', 'python'),
        os.path.join(ROOT, 'tools', 'correct_per_invoke_stats.py'),
        cap, inv, tmap, '--extra', '0.010', '--mode', 'bulk_complete', '--include', 'overlap'
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    union = []
    for line in (r.stdout or '').splitlines():
        if line.startswith('JSON_PER_INVOKE:'):
            try:
                payload = _json.loads(line.split(':',1)[1].strip())
                union = [float(x.get('union_active_s', 0.0) or 0.0) for x in payload]
            except Exception:
                union = []
            break
    return union


def main():
    rows = []
    for name in list_models():
        mdl_dir = os.path.join(RESULTS, name)
        # warm
        warm_dir = os.path.join(mdl_dir, 'warm')
        warm_windows = collect_invoke_windows(warm_dir)
        warm_union = calc_union_active_per_invoke(warm_dir)
        # 去掉首个（预热后的100次）
        warm_invoke_ms = [max(0.0, (w.get('end',0)-w.get('begin',0))*1000.0) for w in warm_windows[1:]] if len(warm_windows)>1 else []
        warm_union_ms = [x*1000.0 for x in warm_union[1:]] if len(warm_union)>1 else []
        # cold: 逐 run*/ 目录聚合
        cold_root = os.path.join(mdl_dir, 'cold')
        cold_invoke_ms_all: List[float] = []
        cold_union_ms_all: List[float] = []
        if os.path.isdir(cold_root):
            for d in sorted(os.listdir(cold_root)):
                dpath = os.path.join(cold_root, d)
                if not os.path.isdir(dpath):
                    continue
                w = collect_invoke_windows(dpath)
                cold_invoke_ms_all += [max(0.0,(x.get('end',0)-x.get('begin',0))*1000.0) for x in w]
                u = calc_union_active_per_invoke(dpath)
                cold_union_ms_all += [x*1000.0 for x in u]
        # 求均值
        import statistics
        def mean_or_blank(v: List[float]):
            return f"{statistics.mean(v):.3f}" if v else ''
        rows.append([name,
                    len(warm_invoke_ms), mean_or_blank(warm_invoke_ms), mean_or_blank(warm_union_ms),
                    len(cold_invoke_ms_all), mean_or_blank(cold_invoke_ms_all), mean_or_blank(cold_union_ms_all)])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model','warm_count','warm_avg_invoke_ms','warm_avg_union_active_ms','cold_count','cold_avg_invoke_ms','cold_avg_union_active_ms'])
        w.writerows(rows)
    print(f"saved: {OUT_CSV}, rows={len(rows)}")


if __name__ == '__main__':
    main()


