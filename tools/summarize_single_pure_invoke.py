#!/usr/bin/env python3
import os
import json
import csv

SINGLE_ROOT = "/home/10210/Desktop/OS/results/models_local_batch_usbmon/single"
MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
OUT_CSV = "/home/10210/Desktop/OS/results/single_pure_invoke_times.csv"


def load_cut_points(model: str):
    cut_start = {}
    cut_end = {}
    try:
        summ = os.path.join(MODELS_BASE, model, 'full_split_pipeline_local', 'summary.json')
        SJ = json.load(open(summ))
        cps = SJ.get('cut_points') or []
        for seg in range(1, 9):
            if seg == 1:
                cut_start[seg] = 'INPUT'
                cut_end[seg] = cps[0] if len(cps) >= 1 else 'OUTPUT'
            elif seg == 8:
                cut_start[seg] = cps[6] if len(cps) >= 7 else (cps[-1] if cps else 'INPUT')
                cut_end[seg] = 'OUTPUT'
            else:
                i0 = seg - 2
                i1 = seg - 1
                cut_start[seg] = cps[i0] if i0 < len(cps) else (cps[-1] if cps else 'INPUT')
                cut_end[seg] = cps[i1] if i1 < len(cps) else 'OUTPUT'
    except Exception:
        for seg in range(1, 9):
            cut_start[seg] = f'seg{seg}_start'
            cut_end[seg] = f'seg{seg}_end'
    return cut_start, cut_end


def main():
    rows = []
    if not os.path.isdir(SINGLE_ROOT):
        print(f"not found: {SINGLE_ROOT}")
        return
    for model in sorted(os.listdir(SINGLE_ROOT)):
        mdir = os.path.join(SINGLE_ROOT, model)
        if not os.path.isdir(mdir):
            continue
        cut_start, cut_end = load_cut_points(model)
        for seg in range(1, 9):
            sdir = os.path.join(mdir, f"seg{seg}")
            perfp = os.path.join(sdir, 'performance_summary.json')
            if not os.path.isfile(perfp):
                continue
            try:
                P = json.load(open(perfp))
            except Exception:
                P = {}
            inf = P.get('inference_performance') or {}
            pit = inf.get('pure_invoke_times') or {}
            mean_ms = pit.get('mean_ms')
            stdev_ms = pit.get('stdev_ms')
            all_times = pit.get('all_times_ms') or []
            count = len(all_times)
            rows.append([
                model,
                f"seg{seg}",
                cut_start.get(seg, ''),
                cut_end.get(seg, ''),
                f"{float(mean_ms) if mean_ms is not None else 0:.3f}",
                f"{float(stdev_ms) if stdev_ms is not None else 0:.3f}",
                count,
            ])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'segment', 'cut_start_layer', 'cut_end_layer', 'pure_invoke_mean_ms', 'pure_invoke_stdev_ms', 'count'])
        w.writerows(rows)
    print(f"saved: {OUT_CSV}, rows={len(rows)}")


if __name__ == '__main__':
    main()


