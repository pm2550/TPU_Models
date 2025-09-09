#!/usr/bin/env python3
import os
import json
import csv

RESULTS_ROOT = "/home/10210/Desktop/OS/results/models_local_combo_chain"
MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
OUTPUT_CSV = "/home/10210/Desktop/OS/results/combo_pure_invoke_times.csv"

def main():
    rows = []
    if not os.path.isdir(RESULTS_ROOT):
        print(f"not found: {RESULTS_ROOT}")
        return
    for model_name in sorted(os.listdir(RESULTS_ROOT)):
        model_dir = os.path.join(RESULTS_ROOT, model_name)
        if not os.path.isdir(model_dir):
            continue
        for k in range(2, 9):
            kdir = os.path.join(model_dir, f"K{k}")
            if not os.path.isdir(kdir):
                continue
            # 支持 seg1、seg2to8 等目录
            # 读取 full split summary 切点
            try:
                summ = os.path.join(MODELS_BASE, model_name, 'full_split_pipeline_local', 'summary.json')
                SJ = json.load(open(summ))
                cps = SJ.get('cut_points') or []
            except Exception:
                cps = []

            for seg_name in sorted(os.listdir(kdir)):
                seg_dir = os.path.join(kdir, seg_name)
                if not (os.path.isdir(seg_dir) and seg_name.startswith('seg')):
                    continue
                perf = os.path.join(seg_dir, 'performance_summary.json')
                if not os.path.isfile(perf):
                    continue
                try:
                    J = json.load(open(perf))
                    pure = J.get('inference_performance', {}).get('pure_invoke_times', {})
                    # seg 索引：'seg2to8' 取起始数字 2
                    import re
                    m = re.match(r"seg(\d+)", seg_name)
                    seg_idx = int(m.group(1)) if m else 1
                    # 切点层名（同 single 规则）
                    if seg_idx == 1:
                        cut_start = 'INPUT'
                        cut_end = cps[0] if len(cps) >= 1 else 'OUTPUT'
                    elif seg_idx == 8:
                        cut_start = cps[6] if len(cps) >= 7 else (cps[-1] if cps else 'INPUT')
                        cut_end = 'OUTPUT'
                    else:
                        i0 = seg_idx - 2
                        i1 = seg_idx - 1
                        cut_start = cps[i0] if i0 < len(cps) else (cps[-1] if cps else 'INPUT')
                        cut_end = cps[i1] if i1 < len(cps) else 'OUTPUT'
                    rows.append([
                        model_name, k, seg_name, cut_start, cut_end,
                        f"{float(pure.get('mean_ms', 0.0) or 0.0):.3f}",
                        f"{float(pure.get('stdev_ms', 0.0) or 0.0):.3f}",
                        int(pure.get('count', J.get('inference_performance', {}).get('count', 0)) or 0),
                    ])
                except Exception:
                    continue
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["model", "K", "segment", "cut_start_layer", "cut_end_layer", "pure_invoke_mean_ms", "pure_invoke_stdev_ms", "count"])
        w.writerows(rows)
    print(f"saved: {OUTPUT_CSV}, rows={len(rows)}")

if __name__ == '__main__':
    main()


