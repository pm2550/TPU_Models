#!/usr/bin/env python3
import os
import json
import csv
from typing import List, Dict

RESULTS_ROOT = "/home/10210/Desktop/OS/results/models_local_combo_chain"
OUTPUT_CSV = "/home/10210/Desktop/OS/results/combo_cycle_times.csv"

def read_pure_invoke_times(perf_path: str) -> List[float]:
    try:
        with open(perf_path, 'r') as f:
            j = json.load(f)
        pit = (j.get('inference_performance') or {}).get('pure_invoke_times') or {}
        times = pit.get('all_times_ms') or []
        # ensure floats
        return [float(x) for x in times]
    except Exception:
        return []


def read_invoke_times(perf_path: str) -> List[float]:
    try:
        with open(perf_path, 'r') as f:
            j = json.load(f)
        it = (j.get('inference_performance') or {}).get('invoke_times') or {}
        times = it.get('all_times_ms') or []
        return [float(x) for x in times]
    except Exception:
        return []


def read_spans(invokes_path: str) -> List[Dict[str, float]]:
    try:
        with open(invokes_path, 'r') as f:
            j = json.load(f)
        return j.get('spans') or []
    except Exception:
        return []

def sum_cycle_time_ms(seg_to_pure_ms: Dict[str, List[float]]) -> List[float]:
    if not seg_to_pure_ms:
        return []
    # 对齐循环次数：使用所有段的最小长度，避免越界
    min_len = min(len(v) for v in seg_to_pure_ms.values() if v)
    totals: List[float] = []
    for i in range(min_len):
        total_ms = 0.0
        for _seg, times in seg_to_pure_ms.items():
            total_ms += float(times[i])
        totals.append(total_ms)
    return totals

def main():
    rows = []
    if not os.path.isdir(RESULTS_ROOT):
        print(f"not found: {RESULTS_ROOT}")
        return
    # 遍历模型
    for model_name in sorted(os.listdir(RESULTS_ROOT)):
        model_dir = os.path.join(RESULTS_ROOT, model_name)
        if not os.path.isdir(model_dir):
            continue
        # 遍历 K=2..8
        for k in range(2, 9):
            kdir = os.path.join(model_dir, f"K{k}")
            if not os.path.isdir(kdir):
                continue
            # 收集所有以 seg 开头的阶段目录（支持 seg2to8 等）
            seg_dirs = [d for d in sorted(os.listdir(kdir)) if d.startswith('seg') and os.path.isdir(os.path.join(kdir, d))]
            if not seg_dirs:
                continue
            seg_to_invoke: Dict[str, List[float]] = {}
            for sd in seg_dirs:
                perfp = os.path.join(kdir, sd, 'performance_summary.json')
                times = read_invoke_times(perfp)
                if times:
                    seg_to_invoke[sd] = times

            # 若不存在 invoke_times 明细，回退到 invokes.json 的窗口时长
            if not seg_to_invoke:
                seg_to_spans: Dict[str, List[Dict[str, float]]] = {}
                for sd in seg_dirs:
                    invp = os.path.join(kdir, sd, 'invokes.json')
                    spans = read_spans(invp)
                    if spans:
                        seg_to_spans[sd] = spans
                if not seg_to_spans:
                    continue
                # 用窗口时长（秒）相加，转换为毫秒
                totals_ms: List[float] = []
                min_len = min(len(v) for v in seg_to_spans.values() if v)
                for i in range(min_len):
                    total_s = 0.0
                    for _seg, spans in seg_to_spans.items():
                        s = spans[i]
                        b = float(s.get('begin', 0.0) or 0.0)
                        e = float(s.get('end', 0.0) or 0.0)
                        if e > b:
                            total_s += (e - b)
                    totals_ms.append(total_s * 1000.0)
            else:
                totals_ms = sum_cycle_time_ms(seg_to_invoke)
            for idx, ms in enumerate(totals_ms, start=1):
                rows.append([model_name, k, idx, f"{ms:.3f}"])

    # 写 CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["model", "K", "cycle_index", "total_cycle_ms"])
        w.writerows(rows)
    print(f"saved: {OUTPUT_CSV}, rows={len(rows)}")

if __name__ == '__main__':
    main()


