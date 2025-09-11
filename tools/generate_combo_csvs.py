#!/usr/bin/env python3
import os
import re
import csv
import json
from typing import Dict, List, Optional


ROOT_DIR = "/home/10210/Desktop/OS"
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "models_local_combo_chain")
OUT_DIR = os.path.join(ROOT_DIR, "five_models", "results")
THEORY_COMBOS_JSON = os.path.join(ROOT_DIR, "baselines", "theory_io_combos.json")


def safe_load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def list_segment_dirs_for_k(k_dir: str, k_value: int) -> List[str]:
    if not os.path.isdir(k_dir):
        return []
    labels = [
        d for d in os.listdir(k_dir)
        if d.startswith("seg") and os.path.isdir(os.path.join(k_dir, d))
    ]
    def seg_key(lbl: str) -> int:
        m = re.match(r"seg(\d+)", lbl)
        return int(m.group(1)) if m else 999
    labels.sort(key=seg_key)

    if k_value == 8:
        return [s for s in labels if s in [f"seg{i}" for i in range(1, 9)]]
    expected_heads = {f"seg{i}": True for i in range(1, k_value)}
    tail = f"seg{k_value}to8"
    return [s for s in labels if (s in expected_heads) or (s == tail)]


def generate_combo_cycle_times() -> int:
    rows: List[List[str]] = []
    if not os.path.isdir(RESULTS_DIR):
        return 0
    for model_name in sorted(os.listdir(RESULTS_DIR)):
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue
        for k in range(2, 9):
            k_dir = os.path.join(model_dir, f"K{k}")
            if not os.path.isdir(k_dir):
                continue
            seg_labels = list_segment_dirs_for_k(k_dir, k)
            if not seg_labels:
                continue
            # 收集每段的 invoke 全量序列（用于计算完整 cycle 耗时）
            seg_to_series: Dict[str, List[float]] = {}
            for seg_label in seg_labels:
                summary_path = os.path.join(k_dir, seg_label, "performance_summary.json")
                sj = safe_load_json(summary_path)
                if not sj:
                    continue
                invoke = (((sj.get("inference_performance") or {}).get("invoke_times")) or {})
                series = invoke.get("all_times_ms") or []
                if series:
                    seg_to_series[seg_label] = [float(x) for x in series]
            if not seg_to_series:
                continue
            min_len = min(len(v) for v in seg_to_series.values() if v)
            for idx in range(min_len):
                total_ms = 0.0
                for _seg, series in seg_to_series.items():
                    total_ms += float(series[idx])
                rows.append([model_name, k, idx + 1, f"{total_ms:.3f}"])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "combo_cycle_times.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "K", "cycle_index", "total_cycle_ms"])
        w.writerows(rows)
    return len(rows)


def mibps_from_bytes_and_active_ms(num_bytes: float, active_ms: Optional[float]) -> str:
    if not active_ms or active_ms <= 0:
        return ""
    mib = float(num_bytes) / 1024.0 / 1024.0
    sec = active_ms / 1000.0
    if sec <= 0:
        return ""
    return f"{(mib / sec):.3f}"


def generate_combo_segment_metrics() -> int:
    theory = safe_load_json(THEORY_COMBOS_JSON) or {}
    rows: List[List[str]] = []
    if not os.path.isdir(RESULTS_DIR):
        return 0
    for model_name in sorted(os.listdir(RESULTS_DIR)):
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.isdir(model_dir):
            continue
        theory_model = (theory.get(model_name) or {}).get("combos") or {}
        for k in range(2, 9):
            k_dir = os.path.join(model_dir, f"K{k}")
            if not os.path.isdir(k_dir):
                continue
            for seg_label in list_segment_dirs_for_k(k_dir, k):
                summary_path = os.path.join(k_dir, seg_label, "performance_summary.json")
                sj = safe_load_json(summary_path)
                if not sj:
                    continue
                overall = (((sj.get("io_performance") or {}).get("strict_window") or {}).get("overall_avg")) or {}
                avg_in_b = float(overall.get("avg_bytes_in_per_invoke") or 0.0)
                avg_out_b = float(overall.get("avg_bytes_out_per_invoke") or 0.0)
                union = (sj.get("io_active_union_avg") or {})
                avg_active_ms = union.get("avg_active_ms")
                try:
                    avg_active_ms = float(avg_active_ms) if avg_active_ms is not None else None
                except Exception:
                    avg_active_ms = None

                out_mibps = mibps_from_bytes_and_active_ms(avg_out_b, avg_active_ms)
                in_mibps = mibps_from_bytes_and_active_ms(avg_in_b, avg_active_ms)

                t_entry = ((theory_model.get(f"K{k}") or {}).get(seg_label)) or {}
                th_out = t_entry.get("theory_OUT_bytes")
                th_in = t_entry.get("theory_IN_bytes")

                rows.append([
                    model_name,
                    k,
                    seg_label,
                    f"{avg_out_b:.0f}",
                    f"{avg_in_b:.0f}",
                    f"{avg_active_ms:.3f}" if isinstance(avg_active_ms, float) else "",
                    out_mibps,
                    in_mibps,
                    th_out if th_out is not None else "",
                    th_in if th_in is not None else "",
                ])

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "combo_segment_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "model",
            "K",
            "segment",
            "actual_avg_OUT_bytes",
            "actual_avg_IN_bytes",
            "avg_active_ms",
            "OUT_MiBps_active",
            "IN_MiBps_active",
            "theory_OUT_bytes",
            "theory_IN_bytes",
        ])
        w.writerows(rows)
    return len(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    n1 = generate_combo_cycle_times()
    n2 = generate_combo_segment_metrics()
    print(f"saved: {os.path.join(OUT_DIR, 'combo_cycle_times.csv')}, rows={n1}")
    print(f"saved: {os.path.join(OUT_DIR, 'combo_segment_metrics.csv')}, rows={n2}")


if __name__ == "__main__":
    main()


