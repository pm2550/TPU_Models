#!/usr/bin/env python3
"""
基于 analyze_usbmon_active.py 的“修复 time_map/对齐”口径，输出你期望的汇总表：
- 传输时间：使用 out_union_ms / in_union_ms
- 传输速率：使用 总bytes / 总time（避免 per-invoke 平均带来的偏差）

关键点（dual 场景）：
- usb:1 的设备(通常 dev4)相对 usb:0 存在 ~704ms 的相位偏移，需要用首个大包 Submit 对齐窗口。
- 为了让 dev3 与 dev4 在同一参考系下可比，这里允许“对齐用设备A、统计用设备B”：
  - 通过 analyze_usbmon_active.py 新增的 USBMON_ALIGN_DEV + REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT=1 实现。
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
ANALYZE = REPO / "analyze_usbmon_active.py"


def _load_invokes(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    return (data.get("per_invoke") or [])[1:]  # skip warmup


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _speed_mibps(bytes_list: List[int], ms_list: List[float]) -> float:
    total_ms = sum(ms_list)
    if total_ms <= 0:
        return 0.0
    return (sum(bytes_list) / (1024 * 1024)) / (total_ms / 1000.0)

def _mean_per_invoke_speed_mibps(bytes_list: List[int], ms_list: List[float], *, min_ms: float = 0.001) -> Tuple[float, int]:
    """
    单次速度 v_i = (bytes_i/MiB) / (ms_i/1000)，然后对 v_i 做算术平均。
    - min_ms: 过滤极小/为0的 union 时长，避免无穷大/异常放大（默认 0.001ms = 1us）。
    返回 (mean_speed, used_count)。
    """
    speeds: List[float] = []
    for b, ms in zip(bytes_list, ms_list):
        if ms is None:
            continue
        ms = float(ms)
        if ms <= min_ms:
            continue
        if b is None:
            continue
        b = int(b)
        if b <= 0:
            continue
        speeds.append((b / (1024 * 1024)) / (ms / 1000.0))
    return (_avg(speeds), len(speeds))


def _summarize(inv: List[Dict[str, Any]]) -> Dict[str, float]:
    invoke_ms = [x["invoke_span_s"] * 1000.0 for x in inv]
    out_ms = [float(x.get("out_union_ms") or 0.0) for x in inv]
    in_ms = [float(x.get("in_union_ms") or 0.0) for x in inv]
    out_b = [int(x.get("bytes_out") or 0) for x in inv]
    in_b = [int(x.get("bytes_in") or 0) for x in inv]

    out_mean_speed, out_mean_n = _mean_per_invoke_speed_mibps(out_b, out_ms)
    in_mean_speed, in_mean_n = _mean_per_invoke_speed_mibps(in_b, in_ms)
    return {
        "n": float(len(inv)),
        "invoke_ms": _avg(invoke_ms),
        "out_mib": _avg([b / (1024 * 1024) for b in out_b]),
        "out_ms": _avg(out_ms),
        # 两种“速度”口径：
        # - out_speed_mean: 单次速度平均（你说的“单次算然后平均”）
        # - out_speed_sum: 总bytes / 总union_time（时间加权的平均吞吐）
        "out_speed_mean": out_mean_speed,
        "out_speed_mean_n": float(out_mean_n),
        "out_speed_sum": _speed_mibps(out_b, out_ms),
        "in_mib": _avg([b / (1024 * 1024) for b in in_b]),
        "in_ms": _avg(in_ms),
        "in_speed_mean": in_mean_speed,
        "in_speed_mean_n": float(in_mean_n),
        "in_speed_sum": _speed_mibps(in_b, in_ms),
    }


def _run_analyze(out_file: Path, usbmon: Path, invokes: Path, time_map: Path, env: Dict[str, str]) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", str(ANALYZE), str(usbmon), str(invokes), str(time_map)]
    res = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})
    if res.returncode != 0:
        raise RuntimeError(f"analyze_usbmon_active failed: {res.stderr.strip() or res.stdout.strip()}")
    out_file.write_text(res.stdout)


def ensure_dual_fixed() -> None:
    # dual_mn7_mn7_sync
    d = ROOT / "results" / "dual_mn7_mn7_sync"
    if d.exists():
        usbmon = d / "usbmon.txt"
        invokes = d / "invokes.json"
        time_map = d / "time_map.json"
        # dev4: usb:1 + dev4 + align dev4
        dev4 = d / "active_analysis_dev4_fixed.json"
        if not dev4.exists():
            _run_analyze(
                dev4, usbmon, invokes, time_map,
                env={
                    "USB_DEVICE": "usb:1",
                    "USBMON_DEV": "4",
                    "USBMON_ALIGN_DEV": "4",
                    "REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT": "1",
                },
            )
        # dev3: usb:0 + dev3，但对齐参考用 dev4（关键！）
        dev3 = d / "active_analysis_dev3_align4.json"
        if not dev3.exists():
            _run_analyze(
                dev3, usbmon, invokes, time_map,
                env={
                    "USB_DEVICE": "usb:0",
                    "USBMON_DEV": "3",
                    "USBMON_ALIGN_DEV": "4",
                    "REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT": "1",
                },
            )

    # dual_deeplab_deeplab_sync
    d = ROOT / "results" / "dual_deeplab_deeplab_sync"
    if d.exists():
        usbmon = d / "usbmon.txt"
        invokes = d / "invokes.json"
        time_map = d / "time_map.json"
        dev4 = d / "active_analysis_dev4_fixed.json"
        if not dev4.exists():
            _run_analyze(
                dev4, usbmon, invokes, time_map,
                env={
                    "USB_DEVICE": "usb:1",
                    "USBMON_DEV": "4",
                    "USBMON_ALIGN_DEV": "4",
                    "REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT": "1",
                },
            )
        dev3 = d / "active_analysis_dev3_align4.json"
        if not dev3.exists():
            _run_analyze(
                dev3, usbmon, invokes, time_map,
                env={
                    "USB_DEVICE": "usb:0",
                    "USBMON_DEV": "3",
                    "USBMON_ALIGN_DEV": "4",
                    "REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT": "1",
                },
            )

    # dual_deeplab_mn7_sync (mixed)
    d = ROOT / "results" / "dual_deeplab_mn7_sync"
    if d.exists():
        usbmon = d / "usbmon.txt"
        invokes = d / "invokes.json"
        time_map = d / "time_map.json"
        # usb:1 侧（通常 dev4）存在相位偏移，先修复它（对齐用 dev4）
        dev4 = d / "active_analysis_dev4_fixed.json"
        if not dev4.exists():
            _run_analyze(
                dev4, usbmon, invokes, time_map,
                env={
                    "USB_DEVICE": "usb:1",
                    "USBMON_DEV": "4",
                    "USBMON_ALIGN_DEV": "4",
                    "REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT": "1",
                },
            )
        # usb:0 侧（dev3）为了与 usb:1 同参考系，这里也用 dev4 作为对齐参考
        dev3 = d / "active_analysis_dev3_align4.json"
        if not dev3.exists():
            _run_analyze(
                dev3, usbmon, invokes, time_map,
                env={
                    "USB_DEVICE": "usb:0",
                    "USBMON_DEV": "3",
                    "USBMON_ALIGN_DEV": "4",
                    "REBASE_WINDOWS_BY_FIRST_BIG_SUBMIT": "1",
                },
            )


def main() -> None:
    ensure_dual_fixed()

    # MN7
    mn7_single = _summarize(_load_invokes(ROOT / "results" / "baseline_single_mn7" / "active_analysis_aligned.json"))
    mn7_dev3 = _summarize(_load_invokes(ROOT / "results" / "dual_mn7_mn7_sync" / "active_analysis_dev3_align4.json"))
    mn7_dev4 = _summarize(_load_invokes(ROOT / "results" / "dual_mn7_mn7_sync" / "active_analysis_dev4_fixed.json"))

    # DeepLab
    dl_single = _summarize(_load_invokes(ROOT / "results" / "baseline_single_deeplab" / "active_analysis_aligned.json"))
    dl_dev3 = _summarize(_load_invokes(ROOT / "results" / "dual_deeplab_deeplab_sync" / "active_analysis_dev3_align4.json"))
    dl_dev4 = _summarize(_load_invokes(ROOT / "results" / "dual_deeplab_deeplab_sync" / "active_analysis_dev4_fixed.json"))

    # Mixed (DeepLab + MN7)
    mixed_dir = ROOT / "results" / "dual_deeplab_mn7_sync"
    mixed_dev3 = _summarize(_load_invokes(mixed_dir / "active_analysis_dev3_align4.json"))
    mixed_dev4 = _summarize(_load_invokes(mixed_dir / "active_analysis_dev4_fixed.json"))

    def fmt(x: float) -> str:
        return f"{x:.6g}"

    rows: List[Tuple[str, str, Dict[str, float]]] = [
        ("MN7", "单TPU", mn7_single),
        ("MN7", "双TPU dev3", mn7_dev3),
        ("MN7", "双TPU dev4", mn7_dev4),
        ("DeepLab", "单TPU", dl_single),
        ("DeepLab", "双TPU dev3", dl_dev3),
        ("DeepLab", "双TPU dev4", dl_dev4),
        ("Mixed", "dev3(usb:0)", mixed_dev3),
        ("Mixed", "dev4(usb:1)", mixed_dev4),
    ]

    # 全量列（统一口径，TSV 输出，便于你直接复制到表格软件）
    header = [
        "model",
        "scene",
        "n_invokes",
        "invoke_ms_avg",
        "out_mib_avg",
        "out_union_ms_avg",
        "out_speed_mean_mibps",
        "out_speed_mean_used_n",
        "out_speed_sum_mibps",
        "in_mib_avg",
        "in_union_ms_avg",
        "in_speed_mean_mibps",
        "in_speed_mean_used_n",
        "in_speed_sum_mibps",
    ]
    print("\t".join(header))
    for model, scene, s in rows:
        print(
            "\t".join(
                [
                    model,
                    scene,
                    fmt(s.get("n", 0.0)),
                    fmt(s.get("invoke_ms", 0.0)),
                    fmt(s.get("out_mib", 0.0)),
                    fmt(s.get("out_ms", 0.0)),
                    fmt(s.get("out_speed_mean", 0.0)),
                    fmt(s.get("out_speed_mean_n", 0.0)),
                    fmt(s.get("out_speed_sum", 0.0)),
                    fmt(s.get("in_mib", 0.0)),
                    fmt(s.get("in_ms", 0.0)),
                    fmt(s.get("in_speed_mean", 0.0)),
                    fmt(s.get("in_speed_mean_n", 0.0)),
                    fmt(s.get("in_speed_sum", 0.0)),
                ]
            )
        )


if __name__ == "__main__":
    main()


