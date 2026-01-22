#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对四模型的周期/突发推理实验：
1. 连续无周期约束
2. 固定周期 50/100/150 ms
3. 连续推理 + 周期性空闲

脚本会：
- 针对选定模型运行三类场景，记录冷/热启动时间
- 可选锁定 CPU 频率，并将当前进程 Pin 到指定 CPU 核
- 生成每个场景的 warm invoke 直方图与周期（cycle）分布图
- 产出 JSON 汇总，便于与基础单模型平均耗时对比

依赖：psutil、matplotlib、numpy、torch、pycoral
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import psutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure_four_models import MODELS, test_cpu_netvlad, test_tpu_model

DEFAULT_OUTPUT_DIR = Path("/home/10210/Desktop/OS/results/burst_tests")
DEFAULT_PERIODS_MS = [50, 100, 150]
DEFAULT_BURST_SIZE = 20
DEFAULT_IDLE_MS = 500.0
DEFAULT_TPU_MAP = {
    # 默认：deeplab 放 TPU1，其它 TPU 模型放 TPU0；netvlad_head 走 CPU（自动 unset）
    "ssd_mobilenet_v2": "usb:0",
    "ssd_mobilenet_v2_joint": "usb:0",
    "mobilenet_v2": "usb:0",
    "mobilenet_v2_joint": "usb:0",
    "deeplabv3_dm05": "usb:1",
}
PASSWORD_FILE = Path("/home/10210/Desktop/OS/password.text")

_PASSWORD_CACHE: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四模型周期/突发推理实验")
    parser.add_argument(
        "--workloads",
        type=str,
        nargs="+",
        default=["w3"],
        choices=["w1", "w2", "w3", "all"],
        help=(
            "选择 workload 预设："
            "w1=单TPU(全部 TPU 模型用 usb:0), "
            "w2=分摊(检测+deeplab 用 usb:0, mobilenet 用 usb:1), "
            "w3=deeplab@usb:1，其它@usb:0（你要求的），"
            "all=依次跑全部。"
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        choices=list(MODELS.keys()) + ["all"],
        help="选择要测试的模型（默认 all）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="结果输出目录",
    )
    parser.add_argument(
        "--periods",
        type=float,
        nargs="+",
        default=DEFAULT_PERIODS_MS,
        help="周期场景的目标周期 (ms)",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=DEFAULT_BURST_SIZE,
        help="突发 + 空闲 场景中连续推理次数",
    )
    parser.add_argument(
        "--idle-ms",
        type=float,
        default=DEFAULT_IDLE_MS,
        help="突发 + 空闲 场景的空闲时长 (ms)",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=60,
        help="热运行采样次数",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="正式采样前的预热次数",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        nargs="+",
        help="分别为每个模型指定 CPU 核 (长度需与模型数一致)",
    )
    parser.add_argument(
        "--lock-freq",
        type=str,
        help="锁定指定 CPU 核的频率 (例如 1.80GHz)",
    )
    parser.add_argument(
        "--tpu-map",
        type=str,
        nargs="*",
        help="指定模型与 EdgeTPU 设备映射，如 ssd_mobilenet_v2=usb:0",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="仅输出 JSON，不生成图表",
    )
    args = parser.parse_args()

    if "all" in args.models:
        # 默认“全套四模型”定义：SSD + DeepLab + MobileNetV2(backbone) + NetVLAD head(CPU)
        # （不默认包含 EfficientDet）
        args.models = [
            "ssd_mobilenet_v2",
            "deeplabv3_dm05",
            "mobilenet_v2",
            "netvlad_head",
        ]

    if args.cpu_cores and len(args.cpu_cores) != len(args.models):
        parser.error("--cpu-cores 长度必须与模型数量一致")

    if any(p <= 0 for p in args.periods):
        parser.error("周期必须为正数")

    return args


def workload_device_maps() -> Dict[str, Dict[str, str]]:
    """三种 workload 的模型->TPU 设备映射。"""
    # 仅对 TPU 模型生效；netvlad_head 会自动走 CPU
    w1 = {
        "ssd_mobilenet_v2": "usb:0",
        "ssd_mobilenet_v2_joint": "usb:0",
        "mobilenet_v2": "usb:0",
        "mobilenet_v2_joint": "usb:0",
        "deeplabv3_dm05": "usb:0",
    }
    w2 = {
        "ssd_mobilenet_v2": "usb:0",
        "ssd_mobilenet_v2_joint": "usb:0",
        "deeplabv3_dm05": "usb:0",
        "mobilenet_v2": "usb:1",
        "mobilenet_v2_joint": "usb:1",
    }
    # 你要求的：deeplab tpu1，其它两个 TPU 模型 tpu0；netvlad 用 CPU
    w3 = {
        "ssd_mobilenet_v2": "usb:0",
        "ssd_mobilenet_v2_joint": "usb:0",
        "mobilenet_v2": "usb:0",
        "mobilenet_v2_joint": "usb:0",
        "deeplabv3_dm05": "usb:1",
    }
    return {"w1": w1, "w2": w2, "w3": w3}


def apply_cpu_affinity(core: Optional[int]) -> None:
    if core is None:
        return
    proc = psutil.Process()
    try:
        proc.cpu_affinity([core])
    except AttributeError:
        os.sched_setaffinity(0, {core})


def try_lock_frequency(cores: List[int], freq: str) -> None:
    password = None if os.geteuid() == 0 else get_sudo_password()
    for core in cores:
        base_cmd = ["cpufreq-set", "-c", str(core), "-f", freq]
        if os.geteuid() == 0:
            cmd = base_cmd
            input_data = None
        elif password:
            cmd = ["sudo", "-S", *base_cmd]
            input_data = password + "\n"
        else:
            print(
                f"[warn] 需要锁频但无法读取密码文件；请手动运行: sudo {' '.join(base_cmd)}"
            )
            continue

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                input=input_data,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            printable_cmd = " ".join(cmd if input_data is None else ["sudo", *base_cmd])
            print(f"[warn] 无法锁定 CPU{core} 频率: {exc}\n       请手动执行: {printable_cmd}")


def parse_tpu_map(overrides: Optional[List[str]]) -> Dict[str, str]:
    mapping = DEFAULT_TPU_MAP.copy()
    if not overrides:
        return mapping
    for item in overrides:
        if "=" not in item:
            print(f"[warn] 忽略无效的 --tpu-map 项: {item}")
            continue
        model, device = item.split("=", 1)
        model = model.strip()
        device = device.strip()
        if model not in MODELS:
            print(f"[warn] 未知模型 {model}，忽略映射")
            continue
        mapping[model] = device
    return mapping


def set_tpu_device(model_key: str, device_map: Dict[str, str]) -> None:
    if model_key == "netvlad_head":
        os.environ.pop("EDGETPU_DEVICE", None)
        return
    device = device_map.get(model_key)
    if device:
        os.environ["EDGETPU_DEVICE"] = device
    else:
        os.environ.pop("EDGETPU_DEVICE", None)


def get_sudo_password() -> Optional[str]:
    global _PASSWORD_CACHE
    if _PASSWORD_CACHE is not None:
        return _PASSWORD_CACHE
    try:
        pw = PASSWORD_FILE.read_text(encoding="utf-8").strip()
    except OSError as exc:
        print(f"[warn] 无法读取密码文件 {PASSWORD_FILE}: {exc}")
        return None
    if not pw:
        print(f"[warn] 密码文件 {PASSWORD_FILE} 为空")
        return None
    _PASSWORD_CACHE = pw
    return pw


def save_histogram(data: List[float], output_path: Path, xlabel: str, title: str) -> None:
    if not data:
        return
    plt.figure(figsize=(7, 4))
    plt.hist(data, bins="auto", edgecolor="black", alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def annotate_vs_base(result: Dict, base_avg: float) -> None:
    warm_avg = result.get("statistics", {}).get("warm_avg_ms", 0.0)
    if warm_avg:
        result.setdefault("statistics", {})["delta_vs_base_ms"] = warm_avg - base_avg
        result["statistics"]["ratio_vs_base"] = warm_avg / base_avg if base_avg else 0.0
    if "cycle_ms" in result and result["cycle_ms"]:
        cycle_avg = float(np.mean(result["cycle_ms"]))
        result.setdefault("statistics", {})["cycle_delta_vs_base_ms"] = cycle_avg - base_avg


def main() -> None:
    args = parse_args()
    base_output_dir: Path = args.output_dir
    base_output_dir.mkdir(parents=True, exist_ok=True)

    proc = psutil.Process()
    try:
        original_affinity = proc.cpu_affinity()
    except AttributeError:
        original_affinity = None

    cpu_core_map = dict(zip(args.models, args.cpu_cores)) if args.cpu_cores else {}
    device_map_cli = parse_tpu_map(args.tpu_map)

    # workload 选择
    wl_maps = workload_device_maps()
    workloads = args.workloads
    if "all" in workloads:
        workloads = ["w1", "w2", "w3"]

    if args.lock_freq:
        if args.cpu_cores:
            try_lock_frequency(sorted(set(args.cpu_cores)), args.lock_freq)
        else:
            print("[warn] 指定了 --lock-freq 但没有 --cpu-cores，跳过自动锁频")

    for wl in workloads:
        output_dir: Path = base_output_dir / wl
        output_dir.mkdir(parents=True, exist_ok=True)

        # 以 workload 预设为基础，再叠加 CLI 覆盖（--tpu-map 优先生效）
        device_map = DEFAULT_TPU_MAP.copy()
        device_map.update(wl_maps.get(wl, {}))
        device_map.update(device_map_cli or {})

        summary = {
            "metadata": {
                "workload": wl,
                "periods_ms": args.periods,
                "burst_size": args.burst_size,
                "idle_ms": args.idle_ms,
                "warm_repeats": args.warm_repeats,
                "warmup": args.warmup,
                "cpu_cores": args.cpu_cores,
                "lock_freq": args.lock_freq,
                "tpu_map": device_map,
            },
            "models": {},
        }

        for model_key in args.models:
            model_path = MODELS[model_key]
            summary["models"][model_key] = {
                "path": model_path,
                "scenarios": {},
            }

            if model_key in cpu_core_map:
                apply_cpu_affinity(cpu_core_map[model_key])

            set_tpu_device(model_key, device_map)

            # 场景 1：连续推理
            continuous_result = (
                test_cpu_netvlad(
                    model_path,
                    model_key,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup,
                    capture_cycle=True,
                )
                if model_key == "netvlad_head"
                else test_tpu_model(
                    model_path,
                    model_key,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup,
                    use_tpu=True,
                    capture_cycle=True,
                )
            )

            summary["models"][model_key]["base_avg_ms"] = continuous_result["statistics"]["warm_avg_ms"]
            scenario_dir = output_dir / model_key
            scenario_dir.mkdir(exist_ok=True, parents=True)

            plots = {}
            if not args.skip_plots:
                warm_hist_path = scenario_dir / "continuous_warm_hist.png"
                cycle_hist_path = scenario_dir / "continuous_cycle_hist.png"
                save_histogram(
                    continuous_result.get("warm_run_ms", []),
                    warm_hist_path,
                    "Invoke Time (ms)",
                    f"{model_key} Continuous Warm Invoke",
                )
                save_histogram(
                    continuous_result.get("cycle_ms", []),
                    cycle_hist_path,
                    "Cycle Time (ms)",
                    f"{model_key} Continuous Cycle",
                )
                plots = {
                    "warm_hist": str(warm_hist_path),
                    "cycle_hist": str(cycle_hist_path),
                }

            summary["models"][model_key]["scenarios"]["continuous"] = {
                "result": continuous_result,
                "plots": plots,
            }

            base_avg = continuous_result["statistics"]["warm_avg_ms"]

            # 场景 2：固定周期
            periodic_results = {}
            for period_ms in args.periods:
                set_tpu_device(model_key, device_map)
                res = (
                    test_cpu_netvlad(
                        model_path,
                        model_key,
                        warm_repeats=args.warm_repeats,
                        warmup=args.warmup,
                        sleep_between_ms=period_ms,
                        capture_cycle=True,
                    )
                    if model_key == "netvlad_head"
                    else test_tpu_model(
                        model_path,
                        model_key,
                        warm_repeats=args.warm_repeats,
                        warmup=args.warmup,
                        use_tpu=True,
                        sleep_between_ms=period_ms,
                        capture_cycle=True,
                    )
                )
                annotate_vs_base(res, base_avg)

                plot_dict = {}
                if not args.skip_plots:
                    warm_hist_path = scenario_dir / f"periodic_{int(period_ms)}ms_warm_hist.png"
                    cycle_hist_path = scenario_dir / f"periodic_{int(period_ms)}ms_cycle_hist.png"
                    save_histogram(
                        res.get("warm_run_ms", []),
                        warm_hist_path,
                        "Invoke Time (ms)",
                        f"{model_key} Period {period_ms} ms Warm",
                    )
                    save_histogram(
                        res.get("cycle_ms", []),
                        cycle_hist_path,
                        "Cycle Time (ms)",
                        f"{model_key} Period {period_ms} ms Cycle",
                    )
                    plot_dict = {
                        "warm_hist": str(warm_hist_path),
                        "cycle_hist": str(cycle_hist_path),
                    }

                periodic_results[f"{period_ms}ms"] = {
                    "result": res,
                    "plots": plot_dict,
                }

            summary["models"][model_key]["scenarios"]["periodic"] = periodic_results

            # 场景 3：突发 + 空闲
            set_tpu_device(model_key, device_map)
            burst_result = (
                test_cpu_netvlad(
                    model_path,
                    model_key,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup,
                    idle_every=args.burst_size,
                    idle_duration_ms=args.idle_ms,
                    capture_cycle=True,
                )
                if model_key == "netvlad_head"
                else test_tpu_model(
                    model_path,
                    model_key,
                    warm_repeats=args.warm_repeats,
                    warmup=args.warmup,
                    use_tpu=True,
                    idle_every=args.burst_size,
                    idle_duration_ms=args.idle_ms,
                    capture_cycle=True,
                )
            )
            annotate_vs_base(burst_result, base_avg)

            burst_plots = {}
            if not args.skip_plots:
                warm_hist_path = scenario_dir / "burst_idle_warm_hist.png"
                cycle_hist_path = scenario_dir / "burst_idle_cycle_hist.png"
                save_histogram(
                    burst_result.get("warm_run_ms", []),
                    warm_hist_path,
                    "Invoke Time (ms)",
                    f"{model_key} Burst Warm",
                )
                save_histogram(
                    burst_result.get("cycle_ms", []),
                    cycle_hist_path,
                    "Cycle Time (ms)",
                    f"{model_key} Burst Cycle",
                )
                burst_plots = {
                    "warm_hist": str(warm_hist_path),
                    "cycle_hist": str(cycle_hist_path),
                }

            summary["models"][model_key]["scenarios"]["burst_idle"] = {
                "result": burst_result,
                "plots": burst_plots,
            }

        summary_path = output_dir / "burst_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    if original_affinity is not None:
        try:
            proc.cpu_affinity(original_affinity)
        except AttributeError:
            pass

    print(f"\n✅ 全部 workload 实验完成，结果目录: {base_output_dir}")
    if not args.skip_plots:
        print(f"图表输出目录: {base_output_dir}")


if __name__ == "__main__":
    main()
