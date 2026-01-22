#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按“图里的 workload”执行的并行 cycle runner（解决你说的 0.7s 启动相位差 + pin/锁频）。

核心思路：
- TPU0/TPU1/CPU 各自一个常驻进程（提前 create/allocate + warmup）
- 一个 orchestrator 进程按图里的**顺序/并行关系**驱动每个 cycle：
  - 同一 cycle 内：SSD@TPU0 与 DeepLab@TPU1 并行触发
  - 然后（可选）执行 CPU 阶段（odometry/SLAM/融合/规划/MPC 的占位或固定耗时）
  - “低置信度”才触发 NetVLAD（CPU）阶段
- 默认 sync-start：所有 worker ready 后，统一设置 start_time，之后周期调度按绝对时间对齐。

输出：
- /home/10210/Desktop/OS/results/workload_cycles/<workload>/<timestamp>/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
OUT_BASE = Path("/home/10210/Desktop/OS/results/workload_cycles")

# Reuse model registry from measure_four_models
from measure_four_models import MODELS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="并行工作流 cycle 测试（sync-start + pin + 锁频）")
    p.add_argument("--workload", choices=["w1", "w2", "w3"], default="w3")
    p.add_argument("--use-joint", action="store_true", help="优先使用 *_joint 模型（若存在）")

    p.add_argument("--tpu0-device", type=str, default="usb:0")
    p.add_argument("--tpu1-device", type=str, default="usb:1")

    p.add_argument("--tpu0-core", type=int, default=2)
    p.add_argument("--tpu1-core", type=int, default=3)
    p.add_argument("--cpu-core", type=int, default=-1, help="NetVLAD CPU worker 的 pin 核；-1 表示不 pin")
    p.add_argument("--cpu-no-pin", action="store_true", help="NetVLAD 不设置 CPU affinity（允许其使用所有核）")
    p.add_argument("--lock-freq", type=str, default="", help="锁频，例如 1800000 或 1.80GHz（依赖 cpufreq-set）")

    p.add_argument("--warmup", type=int, default=20, help="每个模型 warmup invoke 次数（不计时）")
    p.add_argument("--cycles", type=int, default=120, help="cycle 次数（continuous/periodic/burst 都用这个上限）")

    # 3 种模式
    p.add_argument("--mode", choices=["continuous", "periodic", "burst_idle"], default="continuous")
    p.add_argument("--period-ms", type=float, default=50.0, help="periodic 模式周期(ms)")
    p.add_argument("--burst-size", type=int, default=20, help="burst_idle：每 burst 的 cycle 数")
    p.add_argument("--idle-ms", type=float, default=500.0, help="burst_idle：每个 burst 后 idle 时长(ms)")

    # CPU stages（按图占位；不想模拟就保持 0）
    p.add_argument("--cpu-odom-ms", type=float, default=0.0, help="Short-term odometry CPU 耗时占位(ms)")
    p.add_argument("--cpu-slam-ms", type=float, default=0.0, help="SLAM/Loop closure CPU 耗时占位(ms)")
    p.add_argument("--cpu-fusion-ms", type=float, default=0.0, help="Obstacle fusion/tracking CPU 耗时占位(ms)")
    p.add_argument("--cpu-mpc-ms", type=float, default=0.0, help="MPC/Stanley control CPU 耗时占位(ms)")

    # NetVLAD 触发策略（按图：低置信度触发；这里提供可复现实验的策略）
    p.add_argument("--netvlad-policy", choices=["never", "always", "every_n", "prob"], default="every_n")
    p.add_argument("--netvlad-every-n", type=int, default=10, help="policy=every_n 时，每 N 个 cycle 触发一次")
    p.add_argument("--netvlad-prob", type=float, default=0.1, help="policy=prob 时，触发概率 [0,1]")

    p.add_argument("--output-dir", type=Path, default=OUT_BASE)
    return p.parse_args()


def apply_cpu_affinity(core: int) -> None:
    if core is None or core < 0:
        return
    # clamp to available cores to avoid crash
    try:
        max_cpu = (os.cpu_count() or 1) - 1
    except Exception:
        max_cpu = 0
    if core < 0:
        core = 0
    if core > max_cpu:
        core = max_cpu
    proc = psutil.Process()
    try:
        proc.cpu_affinity([core])
    except AttributeError:
        os.sched_setaffinity(0, {core})


def try_lock_frequency(cores: List[int], freq: str) -> None:
    if not freq:
        return
    for c in sorted(set(cores)):
        try:
            subprocess.run(["sudo", "-n", "cpufreq-set", "-c", str(c), "-f", str(freq)], check=False)
        except Exception:
            pass


def _make_dummy_input(inp_detail: dict) -> np.ndarray:
    shape = inp_detail["shape"]
    dtype = inp_detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        return np.random.randint(0, 256, size=shape, dtype=dtype)
    return np.random.random_sample(shape).astype(dtype)


def _create_tpu_interpreter(model_path: str):
    # prefer pycoral if available, fallback to tflite_runtime delegate
    try:
        from pycoral.utils.edgetpu import make_interpreter as coral_make  # type: ignore

        it = coral_make(model_path)
    except Exception:
        from tflite_runtime.interpreter import Interpreter, load_delegate

        dev = os.environ.get("EDGETPU_DEVICE")
        delegate = load_delegate("libedgetpu.so.1", {"device": dev} if dev else {})
        it = Interpreter(model_path=model_path, experimental_delegates=[delegate], num_threads=1)
    it.allocate_tensors()
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    dummy = _make_dummy_input(inp)
    return it, inp, out, dummy


def _invoke_once(it, inp, out, dummy) -> float:
    it.set_tensor(inp["index"], dummy)
    t0 = time.perf_counter()
    it.invoke()
    t1 = time.perf_counter()
    _ = it.get_tensor(out["index"])
    return (t1 - t0) * 1000.0


def _cpu_netvlad_forward(checkpoint_path: str):
    import torch
    import numpy as np

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    n_clusters = 64
    descriptor_dim = 1280
    conv_weights = torch.randn(n_clusters, descriptor_dim, 1, 1)
    cluster_centers = [torch.randn(descriptor_dim, 1) for _ in range(n_clusters)]
    input_features = torch.randn(1, 1280, 7, 7)

    def forward_once():
        soft_assign = torch.nn.functional.conv2d(input_features, conv_weights)
        soft_assign = torch.nn.functional.softmax(soft_assign, dim=1)
        feature_flat = input_features.view(1, descriptor_dim, -1)
        soft_assign_flat = soft_assign.view(1, n_clusters, -1)
        residuals = []
        for k in range(n_clusters):
            residual = (feature_flat - cluster_centers[k]) * soft_assign_flat[:, k : k + 1, :]
            residuals.append(residual.sum(dim=2))
        vlad = torch.cat(residuals, dim=1)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)
        if "WPCA" in checkpoint:
            wpca_matrix = checkpoint["WPCA"]
            if isinstance(wpca_matrix, np.ndarray):
                wpca_matrix = torch.from_numpy(wpca_matrix).float()
            if wpca_matrix.shape[0] == vlad.shape[1]:
                final_descriptor = torch.matmul(vlad, wpca_matrix.t())
            else:
                final_descriptor = vlad[:, :512]
        else:
            final_descriptor = vlad[:, :512]
        return torch.nn.functional.normalize(final_descriptor, p=2, dim=1)

    return forward_once


@dataclass
class WorkerResult:
    name: str
    core: int
    device: str
    mode: str
    cycles: int
    per_cycle_ms: List[float]
    per_stage_ms: Dict[str, List[float]]


def _schedule_wait(mode: str, start_t: float, k: int, period_s: float, burst_size: int, idle_s: float):
    if mode == "continuous":
        return
    if mode == "periodic":
        target = start_t + k * period_s
        now = time.perf_counter()
        if target > now:
            time.sleep(target - now)
        return
    # burst_idle: cycles free-run inside burst, then idle
    if mode == "burst_idle":
        if burst_size > 0 and (k > 0) and (k % burst_size == 0):
            time.sleep(idle_s)

def tpu_service(
    name: str,
    core: int,
    device: str,
    models: List[Tuple[str, str]],
    warmup: int,
    conn,
):
    """常驻 TPU 服务：等待 orchestrator 发来 invoke 请求，返回耗时。

    支持同一 TPU 进程内加载多个模型（例如 TPU0: SSD + MobileNetV2）。
    """
    apply_cpu_affinity(core)
    os.environ["EDGETPU_DEVICE"] = device

    interpreters: Dict[str, Tuple[Any, dict, dict, np.ndarray]] = {}
    for model_key, model_path in models:
        it, inp, out, dummy = _create_tpu_interpreter(model_path)
        for _ in range(max(0, warmup)):
            _ = _invoke_once(it, inp, out, dummy)
        interpreters[model_key] = (it, inp, out, dummy)

    # handshake: ready -> wait start
    conn.send({"op": "ready", "name": name, "device": device, "core": core, "models": list(interpreters.keys())})
    start_msg = conn.recv()
    if not (isinstance(start_msg, dict) and start_msg.get("op") == "start"):
        return

    while True:
        msg = conn.recv()
        if msg is None or msg.get("op") == "stop":
            break
        if msg.get("op") == "invoke":
            mkey = msg.get("model")
            if mkey not in interpreters:
                conn.send({"ok": False, "err": f"unknown model {mkey}", "model": mkey})
                continue
            it, inp, out, dummy = interpreters[mkey]
            try:
                ms = _invoke_once(it, inp, out, dummy)
                conn.send({"ok": True, "ms": ms, "model": mkey})
            except Exception as e:
                conn.send({"ok": False, "err": str(e), "model": mkey})
        else:
            conn.send({"ok": False, "err": f"unknown op {msg.get('op')}"})


def cpu_service(
    core: int,
    checkpoint_path: str,
    warmup: int,
    conn,
):
    """常驻 CPU 服务：只提供 netvlad invoke（按需触发）。"""
    # 若 core < 0，则不设置 affinity（允许使用所有核）
    apply_cpu_affinity(core)
    os.environ.pop("EDGETPU_DEVICE", None)
    forward = _cpu_netvlad_forward(checkpoint_path)
    for _ in range(max(0, warmup)):
        forward()

    conn.send({"op": "ready", "name": "cpu", "device": "cpu", "core": core, "models": ["netvlad_head"]})
    start_msg = conn.recv()
    if not (isinstance(start_msg, dict) and start_msg.get("op") == "start"):
        return

    while True:
        msg = conn.recv()
        if msg is None or msg.get("op") == "stop":
            break
        if msg.get("op") == "netvlad":
            try:
                t0 = time.perf_counter()
                forward()
                t1 = time.perf_counter()
                conn.send({"ok": True, "ms": (t1 - t0) * 1000.0, "model": "netvlad_head"})
            except Exception as e:
                conn.send({"ok": False, "err": str(e), "model": "netvlad_head"})
        else:
            conn.send({"ok": False, "err": f"unknown op {msg.get('op')}"})


def pick_models(args: argparse.Namespace) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], str]:
    # w3 默认：TPU0=SSD+MobileNetV2, TPU1=DeepLab
    def k(base: str) -> str:
        if args.use_joint and f"{base}_joint" in MODELS:
            return f"{base}_joint"
        return base

    # 注意：这里的“workload”指设备分配；执行顺序由 orchestrator 按图驱动。
    if args.workload == "w1":
        # 单TPU：都放 TPU0（TPU1 不用）
        tpu0 = [(k("ssd_mobilenet_v2"), MODELS[k("ssd_mobilenet_v2")]), ("deeplabv3_dm05", MODELS["deeplabv3_dm05"])]
        tpu1 = []
    elif args.workload == "w2":
        # 分摊：SSD+DeepLab->TPU0, MobileNetV2(backbone)->TPU1（如果你后面要加）
        tpu0 = [(k("ssd_mobilenet_v2"), MODELS[k("ssd_mobilenet_v2")]), ("deeplabv3_dm05", MODELS["deeplabv3_dm05"])]
        tpu1 = []
    else:
        # 你当前主诉求：TPU0=SSD，TPU1=DeepLab
        # 同时在 TPU0 里加载第二个模型（MobileNetV2 backbone），用于“另外两个模型 TPU0”
        tpu0 = [
            (k("ssd_mobilenet_v2"), MODELS[k("ssd_mobilenet_v2")]),
            (k("mobilenet_v2"), MODELS[k("mobilenet_v2")]),
        ]
        tpu1 = [("deeplabv3_dm05", MODELS["deeplabv3_dm05"])]

    cpu_ckpt = MODELS["netvlad_head"]
    return tpu0, tpu1, cpu_ckpt


def summarize_stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {}
    arr = np.array(xs, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> None:
    args = parse_args()
    try_lock_frequency([args.tpu0_core, args.tpu1_core, args.cpu_core], args.lock_freq)

    tpu0_models, tpu1_models, cpu_ckpt = pick_models(args)
    mode = args.mode
    period_s = max(0.0, args.period_ms / 1000.0)
    idle_s = max(0.0, args.idle_ms / 1000.0)

    import multiprocessing as mp
    import random

    mp.set_start_method("fork", force=True)

    p0_parent, p0_child = mp.Pipe()
    p1_parent, p1_child = mp.Pipe()
    pc_parent, pc_child = mp.Pipe()

    if not tpu0_models or len(tpu0_models) < 2:
        raise RuntimeError("TPU0 模型不足（w3 需要 SSD+MobileNetV2）")
    if not tpu1_models:
        raise RuntimeError("TPU1 缺少 DeepLab（请用 --workload w3）")

    p0 = mp.Process(target=tpu_service, args=("tpu0", args.tpu0_core, args.tpu0_device, tpu0_models, args.warmup, p0_child))
    p1 = mp.Process(target=tpu_service, args=("tpu1", args.tpu1_core, args.tpu1_device, tpu1_models, args.warmup, p1_child))
    cpu_core = -1 if args.cpu_no_pin else args.cpu_core
    pc = mp.Process(target=cpu_service, args=(cpu_core, cpu_ckpt, args.warmup, pc_child))

    p0.start(); p1.start(); pc.start()

    # handshake: wait ready from all services, then send start (sync-start)
    ready0 = p0_parent.recv()
    ready1 = p1_parent.recv()
    readyc = pc_parent.recv()
    if not (ready0.get("op") == ready1.get("op") == readyc.get("op") == "ready"):
        raise RuntimeError(f"服务初始化失败: {ready0}, {ready1}, {readyc}")
    start_t = time.perf_counter() + 0.05
    for c in (p0_parent, p1_parent, pc_parent):
        c.send({"op": "start", "t0": start_t})

    # orchestrator per-cycle
    period_s = max(0.0, args.period_ms / 1000.0)
    idle_s = max(0.0, args.idle_ms / 1000.0)

    stages: Dict[str, List[float]] = {
        "tpu0_ssd": [],
        "tpu1_deeplab": [],
        # NetVLAD 的 TPU backbone（mobilenet）与 CPU head（netvlad）存在先后依赖：
        # 仅在触发 NetVLAD 时才执行并记录（未触发时填 0）。
        "tpu0_mobilenet": [],
        "cpu_odometry": [],
        "cpu_slam": [],
        "cpu_fusion": [],
        "cpu_mpc": [],
        "cpu_netvlad": [],
    }
    cycle_ms: List[float] = []
    netvlad_triggered: List[int] = []

    def should_netvlad(k: int) -> bool:
        if args.netvlad_policy == "never":
            return False
        if args.netvlad_policy == "always":
            return True
        if args.netvlad_policy == "every_n":
            n = max(1, int(args.netvlad_every_n))
            return (k % n) == 0 and k > 0
        # prob
        p = float(args.netvlad_prob)
        return random.random() < max(0.0, min(1.0, p))

    # recv 超时（秒）：避免某个子进程崩溃后 orchestrator 永远阻塞
    recv_timeout_s = 5.0

    for k in range(args.cycles):
        _schedule_wait(mode, start_t, k, period_s, args.burst_size, idle_s)
        t_cycle0 = time.perf_counter()

        # 按图：SSD@TPU0 与 DeepLab@TPU1 并行触发
        ssd_key, _ = tpu0_models[0]
        mn_key, _ = tpu0_models[1]
        dl_key, _ = tpu1_models[0]

        p0_parent.send({"op": "invoke", "model": ssd_key})
        p1_parent.send({"op": "invoke", "model": dl_key})
        if not p0_parent.poll(recv_timeout_s) or not p1_parent.poll(recv_timeout_s):
            # hard failure: stop early
            break
        r0 = p0_parent.recv()
        r1 = p1_parent.recv()
        stages["tpu0_ssd"].append(float(r0.get("ms", 0.0)) if r0.get("ok") else 0.0)
        stages["tpu1_deeplab"].append(float(r1.get("ms", 0.0)) if r1.get("ok") else 0.0)

        # CPU stages（占位：严格顺序）
        for key, dur_ms in [
            ("cpu_odometry", args.cpu_odom_ms),
            ("cpu_slam", args.cpu_slam_ms),
            ("cpu_fusion", args.cpu_fusion_ms),
            ("cpu_mpc", args.cpu_mpc_ms),
        ]:
            t0 = time.perf_counter()
            if dur_ms and dur_ms > 0:
                time.sleep(float(dur_ms) / 1000.0)
            t1 = time.perf_counter()
            stages[key].append((t1 - t0) * 1000.0)

        # 低置信度触发 NetVLAD（这里用 policy 近似）
        # 严格依赖：先跑 mobilenet(backbone) 生成特征，再跑 netvlad_head
        if should_netvlad(k):
            netvlad_triggered.append(k)
            # 1) NetVLAD backbone（TPU0 mobilenet）
            p0_parent.send({"op": "invoke", "model": mn_key})
            if not p0_parent.poll(recv_timeout_s):
                break
            r2 = p0_parent.recv()
            stages["tpu0_mobilenet"].append(float(r2.get("ms", 0.0)) if r2.get("ok") else 0.0)
            # 2) NetVLAD head（CPU）
            pc_parent.send({"op": "netvlad"})
            if not pc_parent.poll(recv_timeout_s):
                break
            rr = pc_parent.recv()
            stages["cpu_netvlad"].append(float(rr.get("ms", 0.0)) if rr.get("ok") else 0.0)
        else:
            stages["tpu0_mobilenet"].append(0.0)
            stages["cpu_netvlad"].append(0.0)

        t_cycle1 = time.perf_counter()
        cycle_ms.append((t_cycle1 - t_cycle0) * 1000.0)

    # stop services
    for c in (p0_parent, p1_parent, pc_parent):
        try:
            c.send({"op": "stop"})
        except Exception:
            pass
    p0.join(timeout=2); p1.join(timeout=2); pc.join(timeout=2)

    # output
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / args.workload / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"

    actual_cycles = len(cycle_ms)
    miss = {}
    if mode == "periodic" and args.period_ms > 0:
        miss = {
            "deadline_ms": args.period_ms,
            "miss_count": int(sum(1 for x in cycle_ms if x > args.period_ms)),
            "miss_rate": float(sum(1 for x in cycle_ms if x > args.period_ms) / actual_cycles) if actual_cycles else 0.0,
        }

    summary = {
        "metadata": {
            "workload": args.workload,
            "mode": mode,
            "period_ms": args.period_ms,
            "burst_size": args.burst_size,
            "idle_ms": args.idle_ms,
            "cycles_requested": args.cycles,
            "cycles_executed": actual_cycles,
            "warmup": args.warmup,
            "pin": {"tpu0_core": args.tpu0_core, "tpu1_core": args.tpu1_core, "cpu_core": args.cpu_core},
            "devices": {"tpu0": args.tpu0_device, "tpu1": args.tpu1_device},
            "use_joint": args.use_joint,
            "models": {"tpu0": tpu0_models, "tpu1": tpu1_models, "cpu": ("netvlad_head", cpu_ckpt)},
            "cpu_stage_ms": {
                "odometry": args.cpu_odom_ms,
                "slam": args.cpu_slam_ms,
                "fusion": args.cpu_fusion_ms,
                "mpc": args.cpu_mpc_ms,
            },
            "netvlad_policy": {
                "policy": args.netvlad_policy,
                "every_n": args.netvlad_every_n,
                "prob": args.netvlad_prob,
            },
        },
        "deadline": miss,
        "netvlad_triggered_cycles": netvlad_triggered,
        "cycle_ms": cycle_ms,
        "stages_ms": stages,
        "stats": {
            "cycle": summarize_stats(cycle_ms),
            "stages": {k: summarize_stats(v) for k, v in stages.items()},
        },
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()


