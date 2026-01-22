#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三模型（其中 DeepLab 与 MobileNet+NetVLAD 串行）并行突发测试 + USBMon 采集。

特性：
1. 场景：Continuous、Periodic(50/100/150 ms)、Burst+Idle（固定 20 ms idle）。
2. 三个逻辑模型：
   - pipeline_ssd ：EfficientDet Lite2 448 (TPU0)
   - pipeline_dl_mnvlad ：DeepLabv3 dm0.5 (TPU1) -> MobileNetV2 (TPU1) -> NetVLAD Head (CPU)
   - pipeline_mobile_only ：MobileNetV2+NetVLAD 独立进程（可选，默认复用 pipeline_dl_mnvlad 的统计）
   其中根据需求，将 DeepLab 与 MobileNet+NetVLAD 串行置于同一进程。
3. 每个进程绑定指定 CPU 核心并执行 100 次正式推理，无额外 warmup。
4. 自动检测 EdgeTPU 的 USB bus（可用 --usb-bus 覆盖），读取 password.text 以执行 sudo 命令。
5. 使用 usbmon 捕获 IO，产出 active_analysis 与 per invoke 修正结果，并绘制 burst 分布。
6. 结果按场景→模型输出 JSON 与图表，可与单模型基线对比。

注意：NetVLAD 头部仍以随机特征模拟输入；要接入真实 backbone 需提供导出特征的模型。
"""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
PASSWORD_FILE = ROOT / "password.text"
RESULTS_ROOT = ROOT / "results" / "multi_model_bursts"
BASELINE_FILE = ROOT / "results" / "four_models_benchmark.json"
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY_EXEC = str(VENV_PY if VENV_PY.exists() else sys.executable)
LIST_USB = ROOT / "list_usb_buses.py"
ANALYZER = ROOT / "analyze_usbmon_active.py"
CORRECTOR = ROOT / "tools" / "correct_per_invoke_stats.py"

DEFAULT_PERIODS = [50.0, 100.0, 150.0]
DEFAULT_BURST_SIZE = 16
DEFAULT_IDLE_MS = 20.0
DEFAULT_ITERATIONS = 100
DEFAULT_CPU_CORES = [2, 3]
DEFAULT_LOCK_FREQ = "1.80GHz"

_DEFAULT_PASSWORD: Optional[str] = None
_EDGETPU_LOAD_LOCK = threading.Lock()


@dataclass
class StageConfig:
    key: str
    stage_type: str  # 'tpu' 或 'netvlad'
    model_path: str
    tpu_device: Optional[str] = None
    head_path: Optional[str] = None
    output_dir: Optional[Path] = None


@dataclass
class PipelineConfig:
    name: str
    stages: List[StageConfig]
    cpu_core: Optional[int]


@dataclass
class Scenario:
    label: str
    period_ms: Optional[float] = None
    burst_every: Optional[int] = None
    idle_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# 辅助函数：权限、CPU、USB
# ---------------------------------------------------------------------------

def read_password() -> Optional[str]:
    global _DEFAULT_PASSWORD
    if _DEFAULT_PASSWORD is not None:
        return _DEFAULT_PASSWORD
    try:
        pw = PASSWORD_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        print(f"[warn] 缺少密码文件 {PASSWORD_FILE}")
        return None
    if not pw:
        print(f"[warn] 密码文件 {PASSWORD_FILE} 为空")
        return None
    _DEFAULT_PASSWORD = pw
    return pw


def run_sudo(cmd: List[str], password: Optional[str], check: bool = True) -> subprocess.CompletedProcess:
    if os.geteuid() == 0:
        return subprocess.run(cmd, check=check, text=True)
    if not password:
        raise RuntimeError(f"需要 sudo 权限执行 {' '.join(cmd)}，但无法读取 password.text")
    return subprocess.run(["sudo", "-S", *cmd], input=password + "\n", text=True, check=check)


def detect_usb_buses() -> List[int]:
    py = str(VENV_PY if VENV_PY.exists() else sys.executable)
    res = subprocess.run([py, str(LIST_USB)], capture_output=True, text=True, check=True)
    data = json.loads(res.stdout)
    buses = data.get("buses", [])
    if not buses:
        raise RuntimeError("list_usb_buses.py 未返回任何 EdgeTPU 总线")
    return [int(b) for b in buses]


def lock_cpu_frequency(cores: List[int], freq: Optional[str]):
    if not freq:
        return
    password = None if os.geteuid() == 0 else read_password()
    for core in cores:
        cmd = ["cpufreq-set", "-c", str(core), "-f", freq]
        try:
            run_sudo(cmd, password, check=True)
        except Exception as exc:
            print(f"[warn] 锁定 CPU{core} 频率失败: {exc}\n       请手动执行: sudo {' '.join(cmd)}")


def ensure_cpu_affinity(core: Optional[int]):
    if core is None:
        return
    proc = psutil.Process()
    try:
        proc.cpu_affinity([core])
    except AttributeError:
        os.sched_setaffinity(0, {core})


# ---------------------------------------------------------------------------
# 阶段执行器
# ---------------------------------------------------------------------------
class TPUStageRunner:
    def __init__(self, stage: StageConfig):
        self.stage = stage
        self.interpreter = None
        self._load_interpreter()
        self.interpreter.allocate_tensors()
        self.inp_detail = self.interpreter.get_input_details()[0]
        self.out_detail = self.interpreter.get_output_details()[0]
        self.dummy = self._build_dummy_input()

    def _load_interpreter(self):
        stage = self.stage
        # Try PyCoral first for best EdgeTPU handling.
        try:
            from pycoral.utils.edgetpu import make_interpreter
        except Exception:
            make_interpreter = None

        if make_interpreter is not None:
            with _EDGETPU_LOAD_LOCK:
                self.interpreter = make_interpreter(stage.model_path, device=stage.tpu_device)
            return

        # Fallback to tflite_runtime with EdgeTPU delegate.
        from tflite_runtime.interpreter import Interpreter, load_delegate

        delegate = None
        try:
            with _EDGETPU_LOAD_LOCK:
                delegate = load_delegate('libedgetpu.so.1', self._delegate_options())
        except (ValueError, OSError) as err:
            raise RuntimeError(
                "无法加载 libedgetpu.so.1，EdgeTPU 模型需要 PyCoral 或 EdgeTPU delegate 支持"
            ) from err

        init_kwargs = {
            "model_path": stage.model_path,
            "experimental_delegates": [delegate],
            "num_threads": 1,
        }

        # Older tflite_runtime builds don't accept use_xnnpack, so add it only when supported.
        if "use_xnnpack" in inspect.signature(Interpreter.__init__).parameters:
            init_kwargs["use_xnnpack"] = False

        self.interpreter = Interpreter(**init_kwargs)

    def _delegate_options(self):
        if not self.stage.tpu_device:
            return {}
        # libedgetpu expects 'device': 'usb:0' etc.
        return {"device": self.stage.tpu_device}

    def _build_dummy_input(self):
        shape = self.inp_detail['shape']
        dtype = self.inp_detail['dtype']
        if dtype == np.uint8:
            return np.random.randint(0, 256, shape, dtype=np.uint8)
        if dtype == np.int8:
            return np.random.randint(-128, 128, shape, dtype=np.int8)
        return np.random.random_sample(shape).astype(dtype)

    def run_once(self) -> Tuple[float, Dict[str, float], np.ndarray]:
        interpreter = self.interpreter
        inp = self.inp_detail
        out = self.out_detail
        interpreter.set_tensor(inp['index'], self.dummy)
        set_begin = time.clock_gettime(time.CLOCK_BOOTTIME)
        interpreter.invoke()
        invoke_end = time.clock_gettime(time.CLOCK_BOOTTIME)
        output = interpreter.get_tensor(out['index'])
        get_end = time.clock_gettime(time.CLOCK_BOOTTIME)
        duration_ms = (get_end - set_begin) * 1000.0
        span = {
            "begin": set_begin,
            "end": invoke_end,
            "set_begin": set_begin,
            "set_end": set_begin,
            "get_begin": invoke_end,
            "get_end": get_end,
            "checksum": int(np.sum(output) % (2**32)),
        }
        return duration_ms, span, output


class NetVLADRunner:
    def __init__(self, stage: StageConfig, feature_shape=(1, 1280, 7, 7)):
        import torch

        self.stage = stage
        self.torch = torch
        self.feature_shape = feature_shape
        descriptor_dim = feature_shape[1]
        n_clusters = 64
        self.conv_weights = torch.randn(n_clusters, descriptor_dim, 1, 1)
        self.cluster_centers = torch.randn(n_clusters, descriptor_dim)
        if not stage.head_path:
            raise ValueError("NetVLAD 阶段缺少 head_path")
        self.checkpoint = torch.load(stage.head_path, map_location='cpu')

    def run_once(self, features: Optional[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        torch = self.torch
        if features is None:
            tensor = torch.randn(*self.feature_shape)
        else:
            try:
                tensor = torch.from_numpy(features.reshape(self.feature_shape)).float()
            except Exception:
                tensor = torch.randn(*self.feature_shape)
        set_begin = time.clock_gettime(time.CLOCK_BOOTTIME)
        t0 = time.perf_counter()
        soft_assign = torch.nn.functional.conv2d(tensor, self.conv_weights)
        soft_assign = torch.nn.functional.softmax(soft_assign, dim=1)
        descriptor_dim = tensor.shape[1]
        n_clusters = soft_assign.shape[1]
        feature_flat = tensor.view(1, descriptor_dim, -1)
        soft_assign_flat = soft_assign.view(1, n_clusters, -1)
        residuals = []
        for k in range(n_clusters):
            centroid = self.cluster_centers[k].view(1, descriptor_dim, 1)
            residual = (feature_flat - centroid) * soft_assign_flat[:, k:k+1, :]
            residuals.append(residual.sum(dim=2))
        vlad = torch.cat(residuals, dim=1)
        vlad = torch.nn.functional.normalize(vlad, p=2, dim=1)
        if "WPCA" in self.checkpoint:
            wpca = self.checkpoint["WPCA"]
            if isinstance(wpca, np.ndarray):
                wpca = torch.from_numpy(wpca).float()
            if wpca.shape[0] == vlad.shape[1]:
                final_desc = torch.matmul(vlad, wpca.t())
            else:
                final_desc = vlad[:, :512]
        else:
            final_desc = vlad[:, :512]
        final_desc = torch.nn.functional.normalize(final_desc, p=2, dim=1)
        t1 = time.perf_counter()
        get_end = time.clock_gettime(time.CLOCK_BOOTTIME)
        span = {
            "begin": set_begin,
            "end": get_end,
            "set_begin": set_begin,
            "set_end": set_begin,
            "get_begin": get_end,
            "get_end": get_end,
            "checksum": float(final_desc.sum().item()),
        }
        return (t1 - t0) * 1000.0, span


# ---------------------------------------------------------------------------
# 管线执行
# ---------------------------------------------------------------------------

def pipeline_worker(cfg: PipelineConfig, scenario: Scenario, iterations: int, result_queue: mp.Queue, synthetic: bool = False):
    ensure_cpu_affinity(cfg.cpu_core)
    stage_runners = {}
    for stage in cfg.stages:
        if stage.stage_type == 'tpu':
            stage_runners[stage.key] = TPUStageRunner(stage) if not synthetic else None
        elif stage.stage_type == 'netvlad':
            stage_runners[stage.key] = NetVLADRunner(stage)
        else:
            raise ValueError(f"未知 stage_type: {stage.stage_type}")

    per_stage_stats = {stage.key: {"durations": [], "spans": []} for stage in cfg.stages}
    next_deadline: Optional[float] = None
    burst_counter = 0

    for _ in range(iterations):
        cycle_start = time.perf_counter()
        last_output = None
        for stage in cfg.stages:
            runner = stage_runners[stage.key]
            if stage.stage_type == 'tpu':
                if runner is None:
                    time.sleep(0.001)
                    duration_ms = 1.0
                    span = {
                        "begin": time.clock_gettime(time.CLOCK_BOOTTIME),
                        "end": time.clock_gettime(time.CLOCK_BOOTTIME),
                        "set_begin": 0.0,
                        "set_end": 0.0,
                        "get_begin": 0.0,
                        "get_end": 0.0,
                        "checksum": 0,
                    }
                    last_output = None
                else:
                    duration_ms, span, last_output = runner.run_once()
            else:
                duration_ms, span = runner.run_once(last_output)
                last_output = None
            per_stage_stats[stage.key]["durations"].append(duration_ms)
            per_stage_stats[stage.key]["spans"].append(span)
        if scenario.period_ms:
            next_deadline = cycle_start + scenario.period_ms / 1000.0 if next_deadline is None else next_deadline + scenario.period_ms / 1000.0
            sleep_time = next_deadline - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
        elif scenario.burst_every and scenario.idle_ms:
            burst_counter += 1
            if burst_counter >= scenario.burst_every:
                time.sleep(scenario.idle_ms / 1000.0)
                burst_counter = 0

    for stage in cfg.stages:
        stats = per_stage_stats[stage.key]
        durations = stats["durations"]
        stats.update({
            "avg_ms": float(np.mean(durations)) if durations else 0.0,
            "std_ms": float(np.std(durations)) if durations else 0.0,
            "samples": len(durations),
            "invokes": stats["spans"],
        })
    result_queue.put({"pipeline": cfg.name, "scenario": scenario.label, "stats": per_stage_stats})


# ---------------------------------------------------------------------------
# USBMon 采集与分析
# ---------------------------------------------------------------------------

def start_usbmon(bus: int, out_dir: Path) -> Tuple[subprocess.Popen, Path, Path]:
    password = None if os.geteuid() == 0 else read_password()
    run_sudo(["modprobe", "usbmon"], password, check=False)
    usbmon_node = f"/sys/kernel/debug/usb/usbmon/{bus}u"
    cap_path = out_dir / f"usbmon_bus{bus}.txt"
    tm_path = out_dir / f"time_map_bus{bus}.json"
    cap_file = open(cap_path, "wb")
    cmd = ["cat", usbmon_node]
    if os.geteuid() == 0:
        proc = subprocess.Popen(cmd, stdout=cap_file, stderr=subprocess.DEVNULL)
    else:
        proc = subprocess.Popen(["sudo", "-S", *cmd], stdin=subprocess.PIPE, stdout=cap_file, stderr=subprocess.DEVNULL, text=True)
        password = read_password()
        if password and proc.stdin:
            proc.stdin.write(password + "\n")
            proc.stdin.flush()
    _spawn_time_map_writer(cap_path, tm_path)
    return proc, cap_path, tm_path


def stop_usbmon(proc: subprocess.Popen, files: List[Path]):
    password = None if os.geteuid() == 0 else read_password()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
    for f in files:
        if f.exists():
            run_sudo(["chown", f"{os.getuid()}:{os.getgid()}", str(f)], password, check=False)
            run_sudo(["chmod", "0644", str(f)], password, check=False)


def _spawn_time_map_writer(cap_path: Path, tm_path: Path):
    def worker():
        deadline = time.time() + 30
        usb_ts = None
        while time.time() < deadline and usb_ts is None:
            if cap_path.exists():
                try:
                    with open(cap_path, "r", errors='ignore') as handle:
                        for line in handle:
                            parts = line.split()
                            if not parts:
                                continue
                            for idx in (1, 0):
                                if idx < len(parts):
                                    try:
                                        val = float(parts[idx])
                                        usb_ts = val / 1e6 if val > 1e6 else val
                                        break
                                    except Exception:
                                        continue
                            if usb_ts is not None:
                                break
                except Exception:
                    pass
            if usb_ts is None:
                time.sleep(0.05)
        try:
            bt = time.clock_gettime(time.CLOCK_BOOTTIME)
        except Exception:
            bt = float(open('/proc/uptime').read().split()[0])
        tm = {"usbmon_ref": usb_ts, "boottime_ref": bt}
        with open(tm_path, "w", encoding="utf-8") as handle:
            json.dump(tm, handle, ensure_ascii=False, indent=2)
    threading.Thread(target=worker, daemon=True).start()


def run_corrector(usbmon: Path, invokes: Path, tm: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PY_EXEC, str(CORRECTOR), str(usbmon), str(invokes), str(tm), "--mode", "bulk_complete", "--include", "full", "--extra", "0.000"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "analysis").mkdir(exist_ok=True)
    (out_dir / "analysis" / "correct_stdout.txt").write_text(res.stdout)
    (out_dir / "analysis" / "correct_summary.json").write_text(json.dumps({
        "returncode": res.returncode,
        "stderr": res.stderr.strip(),
    }, ensure_ascii=False, indent=2))


def run_active_analysis(usbmon: Path, invokes: Path, tm: Path, out_dir: Path):
    if not ANALYZER.exists():
        print(f"[warn] 缺少 {ANALYZER}，跳过 active analysis")
        return
    cmd = [PY_EXEC, str(ANALYZER), str(usbmon), str(invokes), str(tm)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "analysis").mkdir(exist_ok=True)
    target = out_dir / "analysis" / "active_analysis_strict.json"
    if res.returncode == 0:
        target.write_text(res.stdout)
    else:
        target.write_text(json.dumps({"error": res.stderr}, ensure_ascii=False))


def plot_burst(active_json: Path, out_path: Path, title: str):
    try:
        if not active_json.exists():
            return
        data = json.loads(active_json.read_text())
        per = data.get('per_invoke', [])
        union_ms = [entry.get('union_active_span_s', 0.0) * 1000 for entry in per]
        if not union_ms:
            return
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 4))
        plt.hist(union_ms, bins='auto', edgecolor='black', alpha=0.8)
        plt.xlabel('Union Active Time (ms)')
        plt.ylabel('Count')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as exc:
        print(f"[warn] 绘制 burst 失败 ({active_json}): {exc}")


# ---------------------------------------------------------------------------
# 场景运行
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario, pipelines: List[PipelineConfig], args, baseline: Dict[str, float], usb_buses: List[int]):
    label = scenario.label if scenario.label != 'periodic' else f"periodic_{int(scenario.period_ms)}"
    scenario_root = RESULTS_ROOT / label
    scenario_root.mkdir(parents=True, exist_ok=True)

    usb_monitors = {}
    for bus in usb_buses:
        usb_monitors[bus] = start_usbmon(bus, scenario_root)
    time.sleep(1.0)

    queue_results: mp.Queue = mp.Queue()
    procs = []
    stage_info: Dict[str, StageConfig] = {}
    for pipe in pipelines:
        cfg_copy = PipelineConfig(pipe.name, [], pipe.cpu_core)
        for stage in pipe.stages:
            stage_copy = StageConfig(**stage.__dict__)
            stage_copy.output_dir = scenario_root / stage.key
            stage_copy.output_dir.mkdir(parents=True, exist_ok=True)
            cfg_copy.stages.append(stage_copy)
            stage_info[stage_copy.key] = stage_copy
        p = mp.Process(
            target=pipeline_worker,
            args=(cfg_copy, scenario, args.iterations, queue_results, args.synthetic),
        )
        p.start()
        procs.append((p, cfg_copy))

    stage_stats = {}
    for _ in procs:
        result = queue_results.get()
        stats = result['stats']
        for stage_key, data in stats.items():
            stage_stats.setdefault(stage_key, {
                "durations": [],
                "avg_ms": 0.0,
                "std_ms": 0.0,
                "samples": 0,
                "invokes": []
            })
            stage_stats[stage_key]["avg_ms"] = data["avg_ms"]
            stage_stats[stage_key]["std_ms"] = data["std_ms"]
            stage_stats[stage_key]["samples"] = data["samples"]
            stage_stats[stage_key]["invokes"] = data["invokes"]

    for p, _ in procs:
        p.join()

    time.sleep(args.tail)
    for proc, txt, tm in usb_monitors.values():
        stop_usbmon(proc, [txt, tm])

    summary = {
        "scenario": label,
        "period_ms": scenario.period_ms,
        "burst_every": scenario.burst_every,
        "idle_ms": scenario.idle_ms,
        "models": {}
    }

    for stage_key, data in stage_stats.items():
        stage_dir = scenario_root / stage_key
        invokes_path = stage_dir / "invokes.json"
        invokes_path.write_text(json.dumps({"name": stage_key, "spans": data['invokes']}, ensure_ascii=False, indent=2))

        stage_cfg = stage_info.get(stage_key)
        usbmon_path = None
        tm_path = None
        if stage_cfg and stage_cfg.stage_type == 'tpu' and usb_buses:
            tpu_idx = 0
            if stage_cfg.tpu_device:
                try:
                    tpu_idx = int(stage_cfg.tpu_device.split(':')[1])
                except (IndexError, ValueError):
                    tpu_idx = 0
            bus = usb_buses[min(tpu_idx, len(usb_buses) - 1)]
            entry = usb_monitors.get(bus)
            if entry:
                _, usb_txt, tm_json = entry
                usbmon_path = usb_txt
                tm_path = tm_json

        if usbmon_path and tm_path:
            run_corrector(usbmon_path, invokes_path, tm_path, stage_dir)
            run_active_analysis(usbmon_path, invokes_path, tm_path, stage_dir)
            plot_burst(stage_dir / "analysis" / "active_analysis_strict.json", stage_dir / "analysis" / "burst_hist.png", f"{stage_key} {label}")
        base = baseline.get(stage_key)
        stats_entry = {
            "avg_ms": data['avg_ms'],
            "std_ms": data['std_ms'],
            "samples": data['samples'],
            "invokes_file": str(invokes_path)
        }
        if base is not None:
            stats_entry['delta_vs_base_ms'] = data['avg_ms'] - base
            stats_entry['ratio_vs_base'] = (data['avg_ms'] / base) if base else None
        summary['models'][stage_key] = stats_entry

    summary_path = scenario_root / "scenario_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


# ---------------------------------------------------------------------------
# CLI、基线、主程序
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="三模型并行突发测试")
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument('--periods', type=float, nargs='+', default=DEFAULT_PERIODS)
    parser.add_argument('--burst-size', type=int, default=DEFAULT_BURST_SIZE)
    parser.add_argument('--idle-ms', type=float, default=DEFAULT_IDLE_MS)
    parser.add_argument('--cpu-cores', type=int, nargs='+', default=DEFAULT_CPU_CORES)
    parser.add_argument('--lock-freq', type=str, default=DEFAULT_LOCK_FREQ)
    parser.add_argument('--usb-bus', type=int, help="仅记录指定 USB bus（默认自动检测）")
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--tail', type=float, default=0.5)
    return parser.parse_args()


def load_baseline() -> Dict[str, float]:
    if not BASELINE_FILE.exists():
        return {}
    try:
        data = json.loads(BASELINE_FILE.read_text())
    except Exception:
        return {}
    result = {}
    for key, entry in data.items():
        stats = entry.get('statistics', {})
        result[key] = stats.get('warm_avg_ms')
    return result


def build_pipelines(cpu_cores: List[int]) -> List[PipelineConfig]:
    core_ssd = cpu_cores[0] if cpu_cores else None
    core_dl = cpu_cores[1] if len(cpu_cores) > 1 else None
    pipelines = [
        PipelineConfig(
            name='pipeline_ssd',
            cpu_core=core_ssd,
            stages=[
                StageConfig(
                    key='ssd_mobilenet_v2',
                    stage_type='tpu',
                    model_path=str(ROOT / "models_local" / "public" / "efficientdet_lite2_448_ptq_edgetpu.tflite"),
                    tpu_device='usb:0'
                )
            ]
        ),
        PipelineConfig(
            name='pipeline_dl_mnvlad',
            cpu_core=core_dl,
            stages=[
                StageConfig(
                    key='deeplabv3',
                    stage_type='tpu',
                    model_path=str(ROOT / "models_local" / "public" / "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite"),
                    tpu_device='usb:1'
                ),
                StageConfig(
                    key='mobilenet_v2',
                    stage_type='tpu',
                    model_path="/home/10210/Desktop/ROS/models/edgetpu_compiled/mobilenet_v2_1.0_224_edgetpu.tflite",
                    tpu_device='usb:1'
                ),
                StageConfig(
                    key='netvlad_head',
                    stage_type='netvlad',
                    model_path='',
                    head_path="/home/10210/Desktop/ROS/models/mapillary_WPCA512.pth.tar"
                )
            ]
        )
    ]
    return pipelines


def main():
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    if args.usb_bus is not None:
        usb_buses = [args.usb_bus]
    else:
        usb_buses = detect_usb_buses()
    print(f"[info] 使用 USB bus {', '.join(map(str, usb_buses))}")
    lock_cpu_frequency(args.cpu_cores, args.lock_freq)
    baseline = load_baseline()
    pipelines = build_pipelines(args.cpu_cores)

    scenarios = [Scenario(label='continuous')]
    for p in args.periods:
        scenarios.append(Scenario(label='periodic', period_ms=p))
    scenarios.append(Scenario(label='burst_idle', burst_every=args.burst_size, idle_ms=args.idle_ms))

    overall = []
    for sc in scenarios:
        print(f"\n=== 场景: {sc.label} (period={sc.period_ms}, burst={sc.burst_every}) ===")
        summary = run_scenario(sc, pipelines, args, baseline, usb_buses)
        overall.append(summary)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / 'overall_summary.json').write_text(json.dumps(overall, ensure_ascii=False, indent=2))
    print(f"\n完成，详情见 {RESULTS_ROOT / 'overall_summary.json'}")


if __name__ == '__main__':
    main()
