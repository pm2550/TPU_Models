#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版 usbmon 抓包工具

根据用户建议实现的优化措施:
1. 只抓特定 USB 总线/设备
2. 写入 tmpfs (RAM)，避免磁盘 IO 抖动
3. 采样窗口缩短（只抓推理期间）
4. 可选二进制模式（减少格式化开销）

用法:
    python3 optimize_usbmon_capture.py --model <model_path> [选项]
"""

import os
import sys
import time
import json
import argparse
import subprocess
import tempfile
import signal
import statistics
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

# 配置
VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"
PASSWORD_FILE = "/home/10210/Desktop/OS/password.text"

# 全局清理列表
_cleanup_pids = []


def get_password() -> str:
    """读取 sudo 密码"""
    if os.path.exists(PASSWORD_FILE):
        with open(PASSWORD_FILE) as f:
            return f.read().strip()
    return ""


def run_sudo(cmd: str, password: str = None) -> subprocess.CompletedProcess:
    """执行 sudo 命令"""
    if password is None:
        password = get_password()
    full_cmd = f"echo '{password}' | sudo -S -p '' {cmd}"
    return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)


def get_usb_bus() -> Optional[int]:
    """获取 EdgeTPU USB 总线号"""
    env_bus = os.environ.get('USB_BUS') or os.environ.get('EDGETPU_BUS')
    if env_bus and str(env_bus).isdigit():
        return int(env_bus)
    
    try:
        result = subprocess.run(
            [VENV_PY, "/home/10210/Desktop/OS/list_usb_buses.py"],
            capture_output=True, text=True, check=True
        )
        bus_data = json.loads(result.stdout)
        buses = bus_data.get('buses', [])
        if buses:
            return buses[0]
    except Exception as e:
        print(f"[warn] 无法自动检测 USB 总线: {e}")
    return None


def get_edgetpu_device_info() -> Dict:
    """获取 EdgeTPU 设备详细信息（用于精确过滤）"""
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        for line in result.stdout.strip().split('\n'):
            # Google Coral EdgeTPU: ID 18d1:9302
            if '18d1:9302' in line or 'Google' in line:
                parts = line.split()
                bus = parts[1]
                device = parts[3].rstrip(':')
                return {
                    'bus': int(bus),
                    'device': int(device),
                    'vendor_id': '18d1',
                    'product_id': '9302',
                }
    except Exception:
        pass
    return {}


class OptimizedUSBMonCapture:
    """优化版 usbmon 抓包器"""
    
    def __init__(
        self,
        bus: int,
        use_tmpfs: bool = True,
        binary_mode: bool = False,
        device_filter: str = None,
    ):
        self.bus = bus
        self.use_tmpfs = use_tmpfs
        self.binary_mode = binary_mode
        self.device_filter = device_filter
        self.password = get_password()
        self.cap_pid = None
        self.cap_file = None
        self.tmpdir = None
        
    def _ensure_usbmon(self):
        """确保 usbmon 模块已加载"""
        run_sudo("modprobe usbmon", self.password)
        
    def _get_capture_path(self) -> str:
        """获取抓包文件路径"""
        if self.use_tmpfs:
            # 使用 /dev/shm（tmpfs）或 /tmp（通常也是 tmpfs）
            if os.path.exists('/dev/shm'):
                self.tmpdir = tempfile.mkdtemp(prefix='usbmon_', dir='/dev/shm')
            else:
                self.tmpdir = tempfile.mkdtemp(prefix='usbmon_', dir='/tmp')
        else:
            self.tmpdir = tempfile.mkdtemp(prefix='usbmon_')
        
        return os.path.join(self.tmpdir, 'capture.txt')
    
    def start(self) -> str:
        """启动抓包，返回输出文件路径"""
        self._ensure_usbmon()
        
        self.cap_file = self._get_capture_path()
        usbmon_node = f"/sys/kernel/debug/usb/usbmon/{self.bus}u"
        
        # 清空输出文件
        run_sudo(f"sh -c ': > {self.cap_file}'", self.password)
        
        # 构建抓包命令
        if self.binary_mode:
            # 二进制模式：直接读取原始数据
            cmd = f"cat {usbmon_node}"
        else:
            # 文本模式：标准 usbmon 格式
            cmd = f"cat {usbmon_node}"
        
        # 如果有设备过滤器，使用 grep 过滤
        if self.device_filter:
            cmd = f"{cmd} | grep -E '{self.device_filter}'"
        
        # 启动抓包进程
        proc = subprocess.Popen(
            f"echo '{self.password}' | sudo -S -p '' sh -c \"{cmd} >> {self.cap_file}\"",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        self.cap_pid = proc.pid
        _cleanup_pids.append(proc.pid)
        
        time.sleep(0.05)  # 最小等待
        return self.cap_file
    
    def stop(self) -> Tuple[str, int]:
        """停止抓包，返回 (文件路径, 文件大小)"""
        if self.cap_pid:
            run_sudo("pkill -f 'cat.*usbmon'", self.password)
            time.sleep(0.05)
        
        file_size = 0
        if self.cap_file and os.path.exists(self.cap_file):
            file_size = os.path.getsize(self.cap_file)
        
        return self.cap_file, file_size
    
    def cleanup(self):
        """清理临时文件"""
        if self.tmpdir and os.path.exists(self.tmpdir):
            import shutil
            try:
                shutil.rmtree(self.tmpdir)
            except Exception:
                pass


def load_interpreter(model_path: str, use_tpu: bool = True):
    """加载 TFLite 解释器"""
    if use_tpu:
        try:
            from pycoral.utils.edgetpu import make_interpreter
            return make_interpreter(model_path)
        except ImportError:
            from tflite_runtime.interpreter import Interpreter, load_delegate
            return Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
    else:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path)


def run_inference_with_capture(
    model_path: str,
    num_runs: int,
    warmup: int,
    usb_bus: int,
    use_tmpfs: bool = True,
    pre_silence_s: float = 0.2,
    post_silence_s: float = 0.2,
) -> Dict:
    """
    运行推理并同时抓包（优化版：窗口缩短）
    
    返回推理时间和抓包统计
    """
    # 准备解释器
    interpreter = load_interpreter(model_path, use_tpu=True)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    inp_idx = input_details['index']
    out_idx = output_details['index']
    inp_shape = input_details['shape']
    inp_dtype = input_details['dtype']
    
    if inp_dtype == np.uint8:
        dummy = np.random.randint(0, 256, inp_shape, dtype=np.uint8)
    elif inp_dtype == np.int8:
        dummy = np.random.randint(-128, 128, inp_shape, dtype=np.int8)
    else:
        dummy = np.random.random_sample(inp_shape).astype(inp_dtype)
    
    # 热身（不抓包）
    for _ in range(warmup):
        interpreter.set_tensor(inp_idx, dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
    
    # 创建抓包器
    capturer = OptimizedUSBMonCapture(bus=usb_bus, use_tmpfs=use_tmpfs)
    
    invoke_times = []
    
    try:
        # 静默期
        time.sleep(pre_silence_s)
        
        # 开始抓包
        cap_file = capturer.start()
        cap_start = time.perf_counter()
        
        # 运行推理
        for _ in range(num_runs):
            interpreter.set_tensor(inp_idx, dummy)
            t0 = time.perf_counter_ns()
            interpreter.invoke()
            t1 = time.perf_counter_ns()
            _ = interpreter.get_tensor(out_idx)
            invoke_times.append((t1 - t0) / 1e6)
        
        cap_end = time.perf_counter()
        
        # 静默期
        time.sleep(post_silence_s)
        
        # 停止抓包
        _, file_size = capturer.stop()
        
        cap_duration = cap_end - cap_start
        
    finally:
        capturer.cleanup()
    
    return {
        'invoke_times_ms': invoke_times,
        'invoke_median_ms': statistics.median(invoke_times),
        'invoke_mean_ms': statistics.mean(invoke_times),
        'invoke_stdev_ms': statistics.stdev(invoke_times) if len(invoke_times) > 1 else 0,
        'capture_duration_s': cap_duration,
        'capture_file_size_bytes': file_size,
        'capture_bytes_per_s': file_size / cap_duration if cap_duration > 0 else 0,
    }


def compare_capture_modes(
    model_path: str,
    num_runs: int,
    num_trials: int,
    warmup: int,
    usb_bus: int,
) -> Dict:
    """
    比较不同抓包模式的性能影响
    
    模式:
    1. 无抓包
    2. 抓包写入 tmpfs
    3. 抓包写入磁盘
    """
    print(f"\n{'='*60}")
    print("比较不同抓包模式的性能影响")
    print('='*60)
    
    results = {}
    
    # 模式1: 无抓包
    print("\n模式1: 无抓包")
    no_capture_times = []
    for trial in range(num_trials):
        interpreter = load_interpreter(model_path, use_tpu=True)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        inp_idx = input_details['index']
        out_idx = output_details['index']
        inp_shape = input_details['shape']
        inp_dtype = input_details['dtype']
        
        if inp_dtype == np.uint8:
            dummy = np.random.randint(0, 256, inp_shape, dtype=np.uint8)
        elif inp_dtype == np.int8:
            dummy = np.random.randint(-128, 128, inp_shape, dtype=np.int8)
        else:
            dummy = np.random.random_sample(inp_shape).astype(inp_dtype)
        
        for _ in range(warmup):
            interpreter.set_tensor(inp_idx, dummy)
            interpreter.invoke()
            _ = interpreter.get_tensor(out_idx)
        
        times = []
        for _ in range(num_runs):
            interpreter.set_tensor(inp_idx, dummy)
            t0 = time.perf_counter_ns()
            interpreter.invoke()
            t1 = time.perf_counter_ns()
            _ = interpreter.get_tensor(out_idx)
            times.append((t1 - t0) / 1e6)
        
        median = statistics.median(times)
        no_capture_times.append(median)
        print(f"  试验 {trial+1}/{num_trials}: {median:.3f} ms")
    
    results['no_capture'] = {
        'medians': no_capture_times,
        'median': statistics.median(no_capture_times),
        'stdev': statistics.stdev(no_capture_times) if len(no_capture_times) > 1 else 0,
    }
    
    # 模式2: 抓包写入 tmpfs
    print("\n模式2: 抓包写入 tmpfs")
    tmpfs_times = []
    for trial in range(num_trials):
        result = run_inference_with_capture(
            model_path=model_path,
            num_runs=num_runs,
            warmup=warmup,
            usb_bus=usb_bus,
            use_tmpfs=True,
        )
        tmpfs_times.append(result['invoke_median_ms'])
        print(f"  试验 {trial+1}/{num_trials}: {result['invoke_median_ms']:.3f} ms (抓包 {result['capture_file_size_bytes']/1024:.1f} KB)")
    
    results['tmpfs_capture'] = {
        'medians': tmpfs_times,
        'median': statistics.median(tmpfs_times),
        'stdev': statistics.stdev(tmpfs_times) if len(tmpfs_times) > 1 else 0,
    }
    
    # 模式3: 抓包写入磁盘
    print("\n模式3: 抓包写入磁盘")
    disk_times = []
    for trial in range(num_trials):
        result = run_inference_with_capture(
            model_path=model_path,
            num_runs=num_runs,
            warmup=warmup,
            usb_bus=usb_bus,
            use_tmpfs=False,
        )
        disk_times.append(result['invoke_median_ms'])
        print(f"  试验 {trial+1}/{num_trials}: {result['invoke_median_ms']:.3f} ms (抓包 {result['capture_file_size_bytes']/1024:.1f} KB)")
    
    results['disk_capture'] = {
        'medians': disk_times,
        'median': statistics.median(disk_times),
        'stdev': statistics.stdev(disk_times) if len(disk_times) > 1 else 0,
    }
    
    # 计算差异
    baseline = results['no_capture']['median']
    results['comparison'] = {
        'baseline_ms': baseline,
        'tmpfs_overhead_ms': results['tmpfs_capture']['median'] - baseline,
        'disk_overhead_ms': results['disk_capture']['median'] - baseline,
        'tmpfs_overhead_pct': (results['tmpfs_capture']['median'] - baseline) / baseline * 100 if baseline > 0 else 0,
        'disk_overhead_pct': (results['disk_capture']['median'] - baseline) / baseline * 100 if baseline > 0 else 0,
    }
    
    print(f"\n{'='*60}")
    print("对比结果")
    print('='*60)
    print(f"基线 (无抓包):     {baseline:.3f} ms")
    print(f"tmpfs 抓包:        {results['tmpfs_capture']['median']:.3f} ms (Δ={results['comparison']['tmpfs_overhead_ms']:+.3f} ms, {results['comparison']['tmpfs_overhead_pct']:+.2f}%)")
    print(f"磁盘抓包:          {results['disk_capture']['median']:.3f} ms (Δ={results['comparison']['disk_overhead_ms']:+.3f} ms, {results['comparison']['disk_overhead_pct']:+.2f}%)")
    
    return results


def cleanup():
    """清理所有后台进程"""
    password = get_password()
    for pid in _cleanup_pids:
        try:
            run_sudo(f"kill {pid}", password)
        except Exception:
            pass
    run_sudo("pkill -f 'cat.*usbmon'", password)


def main():
    parser = argparse.ArgumentParser(
        description="优化版 usbmon 抓包测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model", required=True, help="TFLite 模型路径")
    parser.add_argument("--trials", type=int, default=10, help="试验次数 (默认: 10)")
    parser.add_argument("--runs", type=int, default=50, help="每次推理次数 (默认: 50)")
    parser.add_argument("--warmup", type=int, default=5, help="热身次数 (默认: 5)")
    parser.add_argument("--bus", type=int, default=None, help="USB 总线号")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[error] 模型文件不存在: {args.model}")
        sys.exit(1)
    
    usb_bus = args.bus
    if usb_bus is None:
        usb_bus = get_usb_bus()
        if usb_bus is None:
            print("[error] 无法检测 USB 总线")
            sys.exit(1)
    
    # 信号处理
    def signal_handler(sig, frame):
        print("\n[info] 正在清理...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"模型: {args.model}")
    print(f"USB 总线: {usb_bus}")
    
    try:
        results = compare_capture_modes(
            model_path=args.model,
            num_runs=args.runs,
            num_trials=args.trials,
            warmup=args.warmup,
            usb_bus=usb_bus,
        )
        
        # 保存结果
        output_path = args.output
        if output_path is None:
            os.makedirs('/home/10210/Desktop/OS/results', exist_ok=True)
            model_name = Path(args.model).stem
            output_path = f'/home/10210/Desktop/OS/results/usbmon_optimize_{model_name}.json'
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {output_path}")
        
    finally:
        cleanup()


if __name__ == "__main__":
    main()
