#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B 测试脚本：量化 usbmon 对推理时间的影响

用法:
    python3 test_usbmon_overhead_ab.py --model <model_path> [选项]

测试组:
    A组: 不开 usbmon，纯推理测量
    B组: 开 usbmon，同样方式抓取

统计方法:
    - 每组运行 N 次（默认30），取中位数
    - 计算 ΔT = median(T_B) - median(T_A)
    - 分析 ΔT 是否为常数或与吞吐相关
"""

import os
import sys
import time
import json
import argparse
import subprocess
import statistics
import signal
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

# 配置
VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"
SYS_PY = "python3"
PASSWORD_FILE = "/home/10210/Desktop/OS/password.text"

# 全局变量用于清理
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


def set_cpu_performance_governor(enable: bool = True) -> bool:
    """设置 CPU 频率为 performance 模式"""
    password = get_password()
    if not password:
        print("[warn] 无密码文件，跳过 CPU governor 设置")
        return False
    
    governor = "performance" if enable else "ondemand"
    
    try:
        # 查找所有 CPU 核心的 governor 文件
        cpu_paths = list(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor"))
        if not cpu_paths:
            print("[warn] 未找到 CPU governor 文件")
            return False
        
        for path in cpu_paths:
            run_sudo(f"sh -c 'echo {governor} > {path}'", password)
        
        print(f"[info] CPU governor 已设置为: {governor}")
        return True
    except Exception as e:
        print(f"[warn] 设置 CPU governor 失败: {e}")
        return False


def start_usbmon_capture(bus: int, output_file: str) -> Optional[int]:
    """启动 usbmon 抓包"""
    password = get_password()
    if not password:
        return None
    
    usbmon_node = f"/sys/kernel/debug/usb/usbmon/{bus}u"
    
    # 确保 usbmon 模块已加载
    run_sudo("modprobe usbmon", password)
    
    # 清空输出文件
    run_sudo(f"sh -c ': > {output_file}'", password)
    
    # 启动抓包进程
    cmd = f"cat {usbmon_node} >> {output_file}"
    proc = subprocess.Popen(
        f"echo '{password}' | sudo -S -p '' sh -c \"{cmd}\"",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    _cleanup_pids.append(proc.pid)
    time.sleep(0.1)  # 等待抓包启动
    
    return proc.pid


def stop_usbmon_capture(pid: int):
    """停止 usbmon 抓包"""
    if pid:
        try:
            password = get_password()
            # 使用 pkill 杀掉相关的 cat 进程
            run_sudo(f"pkill -f 'cat.*usbmon'", password)
            time.sleep(0.1)
        except Exception:
            pass


def cleanup():
    """清理所有后台进程"""
    password = get_password()
    for pid in _cleanup_pids:
        try:
            run_sudo(f"kill {pid}", password)
        except Exception:
            pass
    run_sudo("pkill -f 'cat.*usbmon'", password)


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


def run_inference_batch(
    model_path: str,
    num_runs: int,
    warmup: int = 5,
    use_tpu: bool = True,
    invoke_gap_ms: float = 0.0
) -> Dict:
    """
    运行一批推理并返回时间统计
    
    返回: {
        'invoke_times_ms': [...],
        'total_times_ms': [...],
        'pre_times_ms': [...],
        'post_times_ms': [...],
    }
    """
    interpreter = load_interpreter(model_path, use_tpu)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    inp_idx = input_details['index']
    out_idx = output_details['index']
    inp_shape = input_details['shape']
    inp_dtype = input_details['dtype']
    
    # 生成随机输入
    if inp_dtype == np.uint8:
        dummy = np.random.randint(0, 256, inp_shape, dtype=np.uint8)
    elif inp_dtype == np.int8:
        dummy = np.random.randint(-128, 128, inp_shape, dtype=np.int8)
    else:
        dummy = np.random.random_sample(inp_shape).astype(inp_dtype)
    
    # 热身
    for _ in range(warmup):
        interpreter.set_tensor(inp_idx, dummy)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_idx)
    
    # 测量
    invoke_times = []
    total_times = []
    pre_times = []
    post_times = []
    
    for _ in range(num_runs):
        t_a = time.perf_counter_ns()
        interpreter.set_tensor(inp_idx, dummy)
        t_b = time.perf_counter_ns()
        
        interpreter.invoke()
        t_c = time.perf_counter_ns()
        
        _ = interpreter.get_tensor(out_idx)
        t_d = time.perf_counter_ns()
        
        pre_times.append((t_b - t_a) / 1e6)
        invoke_times.append((t_c - t_b) / 1e6)
        post_times.append((t_d - t_c) / 1e6)
        total_times.append((t_d - t_a) / 1e6)
        
        if invoke_gap_ms > 0:
            time.sleep(invoke_gap_ms / 1000.0)
    
    return {
        'invoke_times_ms': invoke_times,
        'total_times_ms': total_times,
        'pre_times_ms': pre_times,
        'post_times_ms': post_times,
    }


def compute_stats(values: List[float]) -> Dict:
    """计算统计数据"""
    if not values:
        return {}
    
    return {
        'mean': statistics.mean(values),
        'median': statistics.median(values),
        'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
        'min': min(values),
        'max': max(values),
        'p5': np.percentile(values, 5),
        'p95': np.percentile(values, 95),
        'cv': (statistics.stdev(values) / statistics.mean(values) * 100) if len(values) > 1 and statistics.mean(values) > 0 else 0.0,
    }


def run_group_a(
    model_path: str,
    num_runs: int,
    num_trials: int,
    warmup: int,
    invoke_gap_ms: float
) -> Dict:
    """
    组A: 不开 usbmon，纯推理测量
    """
    print(f"\n{'='*60}")
    print("组A: 不开 usbmon")
    print('='*60)
    
    all_invoke_medians = []
    all_total_medians = []
    all_invoke_values = []
    all_total_values = []
    
    for trial in range(num_trials):
        print(f"  试验 {trial+1}/{num_trials}...", end=" ", flush=True)
        
        result = run_inference_batch(
            model_path=model_path,
            num_runs=num_runs,
            warmup=warmup,
            use_tpu=True,
            invoke_gap_ms=invoke_gap_ms
        )
        
        invoke_median = statistics.median(result['invoke_times_ms'])
        total_median = statistics.median(result['total_times_ms'])
        
        all_invoke_medians.append(invoke_median)
        all_total_medians.append(total_median)
        all_invoke_values.extend(result['invoke_times_ms'])
        all_total_values.extend(result['total_times_ms'])
        
        print(f"invoke中位数={invoke_median:.3f}ms, total中位数={total_median:.3f}ms")
        
        time.sleep(0.5)  # 试验间隔
    
    return {
        'invoke_medians': all_invoke_medians,
        'total_medians': all_total_medians,
        'invoke_all': all_invoke_values,
        'total_all': all_total_values,
        'invoke_stats': compute_stats(all_invoke_medians),
        'total_stats': compute_stats(all_total_medians),
    }


def run_group_b(
    model_path: str,
    num_runs: int,
    num_trials: int,
    warmup: int,
    invoke_gap_ms: float,
    usb_bus: int
) -> Dict:
    """
    组B: 开 usbmon，同样的抓取方式
    """
    print(f"\n{'='*60}")
    print(f"组B: 开 usbmon (总线 {usb_bus})")
    print('='*60)
    
    all_invoke_medians = []
    all_total_medians = []
    all_invoke_values = []
    all_total_values = []
    
    with tempfile.TemporaryDirectory(prefix='usbmon_test_') as tmpdir:
        usbmon_file = os.path.join(tmpdir, 'usbmon.txt')
        
        for trial in range(num_trials):
            print(f"  试验 {trial+1}/{num_trials}...", end=" ", flush=True)
            
            # 启动 usbmon 抓包
            cap_pid = start_usbmon_capture(usb_bus, usbmon_file)
            if cap_pid is None:
                print("[error] 无法启动 usbmon")
                continue
            
            time.sleep(0.3)  # 等待抓包稳定
            
            try:
                result = run_inference_batch(
                    model_path=model_path,
                    num_runs=num_runs,
                    warmup=warmup,
                    use_tpu=True,
                    invoke_gap_ms=invoke_gap_ms
                )
                
                invoke_median = statistics.median(result['invoke_times_ms'])
                total_median = statistics.median(result['total_times_ms'])
                
                all_invoke_medians.append(invoke_median)
                all_total_medians.append(total_median)
                all_invoke_values.extend(result['invoke_times_ms'])
                all_total_values.extend(result['total_times_ms'])
                
                print(f"invoke中位数={invoke_median:.3f}ms, total中位数={total_median:.3f}ms")
                
            finally:
                stop_usbmon_capture(cap_pid)
            
            time.sleep(0.5)  # 试验间隔
    
    return {
        'invoke_medians': all_invoke_medians,
        'total_medians': all_total_medians,
        'invoke_all': all_invoke_values,
        'total_all': all_total_values,
        'invoke_stats': compute_stats(all_invoke_medians),
        'total_stats': compute_stats(all_total_medians),
    }


def run_differential_test(
    model_path: str,
    num_runs: int,
    num_trials: int,
    warmup: int,
    invoke_gap_ms: float,
    usb_bus: int
) -> Dict:
    """
    差分设计测试: cold vs warm（开着 usbmon 做差分）
    """
    print(f"\n{'='*60}")
    print("差分测试: cold vs warm (同开 usbmon)")
    print('='*60)
    
    cold_times = []  # 每次新建 delegate 的首次推理
    warm_times = []  # 热身后的推理
    
    with tempfile.TemporaryDirectory(prefix='usbmon_diff_') as tmpdir:
        usbmon_file = os.path.join(tmpdir, 'usbmon.txt')
        
        for trial in range(num_trials):
            print(f"  试验 {trial+1}/{num_trials}...", end=" ", flush=True)
            
            # 启动 usbmon 抓包
            cap_pid = start_usbmon_capture(usb_bus, usbmon_file)
            if cap_pid is None:
                print("[error] 无法启动 usbmon")
                continue
            
            time.sleep(0.3)
            
            try:
                # Cold: 新建解释器，首次推理
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
                
                # Cold 推理
                interpreter.set_tensor(inp_idx, dummy)
                t0 = time.perf_counter_ns()
                interpreter.invoke()
                t1 = time.perf_counter_ns()
                _ = interpreter.get_tensor(out_idx)
                
                cold_time = (t1 - t0) / 1e6
                cold_times.append(cold_time)
                
                # 热身
                for _ in range(warmup):
                    interpreter.set_tensor(inp_idx, dummy)
                    interpreter.invoke()
                    _ = interpreter.get_tensor(out_idx)
                
                # Warm 推理（取多次中位数）
                warm_results = []
                for _ in range(num_runs):
                    interpreter.set_tensor(inp_idx, dummy)
                    t0 = time.perf_counter_ns()
                    interpreter.invoke()
                    t1 = time.perf_counter_ns()
                    _ = interpreter.get_tensor(out_idx)
                    warm_results.append((t1 - t0) / 1e6)
                
                warm_median = statistics.median(warm_results)
                warm_times.append(warm_median)
                
                delta = cold_time - warm_median
                print(f"cold={cold_time:.3f}ms, warm中位数={warm_median:.3f}ms, Δ={delta:.3f}ms")
                
            finally:
                stop_usbmon_capture(cap_pid)
            
            time.sleep(0.5)
    
    return {
        'cold_times': cold_times,
        'warm_times': warm_times,
        'delta_times': [c - w for c, w in zip(cold_times, warm_times)],
        'cold_stats': compute_stats(cold_times),
        'warm_stats': compute_stats(warm_times),
        'delta_stats': compute_stats([c - w for c, w in zip(cold_times, warm_times)]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="A/B 测试：量化 usbmon 对推理时间的影响",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本测试
  python3 test_usbmon_overhead_ab.py --model ./model/mobilenet.tflite
  
  # 指定参数
  python3 test_usbmon_overhead_ab.py --model ./model.tflite --trials 30 --runs 50 --gap 100
  
  # 使用分段模型
  python3 test_usbmon_overhead_ab.py --model ./models_local/public/resnet50_8seg_uniform_local/full_split_pipeline_local/tpu/seg1_int8_edgetpu.tflite
        """
    )
    
    parser.add_argument("--model", required=True, help="TFLite 模型路径 (.tflite)")
    parser.add_argument("--trials", type=int, default=30, help="每组试验次数 (默认: 30)")
    parser.add_argument("--runs", type=int, default=50, help="每次试验的推理次数 (默认: 50)")
    parser.add_argument("--warmup", type=int, default=5, help="热身推理次数 (默认: 5)")
    parser.add_argument("--gap", type=float, default=0.0, help="推理间隔 (ms, 默认: 0)")
    parser.add_argument("--bus", type=int, default=None, help="USB 总线号 (默认: 自动检测)")
    parser.add_argument("--no-performance", action="store_true", help="不设置 CPU performance governor")
    parser.add_argument("--skip-a", action="store_true", help="跳过组A测试")
    parser.add_argument("--skip-b", action="store_true", help="跳过组B测试")
    parser.add_argument("--skip-diff", action="store_true", help="跳过差分测试")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    
    args = parser.parse_args()
    
    # 验证模型文件
    if not os.path.exists(args.model):
        print(f"[error] 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 获取 USB 总线
    usb_bus = args.bus
    if usb_bus is None:
        usb_bus = get_usb_bus()
        if usb_bus is None:
            print("[error] 无法检测 USB 总线，请使用 --bus 参数指定")
            sys.exit(1)
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n[info] 收到中断信号，正在清理...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("="*60)
    print("usbmon 开销 A/B 测试")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"USB 总线: {usb_bus}")
    print(f"试验次数: {args.trials}")
    print(f"每试验推理次数: {args.runs}")
    print(f"热身次数: {args.warmup}")
    print(f"推理间隔: {args.gap} ms")
    
    # 设置 CPU performance governor
    if not args.no_performance:
        set_cpu_performance_governor(True)
    
    results = {
        'config': {
            'model': args.model,
            'usb_bus': usb_bus,
            'trials': args.trials,
            'runs': args.runs,
            'warmup': args.warmup,
            'invoke_gap_ms': args.gap,
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    try:
        # 组A测试
        if not args.skip_a:
            results['group_a'] = run_group_a(
                model_path=args.model,
                num_runs=args.runs,
                num_trials=args.trials,
                warmup=args.warmup,
                invoke_gap_ms=args.gap
            )
        
        # 组B测试
        if not args.skip_b:
            results['group_b'] = run_group_b(
                model_path=args.model,
                num_runs=args.runs,
                num_trials=args.trials,
                warmup=args.warmup,
                invoke_gap_ms=args.gap,
                usb_bus=usb_bus
            )
        
        # 差分测试
        if not args.skip_diff:
            results['differential'] = run_differential_test(
                model_path=args.model,
                num_runs=args.runs,
                num_trials=args.trials,
                warmup=args.warmup,
                invoke_gap_ms=args.gap,
                usb_bus=usb_bus
            )
        
        # 计算 A/B 差异
        if 'group_a' in results and 'group_b' in results:
            a_invoke = results['group_a']['invoke_stats']
            b_invoke = results['group_b']['invoke_stats']
            a_total = results['group_a']['total_stats']
            b_total = results['group_b']['total_stats']
            
            delta_invoke_median = b_invoke.get('median', 0) - a_invoke.get('median', 0)
            delta_total_median = b_total.get('median', 0) - a_total.get('median', 0)
            
            # 百分比变化
            pct_invoke = (delta_invoke_median / a_invoke['median'] * 100) if a_invoke.get('median', 0) > 0 else 0
            pct_total = (delta_total_median / a_total['median'] * 100) if a_total.get('median', 0) > 0 else 0
            
            results['comparison'] = {
                'delta_invoke_median_ms': delta_invoke_median,
                'delta_total_median_ms': delta_total_median,
                'delta_invoke_pct': pct_invoke,
                'delta_total_pct': pct_total,
                'analysis': {
                    'is_constant_overhead': abs(delta_invoke_median) < 1.0,  # 如果 <1ms 认为是常数
                    'recommendation': 'constant_correction' if abs(delta_invoke_median) < 1.0 else 'need_isolation',
                }
            }
            
            print(f"\n{'='*60}")
            print("A/B 对比结果")
            print('='*60)
            print(f"组A (无 usbmon):")
            print(f"  invoke 中位数: {a_invoke.get('median', 0):.3f} ms (±{a_invoke.get('stdev', 0):.3f})")
            print(f"  total  中位数: {a_total.get('median', 0):.3f} ms (±{a_total.get('stdev', 0):.3f})")
            print(f"\n组B (开 usbmon):")
            print(f"  invoke 中位数: {b_invoke.get('median', 0):.3f} ms (±{b_invoke.get('stdev', 0):.3f})")
            print(f"  total  中位数: {b_total.get('median', 0):.3f} ms (±{b_total.get('stdev', 0):.3f})")
            print(f"\nΔT (B - A):")
            print(f"  invoke: {delta_invoke_median:+.3f} ms ({pct_invoke:+.2f}%)")
            print(f"  total:  {delta_total_median:+.3f} ms ({pct_total:+.2f}%)")
            print(f"\n结论:")
            if abs(delta_invoke_median) < 0.5:
                print(f"  ✓ usbmon 开销可忽略 (<0.5ms)，无需校正")
            elif abs(delta_invoke_median) < 2.0:
                print(f"  → usbmon 开销较小 ({delta_invoke_median:.3f}ms)，可做常数校正")
            else:
                print(f"  ! usbmon 开销较大 ({delta_invoke_median:.3f}ms)，建议使用差分设计或隔离手段")
        
        # 差分测试结果
        if 'differential' in results:
            diff = results['differential']
            print(f"\n{'='*60}")
            print("差分测试结果 (cold vs warm)")
            print('='*60)
            print(f"cold 中位数: {diff['cold_stats'].get('median', 0):.3f} ms")
            print(f"warm 中位数: {diff['warm_stats'].get('median', 0):.3f} ms")
            print(f"Δ (cold-warm) 中位数: {diff['delta_stats'].get('median', 0):.3f} ms")
            print(f"\n说明: 差分设计可抵消 usbmon 的很多固定开销")
        
        # 保存结果
        output_path = args.output
        if output_path is None:
            os.makedirs('/home/10210/Desktop/OS/results', exist_ok=True)
            model_name = Path(args.model).stem
            output_path = f'/home/10210/Desktop/OS/results/usbmon_ab_test_{model_name}.json'
        
        # 移除大数组以节省空间（可选保留）
        results_save = results.copy()
        for group in ['group_a', 'group_b']:
            if group in results_save:
                results_save[group] = {
                    k: v for k, v in results_save[group].items()
                    if k not in ['invoke_all', 'total_all']
                }
        
        with open(output_path, 'w') as f:
            json.dump(results_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: {output_path}")
        
    finally:
        cleanup()
        if not args.no_performance:
            set_cpu_performance_governor(False)


if __name__ == "__main__":
    main()
