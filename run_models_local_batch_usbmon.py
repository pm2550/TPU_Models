#!/usr/bin/env python3
"""
批量测试 models_local 中所有模型的分段推理性能
基于 tools/run_resnet_seg_usbmon.sh 的逻辑，使用Python重写

重要修改：
- 替换了有问题的 analyze_usbmon_active.py URB配对算法
- 使用 show_overlap_positions.py 的逐窗口事件解析方法
- 提供真实准确的IN/OUT重叠计算，而非错误的"重叠=OUT时长"

# Analyzer defaults when invoking analyze_usbmon_active.py:
# - STRICT_INVOKE_WINDOW=1                 # strict invoke window as base
# - SHIFT_POLICY=tail_last_BiC_guard_BoS   # tail align to last IN(C), then guard head by BoS
# - SEARCH_TAIL_MS=40                      # tail-side search for last IN (ms)
# - SEARCH_HEAD_MS=40                      # head-side search window for BoS guard (ms)
# - EXTRA_HEAD_EXPAND_MS=10                # allow small head expand to include BoS
# - MAX_SHIFT_MS=50                        # clamp total shift (ms)
# - SPAN_STRICT_PAIR=1                     # span requires both S and C to lie within window (strict S..C)
# - MIN_URB_BYTES=512                   # ignore tiny URBs when picking S/C for span
# - CLUSTER_GAP_MS=0.1                     # IN C-cluster gap (ms) for hybrid IN intervals
# 
# Important: INVOKE_GAP_MS is ms. For sim, prefer 100–200 ms.
"""

import os
import sys
import json
import glob
import subprocess
import statistics
import argparse
from pathlib import Path

# 配置路径
VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"
SYS_PY = "python3"
MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
RESULTS_BASE = "/home/10210/Desktop/OS/results/models_local_batch_usbmon"
CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_capture_offline.sh"
CHAIN_CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_chain_offline.sh"
SIM_CHAIN_CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_chain_offline_sim.sh"
OFFLINE_ALIGN = "/home/10210/Desktop/OS/tools/offline_align_usbmon_ref.py"

# 是否使用链式模式（seg1..seg8 串联 set→invoke→get）
USE_CHAIN_MODE = False
USE_SIM_CHAIN = False  # 是否使用模拟链式（K 组合用单次/不预热/循环100）


def ensure_usbmon_time_map(usbmon_file: str, time_map_file: str, invokes_file: str):
    """Ensure time_map has usbmon_ref by invoking offline aligner if needed."""
    if not (usbmon_file and time_map_file and invokes_file):
        return
    if not os.path.exists(time_map_file) or not os.path.exists(invokes_file):
        return
    try:
        tm = json.load(open(time_map_file))
    except Exception:
        return
    if tm.get('usbmon_ref') is not None:
        return
    if not os.path.exists(OFFLINE_ALIGN):
        return
    py_exec = VENV_PY if os.path.exists(VENV_PY) else SYS_PY
    cmd = [py_exec, OFFLINE_ALIGN, usbmon_file, invokes_file, time_map_file, '--min-urb-bytes', '512']
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            msg = res.stderr.strip() or res.stdout.strip()
            print(f"[warn] 离线 usbmon 对齐失败 ({res.returncode}): {msg}", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] 离线 usbmon 对齐异常: {exc}", file=sys.stderr)

def check_dependencies():
    """检查依赖"""
    if not os.path.exists(VENV_PY):
        print(f"错误：缺少虚拟环境 Python: {VENV_PY}")
        sys.exit(1)
    
    if not os.path.exists(CAPTURE_SCRIPT):
        print(f"错误：缺少捕获脚本: {CAPTURE_SCRIPT}")
        sys.exit(1)

def get_results_base() -> str:
    """根据模式返回结果根目录，避免不同模式互相覆盖。"""
    if USE_CHAIN_MODE and USE_SIM_CHAIN:
        mode = "chain_sim"
    elif USE_CHAIN_MODE:
        mode = "chain"
    else:
        mode = "single"
    return os.path.join(RESULTS_BASE, mode)

def get_usb_bus():
    """获取USB EdgeTPU总线号"""
    # 允许通过环境变量覆盖自动检测（例如 USB_BUS=2）
    env_bus = os.environ.get('USB_BUS') or os.environ.get('EDGETPU_BUS')
    if env_bus and str(env_bus).isdigit():
        return int(env_bus)
    try:
        result = subprocess.run([VENV_PY, "/home/10210/Desktop/OS/list_usb_buses.py"], 
                              capture_output=True, text=True, check=True)
        bus_data = json.loads(result.stdout)
        buses = bus_data.get('buses', [])
        if not buses:
            print("错误：未检测到 USB EdgeTPU 总线号")
            sys.exit(1)
        return buses[0]
    except Exception as e:
        print(f"错误：获取USB总线失败: {e}")
        sys.exit(1)

def find_models():
    """查找所有可用的模型"""
    models = []
    model_patterns = [
        "densenet201_8seg_uniform_local",
        "inceptionv3_8seg_uniform_local", 
        "resnet101_8seg_uniform_local",
        "resnet50_8seg_uniform_local",
        "xception_8seg_uniform_local"
    ]
    
    for pattern in model_patterns:
        model_dir = os.path.join(MODELS_BASE, pattern, "full_split_pipeline_local", "tpu")
        if os.path.exists(model_dir):
            models.append(pattern)
        else:
            print(f"跳过：模型目录不存在 {model_dir}")
    
    # 环境变量过滤：ONLY_MODEL/ONLY_MODELS=逗号分隔的精确模型名
    only_models_env = os.environ.get('ONLY_MODEL') or os.environ.get('ONLY_MODELS')
    if only_models_env:
        wanted = [m.strip() for m in only_models_env.split(',') if m.strip()]
        models = [m for m in models if m in wanted]
        print(f"仅测试模型: {', '.join(models)}")
    return models

def run_segment_test(model_name, seg_num, model_file, bus, outdir):
    """运行单个分段测试"""
    print(f"=== 测试 {model_name} seg{seg_num} -> {outdir} ===")
    
    # 计数规则：用户期望的 COUNT 仅统计“有效样本”（跳过首帧warm）
    # 因此实际运行次数 = COUNT + 1（多跑1次作为warm）
    env = os.environ.copy()
    try:
        req_count = int(env.get('COUNT', '100'))
    except Exception:
        req_count = 100
    env['COUNT'] = str(max(1, req_count) + 1)
    # 添加推理间隔选项，避免长尾IO影响
    if 'INVOKE_GAP_MS' in os.environ:
        env['INVOKE_GAP_MS'] = os.environ['INVOKE_GAP_MS']
    # 标准分析相关缺省（仅记录、便于追溯）
    env.setdefault('STRICT_INVOKE_WINDOW', '1')
    env.setdefault('SHIFT_POLICY', 'tail_last_BiC_guard_BoS')
    env.setdefault('SEARCH_TAIL_MS', '40')
    env.setdefault('SEARCH_HEAD_MS', '40')
    env.setdefault('EXTRA_HEAD_EXPAND_MS', '10')
    env.setdefault('MAX_SHIFT_MS', '50')
    env.setdefault('MIN_URB_BYTES', '512')
    env.setdefault('CLUSTER_GAP_MS', '0.1')
    # 统一写入 CAP_DUR 到 env，便于记录
    if USE_CHAIN_MODE:
        default_cap = '60'
    else:
        default_cap = '45'
    env['CAP_DUR'] = os.environ.get('CAP_DUR', default_cap)

    # 预写运行环境记录
    try:
        os.makedirs(outdir, exist_ok=True)
        keys = [
            'COUNT','INVOKE_GAP_MS','CAP_DUR',
            'STRICT_INVOKE_WINDOW','SHIFT_POLICY','CLUSTER_GAP_MS',
            'ACTIVE_EXPAND_MS','SEARCH_TAIL_MS','SEARCH_HEAD_MS','MAX_SHIFT_MS','USBMON_DEV',
            # off-chip 修正相关（记录用于复现）
            'OFFCHIP_ENABLE','OFFCHIP_OUT_THEORY_MIBPS','OFFCHIP_OUT_MIBPS','OFFCHIP_OUT_THEORY_MIB_PER_MS',
            'MUTATE_INPUT'
        ]
        meta = {k: env.get(k) for k in keys}
        meta.update({
            'COUNT_requested': req_count,
            'bus': bus,
            'use_chain_mode': USE_CHAIN_MODE,
            'use_sim_chain': USE_SIM_CHAIN,
        })
        with open(os.path.join(outdir, 'run_env.json'), 'w') as _f:
            json.dump(meta, _f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    try:
        if USE_CHAIN_MODE:
            # 链式：对所属模型目录一次性生成 seg*/invokes.json，然后各段单独分析
            # 先确保仅在 seg1 时触发链式采集，避免重复
            if seg_num == 1:
                tpu_dir = os.path.join(MODELS_BASE, model_name, "full_split_pipeline_local", "tpu")
                os.makedirs(outdir, exist_ok=True)
                # 估算采集时长：默认 60s，可用 CAP_DUR 覆盖
                cap_dur = env['CAP_DUR']
                if USE_SIM_CHAIN:
                    result = subprocess.run([
                        SIM_CHAIN_CAPTURE_SCRIPT,
                        tpu_dir,
                        f"{model_name}",
                        os.path.dirname(outdir),
                        str(bus),
                        cap_dur
                    ], env=env, check=True)
                else:
                    result = subprocess.run([
                        CHAIN_CAPTURE_SCRIPT,
                        tpu_dir,
                        f"{model_name}",
                        os.path.dirname(outdir),  # 模型级输出目录
                        str(bus),
                        cap_dur
                    ], env=env, check=True)
            else:
                result = subprocess.CompletedProcess(args=[], returncode=0, stdout='', stderr='')
        else:
            # 非链式：默认 45s，可用 CAP_DUR 覆盖
            cap_dur = env['CAP_DUR']
            result = subprocess.run([
                CAPTURE_SCRIPT,
                model_file,
                f"{model_name}_seg{seg_num}",
                outdir,
                str(bus),  # 确保bus是字符串
                cap_dur
            ], env=env, check=True)
        
        print(f"测试完成: {model_name} seg{seg_num}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"测试失败: {model_name} seg{seg_num}")
        print(f"错误: {e.stderr}")
        return False

def parse_overlap_analysis_output(output_text, total_invokes):
    """解析 show_overlap_positions.py 的输出，提取真实的重叠和活跃时间数据"""
    import re
    
    invoke_data = {}
    current_invoke = None
    
    lines = output_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 匹配 invoke 标题
        invoke_match = re.match(r'=== Invoke #(\d+) ===', line)
        if invoke_match:
            current_invoke = int(invoke_match.group(1))
            invoke_data[current_invoke] = {
                'window_ms': 0.0,
                'in_total_ms': 0.0,
                'out_total_ms': 0.0,
                'union_ms': 0.0,
                'overlap_ms': 0.0
            }
            continue
            
        if current_invoke is None:
            continue
            
        # 匹配窗口时长
        window_match = re.match(r'窗口: ([\d.]+)ms', line)
        if window_match:
            invoke_data[current_invoke]['window_ms'] = float(window_match.group(1))
            continue
            
        # 匹配总重叠
        overlap_match = re.match(r'总重叠: ([\d.]+)ms', line)
        if overlap_match:
            invoke_data[current_invoke]['overlap_ms'] = float(overlap_match.group(1))
            continue
            
        # 匹配IN/OUT总时长
        io_match = re.match(r'IN总时长: ([\d.]+)ms, OUT总时长: ([\d.]+)ms', line)
        if io_match:
            invoke_data[current_invoke]['in_total_ms'] = float(io_match.group(1))
            invoke_data[current_invoke]['out_total_ms'] = float(io_match.group(2))
            continue
            
        # 匹配活跃联合
        union_match = re.match(r'活跃联合: ([\d.]+)ms', line)
        if union_match:
            invoke_data[current_invoke]['union_ms'] = float(union_match.group(1))
            continue
    
    return invoke_data

def generate_active_analysis_from_overlap(overlap_data, spans):
    """从逐窗口重叠分析数据生成兼容格式的活跃时间分析结果"""
    per_invoke = []
    
    for i, span in enumerate(spans):
        invoke_index = i
        invoke_span_s = span['end'] - span['begin']
        
        # 从逐窗口分析数据中获取真实值（invoke编号从1开始，但数组索引从0开始）
        real_invoke_num = i + 1
        if real_invoke_num in overlap_data:
            data = overlap_data[real_invoke_num]
            in_active_span_s = data['in_total_ms'] / 1000.0
            out_active_span_s = data['out_total_ms'] / 1000.0  
            union_active_span_s = data['union_ms'] / 1000.0
        else:
            # 如果没有数据，使用保守估计（假设没有重叠）
            in_active_span_s = 0.0
            out_active_span_s = 0.0
            union_active_span_s = 0.0
        
        per_invoke.append({
            "invoke_index": invoke_index,
            "invoke_span_s": invoke_span_s,
            "bytes_in": 0,  # 字节数由其他脚本提供
            "bytes_out": 0,
            "in_active_span_s": in_active_span_s,
            "out_active_span_s": out_active_span_s,
            "union_active_span_s": union_active_span_s,
            "in_events_count": 0,
            "out_events_count": 0
        })
    
    return {
        "model_name": "corrected_overlap_analysis",
        "total_invokes": len(spans),
        "per_invoke": per_invoke
    }

def analyze_performance(outdir, model_name, seg_num):
    """分析性能数据并生成摘要"""
    invokes_file = os.path.join(outdir, "invokes.json")
    # io_split_bt.json 不再是必需；统一使用 analyzer 结果
    usbmon_file = os.path.join(outdir, "usbmon.txt")
    time_map_file = os.path.join(outdir, "time_map.json")
    # 链式模式下，usbmon与time_map位于模型级目录，做就近回退
    if not os.path.exists(usbmon_file):
        p1 = os.path.join(os.path.dirname(outdir), "usbmon.txt")
        p2 = os.path.join(os.path.dirname(os.path.dirname(outdir)), "usbmon.txt")
        if os.path.exists(p1):
            usbmon_file = p1
        elif os.path.exists(p2):
            usbmon_file = p2
    if not os.path.exists(time_map_file):
        p1 = os.path.join(os.path.dirname(outdir), "time_map.json")
        p2 = os.path.join(os.path.dirname(os.path.dirname(outdir)), "time_map.json")
        if os.path.exists(p1):
            time_map_file = p1
        elif os.path.exists(p2):
            time_map_file = p2
    # 读取 run_env（若存在）
    run_env_file = os.path.join(outdir, 'run_env.json')
    run_env = None
    if os.path.exists(run_env_file):
        try:
            with open(run_env_file) as f:
                run_env = json.load(f)
        except Exception:
            run_env = None
    summary_file = os.path.join(outdir, "performance_summary.json")
    active_analysis_strict_file = os.path.join(outdir, "active_analysis_strict.json")
    active_analysis_loose_file = os.path.join(outdir, "active_analysis_loose.json")
    # 统一禁用宽松窗口，严格窗口口径（用户要求 ENABLE_LOOSE=0）
    enable_loose = False
    
    if not os.path.exists(invokes_file):
        return None

    ensure_usbmon_time_map(usbmon_file, time_map_file, invokes_file)

    try:
        # 读取推理时间数据（需要先读取以获取spans）
        with open(invokes_file) as f:
            invokes_data = json.load(f)
        spans = invokes_data.get('spans', [])
        if not spans:
            return None

        # 先运行严格窗口分析器，生成 active_analysis_strict.json，并加载到内存
        active_analysis_strict = None
        try:
            ana_script = "/home/10210/Desktop/OS/analyze_usbmon_active.py"
            if os.path.exists(ana_script) and os.path.exists(usbmon_file) and os.path.exists(time_map_file):
                env_ana = os.environ.copy()
                # 优先采用 run_env 里的配置用于复现；否则给出缺省
                if run_env and isinstance(run_env, dict):
                    for k in [
                        'STRICT_INVOKE_WINDOW','SHIFT_POLICY','CLUSTER_GAP_MS',
                        'ACTIVE_EXPAND_MS','SEARCH_TAIL_MS','SEARCH_HEAD_MS','MAX_SHIFT_MS','USBMON_DEV',
                        # 传递 off-chip 校正环境
                        'OFFCHIP_ENABLE','OFFCHIP_OUT_THEORY_MIBPS','OFFCHIP_OUT_MIBPS','OFFCHIP_OUT_THEORY_MIB_PER_MS'
                    ]:
                        v = run_env.get(k)
                        if v is not None:
                            env_ana[k] = str(v)
                env_ana.setdefault('STRICT_INVOKE_WINDOW', '1')
                env_ana.setdefault('SHIFT_POLICY', 'tail_last_BiC_guard_BoS')
                env_ana.setdefault('SEARCH_TAIL_MS', '40')
                env_ana.setdefault('SEARCH_HEAD_MS', '40')
                env_ana.setdefault('EXTRA_HEAD_EXPAND_MS', '10')
                env_ana.setdefault('MAX_SHIFT_MS', '50')
                env_ana.setdefault('SPAN_STRICT_PAIR', '1')
                env_ana.setdefault('MIN_URB_BYTES', '512')
                env_ana.setdefault('CLUSTER_GAP_MS', '0.1')
                # 开启 off-chip 校正的默认值；未提供理论速率时 analyzer 默认 320 MiB/s
                env_ana.setdefault('OFFCHIP_ENABLE', '1')
                res_ana = subprocess.run(
                    [SYS_PY, ana_script, usbmon_file, invokes_file, time_map_file],
                    capture_output=True, text=True, check=True, env=env_ana
                )
                ana_out = res_ana.stdout or ""
                try:
                    active_analysis_strict = json.loads(ana_out)
                except Exception:
                    # 若 stdout 混入其他输出，尝试定位首个 JSON 对象
                    import re as _re
                    m = _re.search(r"\{.*\}\s*\Z", ana_out, flags=_re.S)
                    if m:
                        active_analysis_strict = json.loads(m.group(0))
                if isinstance(active_analysis_strict, dict):
                    try:
                        with open(active_analysis_strict_file, 'w') as _fo:
                            json.dump(active_analysis_strict, _fo, indent=2, ensure_ascii=False)
                    except Exception:
                        pass
            else:
                active_analysis_strict = None
        except Exception:
            active_analysis_strict = None

    # 计算推理窗口时间统计
        inference_stats = {}
        import statistics as _st
        warm_invokes = spans[1:] if len(spans) > 1 else spans
        invoke_times = [span['end'] - span['begin'] for span in spans]
        invoke_times_ms = [t * 1000.0 for t in invoke_times]
        pure_invoke_times_ms = None
        # 如果已重算纯推理时间，则用新值
        if 'pure_io_times_ms' in locals() and pure_io_times_ms:
            pure_times = pure_io_times_ms[1:] if len(pure_io_times_ms) > 1 else pure_io_times_ms
            inference_stats['pure_invoke_times'] = {
                'mean_ms': _st.mean(pure_times) if pure_times else 0.0,
                'stdev_ms': _st.stdev(pure_times) if pure_times and len(pure_times) > 1 else 0.0,
                'all_ms': pure_times
            }
        # 普通推理窗口时间
        times = invoke_times_ms[1:] if len(invoke_times_ms) > 1 else invoke_times_ms
        inference_stats['invoke_times'] = {
            'mean_ms': _st.mean(times) if times else 0.0,
            'stdev_ms': _st.stdev(times) if times and len(times) > 1 else 0.0,
            'all_ms': times
        }
        
        # 统一使用 analyzer 输出：active_analysis_strict.json
        io_stats = {}
        # 确保已生成严格分析；若缺失，立即执行
        active_analysis_strict = None
        if os.path.exists(active_analysis_strict_file):
            try:
                with open(active_analysis_strict_file) as af:
                    active_analysis_strict = json.load(af)
            except Exception:
                active_analysis_strict = None
        if not isinstance(active_analysis_strict, dict):
            try:
                ana_script = "/home/10210/Desktop/OS/analyze_usbmon_active.py"
                if os.path.exists(ana_script) and os.path.exists(usbmon_file) and os.path.exists(time_map_file):
                    env_ana = os.environ.copy()
                    if run_env and isinstance(run_env, dict):
                        for k in [
                            'STRICT_INVOKE_WINDOW','SHIFT_POLICY','CLUSTER_GAP_MS',
                            'ACTIVE_EXPAND_MS','SEARCH_TAIL_MS','SEARCH_HEAD_MS','MAX_SHIFT_MS','USBMON_DEV',
                            # 传递 off-chip 校正环境
                            'OFFCHIP_ENABLE','OFFCHIP_OUT_THEORY_MIBPS','OFFCHIP_OUT_MIBPS','OFFCHIP_OUT_THEORY_MIB_PER_MS'
                        ]:
                            v = run_env.get(k)
                            if v is not None:
                                env_ana[k] = str(v)
                    env_ana.setdefault('STRICT_INVOKE_WINDOW', '1')
                    env_ana.setdefault('SHIFT_POLICY', 'tail_last_BiC_guard_BoS')
                    env_ana.setdefault('SEARCH_TAIL_MS', '40')
                    env_ana.setdefault('SEARCH_HEAD_MS', '40')
                    env_ana.setdefault('EXTRA_HEAD_EXPAND_MS', '10')
                    env_ana.setdefault('MAX_SHIFT_MS', '50')
                    env_ana.setdefault('SPAN_STRICT_PAIR', '1')
                    env_ana.setdefault('MIN_URB_BYTES', '512')
                    env_ana.setdefault('CLUSTER_GAP_MS', '0.1')
                    env_ana.setdefault('OFFCHIP_ENABLE', '1')
                    res_ana = subprocess.run(
                        [SYS_PY, ana_script, usbmon_file, invokes_file, time_map_file],
                        capture_output=True, text=True, check=True, env=env_ana
                    )
                    ana_out = res_ana.stdout or ""
                    active_analysis_strict = json.loads(ana_out)
                    with open(active_analysis_strict_file, 'w') as _fo:
                        json.dump(active_analysis_strict, _fo, indent=2, ensure_ascii=False)
            except Exception:
                active_analysis_strict = None
        # 提取纯计算时间，以 analyzer 为准
        if isinstance(active_analysis_strict, dict) and 'per_invoke' in active_analysis_strict:
            pv = active_analysis_strict.get('per_invoke', [])
            if pv and invoke_times_ms:
                pure_invoke_times_ms = []
                for i in range(min(len(invoke_times_ms), len(pv))):
                    pc = pv[i].get('pure_compute_ms')
                    if pc is None:
                        pc = pv[i].get('pure_ms_in_only')
                    if pc is not None:
                        pure_invoke_times_ms.append(max(0.0, float(pc)))
                if pure_invoke_times_ms:
                    pure_io_times_ms = pure_invoke_times_ms
            # 计算 IO 聚合速率（ratio-of-sums）
            warm = pv[1:] if len(pv) > 1 else pv
            sum_in_b = sum(int(x.get('bytes_in', 0) or 0) for x in warm)
            sum_out_b = sum(int(x.get('bytes_out', 0) or 0) for x in warm)
            sum_union_s = sum(float(x.get('union_active_span_s', x.get('union_active_s', 0.0)) or 0.0) for x in warm)
            # 避免除零，且单位统一为 MiB/s
            def to_mib(v):
                return v / (1024.0 * 1024.0)
            in_MiBps = (to_mib(sum_in_b) / sum_union_s) if sum_union_s > 0 else 0.0
            out_MiBps = (to_mib(sum_out_b) / sum_union_s) if sum_union_s > 0 else 0.0
            io_stats = {
                'strict_window_analyzer': {
                    'ratio_of_sums': {
                        'sum_bytes_in': sum_in_b,
                        'sum_bytes_out': sum_out_b,
                        'sum_union_active_s': sum_union_s,
                        'in_MiB_per_s': in_MiBps,
                        'out_MiB_per_s': out_MiBps,
                    }
                }
            }

    # 计算基于活跃IO的平均指标（严格；速率默认用严格）
        def compute_active_union_avg(per_invoke_list):
            try:
                if per_invoke_list and len(per_invoke_list) > 1:
                    warm_invokes = per_invoke_list[1:]
                    avg_active_ms = statistics.mean([
                        ( (inv.get('union_active_s', None) if inv.get('union_active_s', None) is not None else inv.get('union_active_span_s', 0.0)) or 0.0) * 1000.0 for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    avg_in_active_ms = statistics.mean([
                        ( (inv.get('in_active_s', None) if inv.get('in_active_s', None) is not None else inv.get('in_active_span_s', 0.0)) or 0.0) * 1000.0 for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    avg_out_active_ms = statistics.mean([
                        ( (inv.get('out_active_s', None) if inv.get('out_active_s', None) is not None else inv.get('out_active_span_s', 0.0)) or 0.0) * 1000.0 for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    avg_in_bytes = statistics.mean([
                        (inv.get('bytes_in', 0) or 0) for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    avg_out_bytes = statistics.mean([
                        (inv.get('bytes_out', 0) or 0) for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    def to_mib(x_bytes: float) -> float:
                        return x_bytes / (1024.0 * 1024.0)
                    if avg_active_ms > 0:
                        mib_per_ms_in = to_mib(avg_in_bytes) / avg_active_ms
                        mib_per_ms_out = to_mib(avg_out_bytes) / avg_active_ms
                        mib_per_s_in = mib_per_ms_in * 1000.0
                        mib_per_s_out = mib_per_ms_out * 1000.0
                    else:
                        mib_per_ms_in = mib_per_ms_out = mib_per_s_in = mib_per_s_out = 0.0
                    return {
                        'avg_active_ms': avg_active_ms,
                        'avg_in_active_ms': avg_in_active_ms,
                        'avg_out_active_ms': avg_out_active_ms,
                        'avg_bytes_in_per_invoke': avg_in_bytes,
                        'avg_bytes_out_per_invoke': avg_out_bytes,
                        'in_MiB_per_ms': mib_per_ms_in,
                        'out_MiB_per_ms': mib_per_ms_out,
                        'in_MiB_per_s': mib_per_s_in,
                        'out_MiB_per_s': mib_per_s_out,
                    }
            except Exception as e:
                return {'note': f'active_union_avg_failed: {e}'}

        # 使用 analyzer 结果计算活跃IO均值（严格）
        active_union_avg_strict = None
        if active_analysis_strict and isinstance(active_analysis_strict, dict):
            try:
                act = active_analysis_strict.get('per_invoke', [])
                if act:
                    import statistics as _st
                    def _get(x,k1,k2,default=0.0):
                        v = x.get(k1, None)
                        return (v if v is not None else x.get(k2, default))
                    active_ms = [ (_get(x,'union_active_s','union_active_span_s',0.0) or 0.0) * 1000.0 for x in act ]
                    avg_active_ms = _st.mean(active_ms) if active_ms else 0.0
                    warm = act[1:] if len(act) > 1 else act
                    avg_in_b = (sum((x.get('bytes_in',0) or 0) for x in warm) / len(warm)) if warm else 0.0
                    avg_out_b = (sum((x.get('bytes_out',0) or 0) for x in warm) / len(warm)) if warm else 0.0
                    def to_mib(v):
                        return v / (1024.0 * 1024.0)
                    if avg_active_ms > 0:
                        in_mib_per_ms = to_mib(avg_in_b) / avg_active_ms
                        out_mib_per_ms = to_mib(avg_out_b) / avg_active_ms
                    else:
                        in_mib_per_ms = out_mib_per_ms = 0.0
                    active_union_avg_strict = {
                        'avg_active_ms': avg_active_ms,
                        'avg_in_active_ms': _st.mean([ (_get(x,'in_active_s','in_active_span_s',0.0) or 0.0) * 1000.0 for x in act ]) if act else 0.0,
                        'avg_out_active_ms': _st.mean([ (_get(x,'out_active_s','out_active_span_s',0.0) or 0.0) * 1000.0 for x in act ]) if act else 0.0,
                        'avg_bytes_in_per_invoke': avg_in_b,
                        'avg_bytes_out_per_invoke': avg_out_b,
                        'in_MiB_per_ms': in_mib_per_ms,
                        'out_MiB_per_ms': out_mib_per_ms,
                        'in_MiB_per_s': in_mib_per_ms * 1000.0,
                        'out_MiB_per_s': out_mib_per_ms * 1000.0,
                    }
            except Exception as _e:
                active_union_avg_strict = {'note': f'hybrid_active_union_failed: {_e}'}
        
    # 生成综合报告
        summary = {
            'model_name': f"{model_name}_seg{seg_num}",
            'segment_number': seg_num,
            'test_timestamp': os.path.getctime(invokes_file),
            'inference_performance': inference_stats,
            'io_performance': io_stats,
            'io_active_union_avg': active_union_avg_strict
        }
        
        # 保存摘要
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        # 也附带一份分析相关 meta，便于追溯
        try:
            meta = {
                'window_meta': (active_analysis_strict or {}).get('window_meta', {}),
                'run_env': run_env or {
                    k: os.environ.get(k) for k in [
                        'INVOKE_GAP_MS','COUNT','CAP_DUR','STRICT_INVOKE_WINDOW','SHIFT_POLICY','CLUSTER_GAP_MS',
                        'ACTIVE_EXPAND_MS','SEARCH_TAIL_MS','SEARCH_HEAD_MS','MAX_SHIFT_MS','USBMON_DEV',
                        'EXTRA_HEAD_EXPAND_MS','MIN_URB_BYTES',
                        'OFFCHIP_ENABLE','OFFCHIP_OUT_THEORY_MIBPS','OFFCHIP_OUT_MIBPS','OFFCHIP_OUT_THEORY_MIB_PER_MS',
                        'MUTATE_INPUT'
                    ]
                }
            }
            with open(os.path.join(outdir, 'analysis_meta.json'), 'w') as mf:
                json.dump(meta, mf, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return summary
    except Exception as e:
        # 失败时输出最小化信息，便于排查
        try:
            with open(os.path.join(outdir, 'analysis_error.txt'), 'w') as ef:
                ef.write(str(e))
        except Exception:
            pass
        return None

def generate_model_summary(model_results_dir, model_name):
    """生成单个模型的整体摘要"""
    segments_data = {}
    for seg_num in range(1, 9):
        seg_dir = os.path.join(model_results_dir, f'seg{seg_num}')
        perf_file = os.path.join(seg_dir, 'performance_summary.json')
        
        if os.path.exists(perf_file):
            try:
                with open(perf_file) as f:
                    seg_data = json.load(f)
                segments_data[f'seg{seg_num}'] = seg_data
            except Exception as e:
                segments_data[f'seg{seg_num}'] = {'error': str(e)}
    
    # 计算整体统计
    total_inference_time = 0
    total_bytes_in = 0
    total_bytes_out = 0
    valid_segments = 0
    
    for seg_name, seg_data in segments_data.items():
        if 'error' not in seg_data:
            inf_perf = seg_data.get('inference_performance', {})
            pure_invoke_perf = inf_perf.get('pure_invoke_times', {})
            io_root = seg_data.get('io_performance', {})
            # 新口径：strict_window_analyzer.ratio_of_sums
            ratio = io_root.get('strict_window_analyzer', {}).get('ratio_of_sums', {})
            if ratio:
                # 估算每次平均字节（仅用于聚合展示，不用于速率）
                total_inference_time += pure_invoke_perf.get('mean_ms', 0)
                inv_count = seg_data.get('inference_performance', {}).get('invoke_times', {}).get('all_ms', [])
                n = len(inv_count) if isinstance(inv_count, list) else 0
                avg_in = (ratio.get('sum_bytes_in', 0)/max(1,n)) if n>0 else 0
                avg_out = (ratio.get('sum_bytes_out', 0)/max(1,n)) if n>0 else 0
                total_bytes_in += avg_in
                total_bytes_out += avg_out
                valid_segments += 1
            else:
                # 旧口径回退
                io_perf = io_root.get('strict_window', {}).get('overall_avg', {})
                total_inference_time += pure_invoke_perf.get('mean_ms', 0)
                total_bytes_in += io_perf.get('avg_bytes_in_per_invoke', 0)
                total_bytes_out += io_perf.get('avg_bytes_out_per_invoke', 0)
                valid_segments += 1
    
    summary = {
        'model_name': model_name,
        'total_segments': len(segments_data),
        'valid_segments': valid_segments,
        'pipeline_performance': {
            'total_inference_time_ms': total_inference_time,
            'total_bytes_in_per_invoke': total_bytes_in,
            'total_bytes_out_per_invoke': total_bytes_out,
            'total_data_transfer_kb': (total_bytes_in + total_bytes_out) / 1024
        },
        'segments_detail': segments_data
    }
    
    # 保存模型摘要
    summary_file = os.path.join(model_results_dir, 'model_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    return summary

def generate_batch_summary(results_base):
    """生成全局摘要报告"""
    all_models = {}
    
    for model_dir in glob.glob(os.path.join(results_base, "*_8seg_uniform_local")):
        model_name = os.path.basename(model_dir)
        summary_file = os.path.join(model_dir, 'model_summary.json')
        
        if os.path.exists(summary_file):
            try:
                with open(summary_file) as f:
                    model_data = json.load(f)
                all_models[model_name] = model_data
            except Exception as e:
                all_models[model_name] = {'error': str(e)}
    
    batch_summary = {
        'test_description': 'models_local批量性能测试',
        'total_models_tested': len(all_models),
        'models': all_models
    }
    
    # 保存批量摘要
    summary_file = os.path.join(results_base, 'batch_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False)
    
    return batch_summary

def main():
    """主函数"""
    # 确保输出按行刷新，便于通过 tail 实时查看
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description='Batch EdgeTPU usbmon test (analyzer-first).')
    parser.add_argument('--model-dir', type=str, default=None, help='Directory containing segN.tflite for custom segment tests')
    parser.add_argument('--outdir', type=str, default=None, help='Output base directory override')
    parser.add_argument('--model-name', type=str, default=None, help='Logical model name for custom tests')
    parser.add_argument('--segs', type=str, default=None, help='Segments selection, e.g. "0-3" or "1,3"')
    args, unknown = parser.parse_known_args()

    print("==========================================")
    print("开始批量测试")
    print("==========================================")
    
    # 检查依赖
    check_dependencies()
    
    # 创建结果目录（按模式分目录，避免覆盖）
    results_base = args.outdir if args.outdir else get_results_base()
    os.makedirs(results_base, exist_ok=True)
    
    # 获取USB总线号
    bus = get_usb_bus()
    print(f"使用 USB 总线: {bus}")
    
    # 自定义 segment 目录模式
    custom_mode = bool(args.model_dir)
    if custom_mode:
        model_dir = os.path.abspath(args.model_dir)
        if not os.path.isdir(model_dir):
            print(f"错误：--model-dir 不存在: {model_dir}")
            sys.exit(1)
        model_name = args.model_name or os.path.basename(model_dir.rstrip('/'))
        # 解析 segs
        seg_list = []
        if args.segs:
            toks = [t.strip() for t in args.segs.split(',') if t.strip()]
            for t in toks:
                if '-' in t:
                    a,b = t.split('-',1)
                    try:
                        a = int(a); b = int(b)
                        if a <= b:
                            seg_list.extend(list(range(a, b+1)))
                    except Exception:
                        pass
                else:
                    try:
                        v = int(t)
                        seg_list.append(v)
                    except Exception:
                        pass
        if not seg_list:
            # 默认根据目录内 seg*.tflite 推断
            for p in sorted(glob.glob(os.path.join(model_dir, 'seg*.tflite'))):
                base = os.path.basename(p)
                m = None
                try:
                    import re as _re
                    m = _re.search(r"seg(\d+)\\.tflite$", base)
                except Exception:
                    m = None
                if m:
                    seg_list.append(int(m.group(1)))
        seg_list = sorted(set(seg_list))
        print(f"自定义模式: {model_name} @ {model_dir}，测试分段: {seg_list}")
        # 运行自定义分段
        bus = get_usb_bus()
        print(f"使用 USB 总线: {bus}")
        model_results = results_base if args.outdir else os.path.join(results_base, model_name)
        os.makedirs(model_results, exist_ok=True)
        for seg_num in seg_list:
            model_file = os.path.join(model_dir, f"seg{seg_num}.tflite")
            outdir = os.path.join(model_results, f"seg{seg_num}")
            if not os.path.isfile(model_file):
                print(f"跳过：模型文件不存在 {model_file}")
                continue
            os.makedirs(outdir, exist_ok=True)
            if run_segment_test(model_name, seg_num, model_file, bus, outdir):
                analyze_performance(outdir, model_name, seg_num)
        # 自定义模式下不生成 1..8 的模型摘要
        print(f"自定义分段测试完成，结果在: {model_results}")
        return

    # 查找所有模型
    models = find_models()
    print(f"找到 {len(models)} 个模型: {', '.join(models)}")
    print()
    
    # 遍历每个模型
    for model_name in models:
        print("=" * 50)
        print(f"开始测试模型: {model_name}")
        print("=" * 50)
        
        model_dir = os.path.join(MODELS_BASE, model_name, "full_split_pipeline_local", "tpu")
        model_results = os.path.join(results_base, model_name)
        os.makedirs(model_results, exist_ok=True)
        
        # 解析分段过滤：ONLY_SEG/ONLY_SEGS 支持 "1" 或 "1,3,5" 或 "2-4"
        seg_list = list(range(1, 9))
        only_segs_env = os.environ.get('ONLY_SEG') or os.environ.get('ONLY_SEGS')
        if only_segs_env:
            toks = [t.strip() for t in only_segs_env.split(',') if t.strip()]
            segs = []
            for t in toks:
                if '-' in t:
                    a,b = t.split('-',1)
                    try:
                        a = int(a); b = int(b)
                        if a <= b:
                            segs.extend(list(range(max(1,a), min(8,b)+1)))
                    except Exception:
                        pass
                else:
                    try:
                        v = int(t)
                        if 1 <= v <= 8:
                            segs.append(v)
                    except Exception:
                        pass
            if segs:
                seg_list = sorted(set(segs))
            print(f"仅测试分段: {seg_list}")

        # 测试每个分段（按过滤结果）
        for seg_num in seg_list:
            model_file = os.path.join(model_dir, f"seg{seg_num}_int8_edgetpu.tflite")
            outdir = os.path.join(model_results, f"seg{seg_num}")
            
            if not os.path.exists(model_file):
                print(f"跳过：模型文件不存在 {model_file}")
                continue
            
            os.makedirs(outdir, exist_ok=True)
            
            # 运行测试
            if run_segment_test(model_name, seg_num, model_file, bus, outdir):
                # 分析性能
                analyze_performance(outdir, model_name, seg_num)
        
        # 生成模型整体摘要
        print(f"=== 生成 {model_name} 整体摘要 ===")
        generate_model_summary(model_results, model_name)
        print(f"{model_name} 完成，详细结果在: {model_results}")
        print()
    
    # 生成全局摘要报告
    print("=" * 50)
    print("生成全局摘要报告")
    print("=" * 50)
    generate_batch_summary(results_base)
    
    print("=" * 50)
    print("批量测试完成！")
    print("=" * 50)
    print(f"结果目录: {results_base}")
    print(f"全局摘要: {results_base}/batch_summary.json")
    print()
    print("各模型结果：")
    for model_name in models:
        model_path = os.path.join(results_base, model_name)
        if os.path.exists(model_path):
            print(f"- {model_name}: {model_path}/")

if __name__ == "__main__":
    main()
