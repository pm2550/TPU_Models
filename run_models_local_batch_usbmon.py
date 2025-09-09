#!/usr/bin/env python3
"""
批量测试 models_local 中所有模型的分段推理性能
基于 tools/run_resnet_seg_usbmon.sh 的逻辑，使用Python重写

重要修改：
- 替换了有问题的 analyze_usbmon_active.py URB配对算法
- 使用 show_overlap_positions.py 的逐窗口事件解析方法
- 提供真实准确的IN/OUT重叠计算，而非错误的"重叠=OUT时长"
"""

import os
import sys
import json
import glob
import subprocess
import statistics
from pathlib import Path

# 配置路径
VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"
SYS_PY = "python3"
MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
RESULTS_BASE = "/home/10210/Desktop/OS/results/models_local_batch_usbmon"
CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_capture_offline.sh"
CHAIN_CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_chain_offline.sh"
SIM_CHAIN_CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_chain_offline_sim.sh"

# 是否使用链式模式（seg1..seg8 串联 set→invoke→get）
USE_CHAIN_MODE = False
USE_SIM_CHAIN = False  # 是否使用模拟链式（K 组合用单次/不预热/循环100）

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
    
    return models

def run_segment_test(model_name, seg_num, model_file, bus, outdir):
    """运行单个分段测试"""
    print(f"=== 测试 {model_name} seg{seg_num} -> {outdir} ===")
    
    # 设置环境变量COUNT=100进行测试（预热改为10次）
    env = os.environ.copy()
    env['COUNT'] = '100'
    
    try:
        if USE_CHAIN_MODE:
            # 链式：对所属模型目录一次性生成 seg*/invokes.json，然后各段单独分析
            # 先确保仅在 seg1 时触发链式采集，避免重复
            if seg_num == 1:
                tpu_dir = os.path.join(MODELS_BASE, model_name, "full_split_pipeline_local", "tpu")
                os.makedirs(outdir, exist_ok=True)
                # 估算采集时长：默认 60s，可用 CAP_DUR 覆盖
                cap_dur = os.environ.get('CAP_DUR', '60')
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
            cap_dur = os.environ.get('CAP_DUR', '45')
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
    io_file = os.path.join(outdir, "io_split_bt.json")
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
    summary_file = os.path.join(outdir, "performance_summary.json")
    active_analysis_strict_file = os.path.join(outdir, "active_analysis_strict.json")
    active_analysis_loose_file = os.path.join(outdir, "active_analysis_loose.json")
    # 统一禁用宽松窗口，严格窗口口径（用户要求 ENABLE_LOOSE=0）
    enable_loose = False
    
    if not os.path.exists(invokes_file) or not os.path.exists(io_file):
        return None
    
    try:
        # 读取推理时间数据（需要先读取以获取spans）
        with open(invokes_file) as f:
            invokes_data = json.load(f)
        
        spans = invokes_data.get('spans', [])
        if not spans:
            return None
        
        # 使用正确的逐窗口事件解析方法计算真实重叠（替换有问题的URB配对算法）
        overlap_script = "/home/10210/Desktop/OS/show_overlap_positions.py"
        if os.path.exists(overlap_script):
            try:
                # 使用逐窗口分析脚本计算真实重叠和活跃时间
                res_overlap = subprocess.run([
                    VENV_PY, overlap_script, usbmon_file, invokes_file, time_map_file,
                    "--start", "1", "--end", "100"  # 覆盖第1次开始（已在采集侧预热）
                ], capture_output=True, text=True, check=True)
                
                # 解析逐窗口分析结果，提取真实的重叠和活跃时间数据
                overlap_data = parse_overlap_analysis_output(res_overlap.stdout, len(spans))
                
                # 生成兼容格式的活跃时间分析结果
                active_analysis_data = generate_active_analysis_from_overlap(overlap_data, spans)
                with open(active_analysis_strict_file, 'w') as f:
                    json.dump(active_analysis_data, f, indent=2)
                
                print(f"✓ 使用逐窗口事件解析方法计算真实重叠（已修复URB配对算法问题）")
                    
            except Exception as e:
                print(f"警告：逐窗口重叠分析失败: {e}")
                # 回退到简化的默认数据
                active_analysis_data = {"model_name": f"{model_name}_seg{seg_num}", "per_invoke": []}
        
        # 使用简化解析器统计严格窗口的字节与速率（统一口径，single/chain 一致）
        simple_script = "/home/10210/Desktop/OS/analyze_usbmon_simple.py"
        simple_result = None
        if os.path.exists(simple_script):
            try:
                res_simple = subprocess.run([
                    "python3", simple_script, usbmon_file, invokes_file, time_map_file
                ], capture_output=True, text=True, check=True)
                simple_result = json.loads(res_simple.stdout)
            except Exception as e:
                print(f"警告：简化USB统计失败: {e}")
        
        # spans已在上面读取，这里不需要重复读取
        
        # 计算invoke时间统计
        invoke_times = [span['end'] - span['begin'] for span in spans]
        invoke_times_ms = [t * 1000 for t in invoke_times]  # 转换为毫秒
        
        # 尝试读取IO活跃时间分析结果
        pure_io_times_ms = []
        active_analysis_strict = None
        active_analysis_loose = None
        if os.path.exists(active_analysis_strict_file):
            try:
                with open(active_analysis_strict_file) as f:
                    active_analysis_strict = json.load(f)
            except Exception:
                active_analysis_strict = None
        if enable_loose and os.path.exists(active_analysis_loose_file):
            try:
                with open(active_analysis_loose_file) as f:
                    active_analysis_loose = json.load(f)
            except Exception:
                active_analysis_loose = None

        # 跳过第一次，使用严格窗口计算纯invoke（并限制活跃IO不超过invoke时长）
        try:
            per_invoke_strict = (active_analysis_strict or {}).get('per_invoke', [])
            if per_invoke_strict:
                pure_invoke_times_ms = []
                for i, inv in enumerate(per_invoke_strict):
                    if i < len(invoke_times_ms):
                        invoke_time_ms = invoke_times_ms[i]
                        io_ms = (inv.get('union_active_span_s', 0.0) or 0.0) * 1000.0
                        io_ms = min(io_ms, invoke_time_ms)
                        pure_invoke_times_ms.append(max(0.0, invoke_time_ms - io_ms))
                pure_io_times_ms = pure_invoke_times_ms
        except Exception as e:
            print(f"警告：计算纯invoke失败: {e}")
        
        # 如果没有纯IO时间，使用invoke时间
        if not pure_io_times_ms:
            pure_io_times_ms = invoke_times_ms
        
        inference_stats = {
            'count': len(invoke_times_ms),
            'invoke_times': {
                'min_ms': min(invoke_times_ms),
                'max_ms': max(invoke_times_ms),
                'mean_ms': statistics.mean(invoke_times_ms),
                'median_ms': statistics.median(invoke_times_ms),
                'stdev_ms': statistics.stdev(invoke_times_ms) if len(invoke_times_ms) > 1 else 0.0,
                'all_times_ms': invoke_times_ms,
                # 保存一次样本：优先第2次（第一帧warm），否则第1次
                'saved_sample_ms': (invoke_times_ms[1] if len(invoke_times_ms) > 1 else (invoke_times_ms[0] if invoke_times_ms else 0.0))
            },
            'pure_invoke_times': {
                'min_ms': min(pure_io_times_ms) if pure_io_times_ms else 0,
                'max_ms': max(pure_io_times_ms) if pure_io_times_ms else 0,
                'mean_ms': statistics.mean(pure_io_times_ms) if pure_io_times_ms else 0,
                'median_ms': statistics.median(pure_io_times_ms) if pure_io_times_ms else 0,
                'stdev_ms': statistics.stdev(pure_io_times_ms) if len(pure_io_times_ms) > 1 else 0.0,
                'all_times_ms': pure_io_times_ms,
                'saved_sample_ms': (pure_io_times_ms[0] if pure_io_times_ms else 0.0)
            }
        }
        
        # 读取USB监控IO数据（窗口平均口径，统一 MiB 单位）；改用 tools/correct_per_invoke_stats.py 输出作为权威
        io_stats = {}
        try:
            corr_script = "/home/10210/Desktop/OS/tools/correct_per_invoke_stats.py"
            if not os.path.exists(corr_script):
                raise FileNotFoundError(corr_script)
            res_corr = subprocess.run([
                VENV_PY, corr_script, usbmon_file, invokes_file, time_map_file, "--extra", "0.010", "--mode", "bulk_complete"
            ], capture_output=True, text=True, check=True)
            txt = res_corr.stdout or ""
            # 优先解析 JSON_SUMMARY
            import re as _re, json as _json
            mjson = _re.search(r"JSON_SUMMARY:\s*(\{.*\})", txt)
            avg_in_bytes = avg_out_bytes = 0.0
            if mjson:
                try:
                    js = _json.loads(mjson.group(1))
                    avg_in_bytes = float(js.get('warm_avg_in_bytes', 0.0) or 0.0)
                    avg_out_bytes = float(js.get('warm_avg_out_bytes', 0.0) or 0.0)
                except Exception:
                    avg_in_bytes = avg_out_bytes = 0.0
            if avg_in_bytes == 0.0 or avg_out_bytes == 0.0:
                # 兼容旧正则（支持千分位）
                m_in = _re.search(r"IN:\s*总字节=([\d,]+),\s*平均=([\d.]+)", txt)
                m_out = _re.search(r"OUT:\s*总字节=([\d,]+),\s*平均=([\d.]+)", txt)
                if m_in and m_out:
                    try:
                        avg_in_bytes = float(m_in.group(2).replace(',', ''))
                        avg_out_bytes = float(m_out.group(2).replace(',', ''))
                    except Exception:
                        pass
                else:
                    m2 = _re.search(r"平均传输:\s*IN=([\d.]+)MB,\s*OUT=([\d.]+)MB", txt)
                    if m2:
                        _in_mib = float(m2.group(1))
                        _out_mib = float(m2.group(2))
                        avg_in_bytes = _in_mib * 1024 * 1024.0
                        avg_out_bytes = _out_mib * 1024 * 1024.0
            # 计算平均窗口时长（跳过第1次）
            all_spans = [(s['end']-s['begin']) for s in spans]
            avg_span_s = (sum(all_spans)/len(all_spans)) if all_spans else 0.0
            to_MiB = lambda b: (b / (1024.0 * 1024.0))
            MiBps_in = (to_MiB(avg_in_bytes) / avg_span_s) if avg_span_s>0 else 0.0
            MiBps_out = (to_MiB(avg_out_bytes) / avg_span_s) if avg_span_s>0 else 0.0
            # 构造 overall，用平均×次数近似总量
            total_windows = max(1, len(all_spans))
            total_in_bytes = int(avg_in_bytes * total_windows)
            total_out_bytes = int(avg_out_bytes * total_windows)
            io_stats = {
                'strict_window': {
                    'overall_avg': {
                        'span_s': avg_span_s,
                        'bytes_in': total_in_bytes,
                        'bytes_out': total_out_bytes,
                        'MiBps_in': MiBps_in,
                        'MiBps_out': MiBps_out,
                        'avg_bytes_in_per_invoke': avg_in_bytes,
                        'avg_bytes_out_per_invoke': avg_out_bytes,
                    }
                }
            }
            # 暂存，供活跃窗口计速回退使用
            _avg_in_bytes_for_active = avg_in_bytes
            _avg_out_bytes_for_active = avg_out_bytes
        except Exception as e:
            io_stats = {'error': f'Failed to compute IO stats (correct_per_invoke_stats): {str(e)}'}
            _avg_in_bytes_for_active = _avg_out_bytes_for_active = 0.0

        # 计算基于活跃IO的平均指标（严格/宽松各一套；速率默认用严格）
        def compute_active_union_avg(per_invoke_list):
            try:
                if per_invoke_list and len(per_invoke_list) > 1:
                    warm_invokes = per_invoke_list[1:]
                    avg_active_ms = statistics.mean([
                        (inv.get('union_active_span_s', 0.0) or 0.0) * 1000.0 for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    avg_in_active_ms = statistics.mean([
                        (inv.get('in_active_span_s', 0.0) or 0.0) * 1000.0 for inv in warm_invokes
                    ]) if warm_invokes else 0.0
                    avg_out_active_ms = statistics.mean([
                        (inv.get('out_active_span_s', 0.0) or 0.0) * 1000.0 for inv in warm_invokes
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

        # 使用“活跃时间(严格)+简化统计字节”的混合方式，计算活跃IO均值（仅严格）
        active_union_avg_strict = None
        if active_analysis_strict and isinstance(active_analysis_strict, dict):
            try:
                act = active_analysis_strict.get('per_invoke', [])
                if act:
                    import statistics as _st
                    active_ms = [ (x.get('union_active_span_s', 0.0) or 0.0) * 1000.0 for x in act ]
                    avg_active_ms = _st.mean(active_ms) if active_ms else 0.0
                    avg_in_b = _avg_in_bytes_for_active
                    avg_out_b = _avg_out_bytes_for_active
                    def to_mib(v):
                        return v / (1024.0 * 1024.0)
                    if avg_active_ms > 0:
                        in_mib_per_ms = to_mib(avg_in_b) / avg_active_ms
                        out_mib_per_ms = to_mib(avg_out_b) / avg_active_ms
                    else:
                        in_mib_per_ms = out_mib_per_ms = 0.0
                    active_union_avg_strict = {
                        'avg_active_ms': avg_active_ms,
                        'avg_in_active_ms': _st.mean([ (x.get('in_active_span_s', 0.0) or 0.0) * 1000.0 for x in act ]) if act else 0.0,
                        'avg_out_active_ms': _st.mean([ (x.get('out_active_span_s', 0.0) or 0.0) * 1000.0 for x in act ]) if act else 0.0,
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
        
        # 显示关键性能指标
        inf = summary.get('inference_performance', {})
        invoke_perf = inf.get('invoke_times', {})
        pure_invoke_perf = inf.get('pure_invoke_times', {})
        io_strict = summary.get('io_performance', {}).get('strict_window', {}).get('overall_avg', {})
        io_active_union_avg_strict = active_union_avg_strict or {}
        
        print(f"--- {model_name} seg{seg_num} 性能摘要 ---")
        print(f"Invoke时间: 平均 {invoke_perf.get('mean_ms', 0):.2f}ms, 标准差 {invoke_perf.get('stdev_ms', 0):.2f}ms")
        print(f"纯Invoke时间: 平均 {pure_invoke_perf.get('mean_ms', 0):.2f}ms, 标准差 {pure_invoke_perf.get('stdev_ms', 0):.2f}ms")
        print(f"数据传输: IN {io_strict.get('avg_bytes_in_per_invoke', 0)/1024/1024:.3f}MiB/次, OUT {io_strict.get('avg_bytes_out_per_invoke', 0)/1024/1024:.3f}MiB/次")
        # 窗口平均速率（MiB/s）
        print(f"窗口平均速率(严格): IN {io_strict.get('MiBps_in', 0):.2f}MiB/s, OUT {io_strict.get('MiBps_out', 0):.2f}MiB/s")
        # 打印严格指标
        print(
            f"活跃IO均值(严格): {io_active_union_avg_strict.get('avg_active_ms', 0):.2f}ms, "
            f"IN {io_active_union_avg_strict.get('avg_in_active_ms', 0):.2f}ms, "
            f"OUT {io_active_union_avg_strict.get('avg_out_active_ms', 0):.2f}ms, "
            f"IN {io_active_union_avg_strict.get('avg_bytes_in_per_invoke', 0)/1024:.1f}KB/次, "
            f"OUT {io_active_union_avg_strict.get('avg_bytes_out_per_invoke', 0)/1024:.1f}KB/次"
        )
        # 按活跃窗口计速（严格，MiB/s）：以活跃窗口作为分母
        try:
            _avg_active_ms = float(io_active_union_avg_strict.get('avg_active_ms', 0) or 0)
            _avg_in_active_ms = float(io_active_union_avg_strict.get('avg_in_active_ms', 0) or 0)
            _avg_out_active_ms = float(io_active_union_avg_strict.get('avg_out_active_ms', 0) or 0)
            _avg_active_s = _avg_active_ms / 1000.0 if _avg_active_ms else 0.0
            _avg_in_active_s = _avg_in_active_ms / 1000.0 if _avg_in_active_ms else 0.0
            _avg_out_active_s = _avg_out_active_ms / 1000.0 if _avg_out_active_ms else 0.0
            _avg_in_bytes = float(io_active_union_avg_strict.get('avg_bytes_in_per_invoke', 0) or 0)
            _avg_out_bytes = float(io_active_union_avg_strict.get('avg_bytes_out_per_invoke', 0) or 0)
            _in_mib = _avg_in_bytes / (1024.0 * 1024.0)
            _out_mib = _avg_out_bytes / (1024.0 * 1024.0)
            # 按方向以各自活跃时长为分母；UNION 以联合活跃时长为分母
            _in_mibps_act = (_in_mib / _avg_in_active_s) if _avg_in_active_s > 0 else 0.0
            _out_mibps_act = (_out_mib / _avg_out_active_s) if _avg_out_active_s > 0 else 0.0
            _union_mibps_act = ((_in_mib + _out_mib) / _avg_active_s) if _avg_active_s > 0 else 0.0
            print(
                f"按活跃窗口计速(严格): IN {_in_mibps_act:.2f}MiB/s, OUT {_out_mibps_act:.2f}MiB/s, UNION {_union_mibps_act:.2f}MiB/s"
            )
        except Exception:
            pass
        # 活跃时长仍展示占用
        # 已禁用宽松窗口输出
        print()
        
        return summary
        
    except Exception as e:
        print(f"性能分析失败: {e}")
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
            io_perf = seg_data.get('io_performance', {}).get('strict_window', {}).get('overall_avg', {})
            
            total_inference_time += pure_invoke_perf.get('mean_ms', 0)  # 使用纯invoke时间（invoke-IO活跃时间）
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
    print("==========================================")
    print("开始批量测试 models_local 中的所有模型")
    print("==========================================")
    
    # 检查依赖
    check_dependencies()
    
    # 创建结果目录（按模式分目录，避免覆盖）
    results_base = get_results_base()
    os.makedirs(results_base, exist_ok=True)
    
    # 获取USB总线号
    bus = get_usb_bus()
    print(f"使用 USB 总线: {bus}")
    
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
        
        # 测试每个分段（seg1-seg8）
        for seg_num in range(1, 9):
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
