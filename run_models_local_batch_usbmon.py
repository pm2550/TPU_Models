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
            # 映射失败则回退使用 0u（所有总线）再试一次
            try:
                tm_path = os.path.join(outdir, 'time_map.json')
                tm_ok = False
                if os.path.exists(tm_path):
                    with open(tm_path) as _f:
                        tmj = json.load(_f)
                    tm_ok = (tmj.get('usbmon_ref') is not None) and (tmj.get('boottime_ref') is not None)
                if not tm_ok:
                    print("time_map 未就绪，回退到 usbmon 0u 重新采集一次…")
                    result = subprocess.run([
                        CAPTURE_SCRIPT,
                        model_file,
                        f"{model_name}_seg{seg_num}",
                        outdir,
                        '0',
                        cap_dur
                    ], env=env, check=True)
            except Exception as _e:
                print(f"回退重试检查失败: {_e}")
        
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

        # 预载可选的外部活跃分析结果（若存在）以避免 NameError
        active_analysis_strict = None
        try:
            if os.path.exists(active_analysis_strict_file):
                with open(active_analysis_strict_file) as _f:
                    active_analysis_strict = json.load(_f)
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
        
        # 使用简化解析器统计严格窗口的字节与速率（统一 MiB 单位）；改用 tools/correct_per_invoke_stats.py 输出作为权威
        io_stats = {}
        per_invoke_from_corr = None
        try:
            corr_script = "/home/10210/Desktop/OS/tools/correct_per_invoke_stats.py"
            if not os.path.exists(corr_script):
                raise FileNotFoundError(corr_script)
            # 优先严格窗口(full)且不扩展(extra=0)，必要时再小幅放宽
            extras = []
            try:
                extras.append(float(os.environ.get('EXTRA_S', '0.000')))
            except Exception:
                extras.append(0.000)
            for v in (0.005, 0.010):
                if all(abs(v - x) > 1e-6 for x in extras):
                    extras.append(v)
            txt = ""
            chosen_extra = extras[0]
            import re as _re, json as _json
            # 预计算每个窗口的时长(含warm)用于基本合理性校验
            _spans_s = [(s['end']-s['begin']) for s in spans]
            _avg_span = (sum(_spans_s)/len(_spans_s)) if _spans_s else 0.0
            for _extra in extras:
                res_corr = subprocess.run([
                    VENV_PY, corr_script, usbmon_file, invokes_file, time_map_file,
                    "--mode", "bulk_complete", "--include", "full",
                    "--extra", f"{_extra:.3f}"
                ], capture_output=True, text=True, check=True)
                txt = res_corr.stdout or ""
                mp = _re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", txt, flags=_re.S)
                if not mp:
                    continue
                try:
                    arr_try = _json.loads(mp.group(1))
                except Exception:
                    continue
                # 接受条件：至少一半warm窗口有字节，且 union_active 不显著超过窗口
                warm_vals = [x.get('bytes_in', 0) + x.get('bytes_out', 0) for x in arr_try[1:]] if len(arr_try) > 1 else [x.get('bytes_in', 0) + x.get('bytes_out', 0) for x in arr_try]
                ok_bytes = sum(1 for v in warm_vals if v > 0) >= max(1, int(0.5 * len(warm_vals)))
                try:
                    unions = [float(x.get('union_active_s') or 0.0) for x in arr_try]
                    # 容忍极小浮点误差：union 不应大于窗口均值的 110%
                    ok_union = (max(unions[1:], default=0.0) <= (_avg_span * 1.10 + 1e-6)) if _avg_span > 0 else True
                except Exception:
                    ok_union = True
                if ok_bytes and ok_union:
                    chosen_extra = _extra
                    per_invoke_from_corr = arr_try
                    break
            # 保存 stdout，便于排错
            try:
                with open(os.path.join(outdir, 'correct_per_invoke_stdout.txt'), 'w') as _fo:
                    _fo.write(txt)
            except Exception:
                pass
            # 若尚未解析成功，再解析一次（不影响 chosen_extra）
            if per_invoke_from_corr is None:
                mp = _re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", txt, flags=_re.S)
                if mp:
                    try:
                        per_invoke_from_corr = _json.loads(mp.group(1))
                    except Exception:
                        per_invoke_from_corr = None
            # 解析 JSON_SUMMARY 获取 warm 平均字节
            mjson = _re.search(r"JSON_SUMMARY:\s*(\{.*\})", txt)
            avg_in_bytes = avg_out_bytes = 0.0
            if mjson:
                try:
                    js = _json.loads(mjson.group(1))
                    avg_in_bytes = float(js.get('warm_avg_in_bytes', 0.0) or 0.0)
                    avg_out_bytes = float(js.get('warm_avg_out_bytes', 0.0) or 0.0)
                except Exception:
                    avg_in_bytes = avg_out_bytes = 0.0
            # 回退解析（历史格式）
            if avg_in_bytes == 0.0 and avg_out_bytes == 0.0:
                m2 = _re.search(r"平均传输:\s*IN=([\d.]+)MB,\s*OUT=([\d.]+)MB", txt)
                if m2:
                    _in_mib = float(m2.group(1)); _out_mib = float(m2.group(2))
                    avg_in_bytes = _in_mib * 1024 * 1024.0
                    avg_out_bytes = _out_mib * 1024 * 1024.0
            # 窗口均值速率
            all_spans = [(s['end']-s['begin']) for s in spans]
            avg_span_s = (sum(all_spans)/len(all_spans)) if all_spans else 0.0
            to_MiB = lambda b: (b / (1024.0 * 1024.0))
            MiBps_in = (to_MiB(avg_in_bytes) / avg_span_s) if avg_span_s>0 else 0.0
            MiBps_out = (to_MiB(avg_out_bytes) / avg_span_s) if avg_span_s>0 else 0.0
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

            # 若 unionActive 明显大于窗口，触发一次 overlap 回退以剪裁到窗口内
            def _max_union_vs_span(_arr):
                try:
                    u = [float(x.get('union_active_s') or 0.0) for x in _arr]
                    return max(u[1:], default=0.0)
                except Exception:
                    return 0.0
            max_union = _max_union_vs_span(per_invoke_from_corr or [])
            if per_invoke_from_corr and avg_span_s>0 and max_union > avg_span_s*1.10:
                try:
                    res_corr2 = subprocess.run([
                        VENV_PY, corr_script, usbmon_file, invokes_file, time_map_file,
                        "--mode", "bulk_complete", "--include", "overlap",
                        "--extra", "0.000"
                    ], capture_output=True, text=True, check=True)
                    txt2 = res_corr2.stdout or ""
                    mp2 = _re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", txt2, flags=_re.S)
                    if mp2:
                        per_invoke_from_corr = _json.loads(mp2.group(1))
                        # 使用 overlap 的 JSON_SUMMARY 更新均值
                        mjson2 = _re.search(r"JSON_SUMMARY:\s*(\{.*\})", txt2)
                        if mjson2:
                            js2 = _json.loads(mjson2.group(1))
                            avg_in_bytes = float(js2.get('warm_avg_in_bytes', avg_in_bytes) or avg_in_bytes)
                            avg_out_bytes = float(js2.get('warm_avg_out_bytes', avg_out_bytes) or avg_out_bytes)
                            MiBps_in = (to_MiB(avg_in_bytes) / avg_span_s) if avg_span_s>0 else 0.0
                            MiBps_out = (to_MiB(avg_out_bytes) / avg_span_s) if avg_span_s>0 else 0.0
                            total_in_bytes = int(avg_in_bytes * total_windows)
                            total_out_bytes = int(avg_out_bytes * total_windows)
                            io_stats['strict_window']['overall_avg'].update({
                                'bytes_in': total_in_bytes,
                                'bytes_out': total_out_bytes,
                                'MiBps_in': MiBps_in,
                                'MiBps_out': MiBps_out,
                                'avg_bytes_in_per_invoke': avg_in_bytes,
                                'avg_bytes_out_per_invoke': avg_out_bytes,
                            })
                except Exception:
                    pass
        except Exception as e:
            io_stats = {'error': f'Failed to compute IO stats (correct_per_invoke_stats): {str(e)}'}
            per_invoke_from_corr = per_invoke_from_corr  # keep whatever parsed
            _avg_in_bytes_for_active = 0.0
            _avg_out_bytes_for_active = 0.0
        # 若解析器 per_invoke 可用，则用其真并集重算纯invoke
        if per_invoke_from_corr and invoke_times_ms:
            try:
                pure_invoke_times_ms = []
                for i in range(min(len(invoke_times_ms), len(per_invoke_from_corr))):
                    union_s = per_invoke_from_corr[i].get('union_active_s', 0.0) or 0.0
                    io_ms = float(union_s) * 1000.0
                    pure_invoke_times_ms.append(max(0.0, invoke_times_ms[i] - io_ms))
                if pure_invoke_times_ms:
                    pure_io_times_ms = pure_invoke_times_ms
            except Exception:
                pass

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

        # 使用“活跃时间(严格)+简化统计字节”的混合方式，计算活跃IO均值（仅严格）
        active_union_avg_strict = None
        # 若解析器已给出逐次结果，则优先使用解析器的 per_invoke 进行活跃均值计算
        if per_invoke_from_corr:
            try:
                act = per_invoke_from_corr
                import statistics as _st
                def _get(x,k1,k2,default=0.0):
                    v = x.get(k1, None)
                    return (v if v is not None else x.get(k2, default))
                active_ms = [ (_get(x,'union_active_s','union_active_span_s',0.0) or 0.0) * 1000.0 for x in act ]
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
        elif active_analysis_strict and isinstance(active_analysis_strict, dict):
            try:
                act = active_analysis_strict.get('per_invoke', [])
                if act:
                    import statistics as _st
                    def _get(x,k1,k2,default=0.0):
                        v = x.get(k1, None)
                        return (v if v is not None else x.get(k2, default))
                    active_ms = [ (_get(x,'union_active_s','union_active_span_s',0.0) or 0.0) * 1000.0 for x in act ]
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
    # 确保输出按行刷新，便于通过 tail 实时查看
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("==========================================")
    print("开始批量测试 models_local 中的所有模型 或 单模型目录")
    print("==========================================")

    # CLI：支持单模型目录与自定义输出目录
    parser = argparse.ArgumentParser(description='models_local 批量/单模型 usbmon 采集与分析')
    parser.add_argument('--model-dir', dest='single_model_dir', default=None,
                        help='单模型目录（包含 seg*.tflite，例如 seg0.tflite、seg1.tflite）')
    parser.add_argument('--outdir', dest='single_outdir', default=None,
                        help='自定义输出根目录（将创建 segX 子目录）')
    parser.add_argument('--model-name', dest='single_model_name', default=None,
                        help='单模型名称标签（用于结果标注，默认取目录名）')
    parser.add_argument('--segs', dest='single_segs', default=None,
                        help='仅测试的分段列表，例如 "0-3" 或 "0,1,2,3"；留空则自动扫描 seg*.tflite')
    args, _ = parser.parse_known_args()
    
    # 检查依赖
    check_dependencies()

    # 获取USB总线号
    bus = get_usb_bus()
    print(f"使用 USB 总线: {bus}")

    # 单模型自定义模式
    if args.single_model_dir and args.single_outdir:
        single_dir = os.path.abspath(args.single_model_dir)
        out_root = os.path.abspath(args.single_outdir)
        model_name = args.single_model_name or os.path.basename(single_dir.rstrip('/'))

        # 解析 seg 列表：优先 --segs，否则扫描 seg*.tflite
        seg_list = []
        if args.single_segs:
            toks = [t.strip() for t in args.single_segs.split(',') if t.strip()]
            for t in toks:
                if '-' in t:
                    a, b = t.split('-', 1)
                    try:
                        a = int(a); b = int(b)
                        if a <= b:
                            seg_list.extend(list(range(a, b + 1)))
                    except Exception:
                        pass
                else:
                    try:
                        seg_list.append(int(t))
                    except Exception:
                        pass
            seg_list = sorted(set(seg_list))
        else:
            for p in glob.glob(os.path.join(single_dir, 'seg*.tflite')):
                base = os.path.basename(p)
                m = None
                try:
                    # 支持 seg{num}.tflite 或 seg{num}_*.tflite
                    m = [int(''.join(ch for ch in base.split('.')[0] if ch.isdigit()))]
                except Exception:
                    m = None
                if m:
                    seg_list.extend(m)
            seg_list = sorted(set(seg_list))

        if not seg_list:
            print(f"未找到分段文件：{single_dir}/seg*.tflite")
            sys.exit(1)

        print(f"单模型模式: {model_name}")
        print(f"模型目录: {single_dir}")
        print(f"输出目录: {out_root}")
        print(f"分段: {seg_list}")

        os.makedirs(out_root, exist_ok=True)
        models = [model_name]

        for seg_num in seg_list:
            # 支持 seg{n}.tflite 或 seg{n}_*.tflite（优先精确匹配）
            candidates = [
                os.path.join(single_dir, f"seg{seg_num}.tflite")
            ] + sorted(glob.glob(os.path.join(single_dir, f"seg{seg_num}_*.tflite")))
            model_file = next((p for p in candidates if os.path.exists(p)), None)
            outdir = os.path.join(out_root, f"seg{seg_num}")

            if not model_file:
                print(f"跳过：模型文件不存在 seg{seg_num}（{single_dir}）")
                continue

            os.makedirs(outdir, exist_ok=True)
            print(f"=== 测试 {model_name} seg{seg_num} -> {outdir} ===")
            if run_segment_test(model_name, seg_num, model_file, bus, outdir):
                analyze_performance(outdir, model_name, seg_num)

        # 生成简单摘要
        print(f"=== 生成 {model_name} 简要摘要（通配 seg*） ===")
        # 复用批量摘要生成但遍历 seg* 目录
        segments_data = {}
        for seg_path in sorted(glob.glob(os.path.join(out_root, 'seg*'))):
            perf_file = os.path.join(seg_path, 'performance_summary.json')
            seg_key = os.path.basename(seg_path)
            if os.path.exists(perf_file):
                try:
                    with open(perf_file) as f:
                        seg_data = json.load(f)
                    segments_data[seg_key] = seg_data
                except Exception as e:
                    segments_data[seg_key] = {'error': str(e)}
        model_summary = {
            'model_name': model_name,
            'segments': list(segments_data.keys()),
        }
        with open(os.path.join(out_root, 'model_summary.json'), 'w') as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)

        print(f"单模型完成，结果在: {out_root}")
        return

    # 批量模式（原始逻辑）
    # 创建结果目录（按模式分目录，避免覆盖）
    results_base = get_results_base()
    os.makedirs(results_base, exist_ok=True)

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
