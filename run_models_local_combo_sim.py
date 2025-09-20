#!/usr/bin/env python3
#
# Batch combo (sim/chain) runner and analyzer
#
# Per-invoke gap (sleep between invokes):
# - INVOKE_GAP_MS=100               # milliseconds; default 100ms
#   sim: 作为唯一间隔参数传入模拟脚本
#   chain: 映射为 STAGE_GAP_MS 传入真实链脚本（两者单位同为毫秒）
#
# Analyzer defaults when invoking analyze_usbmon_active.py:
# - STRICT_INVOKE_WINDOW=1                 # strict invoke window as base
# - SHIFT_POLICY=tail_last_BiC_guard_BoS   # tail align to last IN(C), then guard head by BoS
# - SEARCH_TAIL_MS=40                      # tail-side search for last IN (ms)
# - SEARCH_HEAD_MS=40                      # head-side search window for BoS guard (ms)
# - EXTRA_HEAD_EXPAND_MS=10                # allow small head expand to include BoS
# - MAX_SHIFT_MS=50                        # clamp total shift (ms)
# - SPAN_STRICT_PAIR=1                     # span requires both S and C to lie within window (strict S..C)
# - MIN_URB_BYTES=65536                    # ignore tiny URBs when picking S/C for span
# - CLUSTER_GAP_MS=0.1                     # IN C-cluster gap (ms) for hybrid IN intervals
# 
# Important: INVOKE_GAP_MS is ms. For sim, prefer 50–200 ms.
import os
import sys
import json
import glob
import subprocess
import statistics
import argparse

VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"
SYS_PY = "python3"
MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
# 保持原模拟结果目录不变；真实链式写入独立目录避免覆盖
RESULTS_BASE_SIM = "/home/10210/Desktop/OS/results/models_local_combo_sim"
RESULTS_BASE_CHAIN = "/home/10210/Desktop/OS/results/models_local_combo_chain"
SIM_CHAIN_CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_chain_offline_sim.sh"
CHAIN_CAPTURE_SCRIPT = "/home/10210/Desktop/OS/run_usbmon_chain_offline.sh"
ANALYZE_ACTIVE = "/home/10210/Desktop/OS/analyze_usbmon_active.py"
CORRECT_PER_INVOKE = "/home/10210/Desktop/OS/tools/correct_per_invoke_stats.py"

MODELS = [
    "densenet201_8seg_uniform_local",
    "inceptionv3_8seg_uniform_local",
    "mobilenetv2_8seg_uniform_local",
    "resnet101_8seg_uniform_local",
    "resnet50_8seg_uniform_local",
    "xception_8seg_uniform_local",
]

def check_deps(mode: str):
    if mode == 'sim':
        if not os.path.exists(SIM_CHAIN_CAPTURE_SCRIPT):
            print(f"缺少脚本: {SIM_CHAIN_CAPTURE_SCRIPT}")
            sys.exit(1)
    else:
        if not os.path.exists(CHAIN_CAPTURE_SCRIPT):
            print(f"缺少脚本: {CHAIN_CAPTURE_SCRIPT}")
            sys.exit(1)
    if not os.path.exists(ANALYZE_ACTIVE):
        print(f"缺少脚本: {ANALYZE_ACTIVE}")
        sys.exit(1)

def get_bus() -> str:
    try:
        out = subprocess.run([VENV_PY, "/home/10210/Desktop/OS/list_usb_buses.py"], capture_output=True, text=True, check=True).stdout
        data = json.loads(out or "{}")
        buses = data.get("buses") or []
        return str(buses[0]) if buses else ""
    except Exception as e:
        print("获取USB总线失败", e)
        return ""

def run_sim_chain(tpu_dir: str, model_name: str, out_dir: str, bus: str):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('WARMUP', '0')
    # 只允许使用 INVOKE_GAP_MS（毫秒），默认 100ms；shell 脚本侧会自行换算为秒
    try:
        ms = float(env.get('INVOKE_GAP_MS', '100'))
    except Exception:
        ms = 100.0
    env['INVOKE_GAP_MS'] = f"{ms:.3f}"
    print(f"[sim] 每次invoke间隔: {env['INVOKE_GAP_MS']} ms")
    dur = env.get('CAP_DUR', '120')
    cmd = [SIM_CHAIN_CAPTURE_SCRIPT, tpu_dir, model_name, out_dir, bus, dur]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)

def run_real_chain(tpu_dir: str, model_name: str, out_dir: str, bus: str):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    # 真实链式（combo）：默认不预热、每段1次；若外部已提供 WARMUP/COUNT，则尊重外部
    env.setdefault('WARMUP', '0')
    env.setdefault('COUNT', '1')
    env.setdefault('STOP_ON_COUNT', '1')
    # 统一：将 INVOKE_GAP_MS（毫秒）映射为 STAGE_GAP_MS（毫秒）供真实链使用
    if 'INVOKE_GAP_MS' in env and str(env['INVOKE_GAP_MS']).strip() != '':
        env['STAGE_GAP_MS'] = env['INVOKE_GAP_MS']
        try:
            print(f"[chain] 每段间隔(STAGE_GAP_MS)来源 INVOKE_GAP_MS: {env['INVOKE_GAP_MS']} ms")
        except Exception:
            pass
    # 默认持续 60s，可通过 CAP_DUR 覆盖
    dur = os.environ.get('CAP_DUR', '60')
    cmd = [CHAIN_CAPTURE_SCRIPT, tpu_dir, model_name, out_dir, bus, dur]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)

def analyze_performance(combo_root: str, seg_dir: str, model_name: str, seg_label: str):
    usbmon_file = os.path.join(combo_root, "usbmon.txt")
    time_map_file = os.path.join(combo_root, "time_map.json")
    invokes_file = os.path.join(seg_dir, "invokes.json")
    active_file = os.path.join(seg_dir, "active_analysis.json")
    summary_file = os.path.join(seg_dir, "performance_summary.json")

    # 仅要求 usbmon/time_map/invokes 存在即可分析
    if not (os.path.exists(usbmon_file) and os.path.exists(time_map_file) and os.path.exists(invokes_file)):
        return None

    # 活跃IO分析：改为基于 correct_per_invoke_stats 的全局“最大重叠分配”结果（跨窗去重）
    active_data = {}

    # 读取invoke spans
    inv = json.load(open(invokes_file))
    spans = inv.get('spans', [])
    if not spans:
        return None

    # 计算invoke与纯invoke
    invoke_ms = [(s['end']-s['begin'])*1000.0 for s in spans]
    pure_ms = []
    per = active_data.get('per_invoke', [])  # 若后面填充
    # 先占位，稍后用 JSON_PER_INVOKE 替换 per
    per_union_active_s = [0.0]*len(spans)

    def stat(arr):
        return {
            'min_ms': min(arr) if arr else 0.0,
            'max_ms': max(arr) if arr else 0.0,
            'mean_ms': statistics.mean(arr) if arr else 0.0,
            'median_ms': statistics.median(arr) if arr else 0.0,
            'stdev_ms': statistics.stdev(arr) if len(arr)>1 else 0.0,
            'count': len(arr),
        }

    # IO 统计（全局最大重叠分配）：一次性处理整段窗口，并按段聚合 per-invoke
    overall_avg = {
        'span_s': 0.0,
        'avg_bytes_in_per_invoke': 0.0,
        'avg_bytes_out_per_invoke': 0.0,
        'MiBps_in': 0.0,
        'MiBps_out': 0.0,
    }
    try:
        # 为 combo 的“全局分配”改造：在 combo 根目录准备 merged_invokes.json（包含所有段窗口及 seg 索引）
        merged_file = os.path.join(combo_root, "merged_invokes.json")
        # 每次分析都重新合并，避免使用过期窗口导致 0 字节
        merged = {'spans': []}
        for name in sorted(os.listdir(combo_root)):
            seg_path = os.path.join(combo_root, name)
            if not (os.path.isdir(seg_path) and name.startswith('seg')):
                continue
            seg_inv = os.path.join(seg_path, 'invokes.json')
            if not os.path.exists(seg_inv):
                continue
            J = json.load(open(seg_inv))
            for sp in (J.get('spans') or []):
                merged['spans'].append({'begin': sp['begin'], 'end': sp['end'], 'seg_label': name})
        merged['spans'].sort(key=lambda x: x['begin'])
        json.dump(merged, open(merged_file, 'w'))

        # 运行权威统计，使用 overlap + 最大重叠分配（脚本内实现），一次性覆盖所有窗口
        if not os.path.exists(CORRECT_PER_INVOKE):
            raise FileNotFoundError(CORRECT_PER_INVOKE)
        # 确保 merged_invokes.json 为最新（每次重建）
        try:
            merged = {'spans': []}
            for name in sorted(os.listdir(combo_root)):
                seg_path = os.path.join(combo_root, name)
                if not (os.path.isdir(seg_path) and name.startswith('seg')):
                    continue
                seg_inv = os.path.join(seg_path, 'invokes.json')
                if not os.path.exists(seg_inv):
                    continue
                J = json.load(open(seg_inv))
                for sp in (J.get('spans') or []):
                    merged['spans'].append({'begin': sp['begin'], 'end': sp['end'], 'seg_label': name})
            merged['spans'].sort(key=lambda x: x['begin'])
            json.dump(merged, open(merged_file, 'w'))
        except Exception:
            pass

        # 尝试不同的 --extra（窗口扩展），以避免 seg1 的输入发生在 invoke 之前而被漏记
        extras_to_try = []
        try:
            extras_to_try.append(float(os.environ.get('EXTRA_S', '0.010')))
        except Exception:
            extras_to_try.append(0.010)
        # 进一步加大窗口（针对真实链式，可能存在较大提前量）
        for v in (0.200, 1.000):
            if all(abs(v - x) > 1e-6 for x in extras_to_try):
                extras_to_try.append(v)

        txt = ""
        chosen_extra = extras_to_try[0]
        best_total_bytes = 0
        import re as _re, json as _json
        seg_label_local = os.path.basename(seg_dir)
        
        # 尝试所有窗口扩展，选择捕获数据最多的
        for _extra in extras_to_try:
            res_corr = subprocess.run([VENV_PY, CORRECT_PER_INVOKE, usbmon_file, merged_file, time_map_file,
                                       "--extra", f"{_extra:.3f}", "--mode", "bulk_complete", "--include", "overlap"],
                                      capture_output=True, text=True, check=True)
            txt_try = res_corr.stdout or ""
            # 解析 per-invoke 结果，计算总数据量
            mj = _re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", txt_try, flags=_re.S)
            if mj:
                try:
                    arr_try = _json.loads(mj.group(1))
                    # 计算所有 invoke 的总 IN+OUT 字节数
                    total_bytes = sum(float(item.get('bytes_in', 0.0)) + float(item.get('bytes_out', 0.0)) 
                                    for item in arr_try)
                    if total_bytes > best_total_bytes:
                        best_total_bytes = total_bytes
                        chosen_extra = _extra
                        txt = txt_try
                except Exception:
                    pass
        # 将完整输出保存，便于排错
        try:
            with open(os.path.join(combo_root, "correct_per_invoke_stdout.txt"), 'w') as _fo:
                _fo.write(txt)
        except Exception:
            pass
        # 不向终端输出统计详情，保持原始格式
        import re as _re, json as _json
        # 解析 per-invoke 明细（JSON_PER_INVOKE），按 seg 聚合出本段平均字节与活跃时长（去重后，真并集）
        avg_in_bytes = None
        avg_out_bytes = None
        mj = _re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", txt, flags=_re.S)
        if mj:
            try:
                arr = _json.loads(mj.group(1))
                # 按 seg_label 聚合：从 merged_invokes.json 中筛出当前段的窗口索引
                merged = _json.loads(open(merged_file).read())
                all_spans = merged.get('spans', [])
                seg_label_local = os.path.basename(seg_dir)
                idxs = [i for i, ss in enumerate(all_spans) if ss.get('seg_label') == seg_label_local]
                vals_in = [float(arr[i].get('bytes_in', 0.0)) for i in idxs if 0 <= i < len(arr)]
                vals_out = [float(arr[i].get('bytes_out', 0.0)) for i in idxs if 0 <= i < len(arr)]
                # 真并集：优先使用 union_active_s，回退到 max(in_active_s,out_active_s)
                per_union_active_s = []
                for i in idxs:
                    if 0 <= i < len(arr):
                        ua = arr[i].get('union_active_s')
                        if ua is None:
                            ua = max(float(arr[i].get('in_active_s', 0.0)), float(arr[i].get('out_active_s', 0.0)))
                        per_union_active_s.append(float(ua))
                if vals_in:
                    avg_in_bytes = sum(vals_in)/len(vals_in)
                if vals_out:
                    avg_out_bytes = sum(vals_out)/len(vals_out)
            except Exception:
                avg_in_bytes = avg_out_bytes = None
        # 回退：解析 JSON_SUMMARY（兼容新老字段名）
        if avg_in_bytes is None or avg_out_bytes is None:
            try:
                ms = _re.search(r"JSON_SUMMARY:\s*(\{.*?\})", txt, flags=_re.S)
                if ms:
                    sj = _json.loads(ms.group(1))
                    v_in = sj.get('avg_bytes_in_per_invoke')
                    v_out = sj.get('avg_bytes_out_per_invoke')
                    if (v_in is None) and ('warm_avg_in_bytes' in sj):
                        v_in = sj.get('warm_avg_in_bytes')
                    if (v_out is None) and ('warm_avg_out_bytes' in sj):
                        v_out = sj.get('warm_avg_out_bytes')
                    if avg_in_bytes is None:
                        avg_in_bytes = float(v_in or 0.0)
                    if avg_out_bytes is None:
                        avg_out_bytes = float(v_out or 0.0)
            except Exception:
                pass
        # 用本段的窗口均值换算速率
        avg_span_s = (sum(invoke_ms)/len(invoke_ms)/1000.0) if invoke_ms else 0.0
        to_MiB = lambda b: (b/(1024.0*1024.0))
        overall_avg = {
            'span_s': avg_span_s,
            'avg_bytes_in_per_invoke': float(avg_in_bytes or 0.0),
            'avg_bytes_out_per_invoke': float(avg_out_bytes or 0.0),
            'MiBps_in': (to_MiB(avg_in_bytes)/avg_span_s) if avg_span_s>0 else 0.0,
            'MiBps_out': (to_MiB(avg_out_bytes)/avg_span_s) if avg_span_s>0 else 0.0,
        }
    except Exception as _:
        pass

    # 活跃IO均值（严格窗口，并集＝max(in_active,out_active)，URB 已去重）
    active_union_avg = None
    if per_union_active_s:
        avg_active_ms = statistics.mean([s*1000.0 for s in per_union_active_s])
        active_union_avg = {
            'avg_active_ms': avg_active_ms,
            'avg_bytes_in_per_invoke': overall_avg.get('avg_bytes_in_per_invoke', 0.0),
            'avg_bytes_out_per_invoke': overall_avg.get('avg_bytes_out_per_invoke', 0.0),
        }

    # 用分配后的活跃时间（真并集）重算纯净 invoke（不强制 union<=invoke）
    if per_union_active_s:
        pure_ms = [max(0.0, iv - ua*1000.0) for iv, ua in zip(invoke_ms, per_union_active_s)]
    else:
        pure_ms = [iv for iv in invoke_ms]

    # 尝试解析段序号（用于展示）；优先 segX or segXtoY 的起始 X
    import re as _re
    m = _re.match(r"seg(\d+)", seg_label)
    seg_num = int(m.group(1)) if m else None

    # 运行严格窗口分析器，获取 per-invoke 的 in/out span 与纯间隙（last Co->next Bi）
    active_spans_summary = None
    try:
        if os.path.exists(ANALYZE_ACTIVE):
            env = os.environ.copy()
            # 严格窗口 + 尾对齐IN并以BoS守卫头部 的默认参数（可被外部覆盖）
            env.setdefault('STRICT_INVOKE_WINDOW', '1')
            env.setdefault('SHIFT_POLICY', 'tail_last_BiC_guard_BoS')
            env.setdefault('SEARCH_TAIL_MS', '40')
            env.setdefault('SEARCH_HEAD_MS', '40')
            env.setdefault('EXTRA_HEAD_EXPAND_MS', '10')
            env.setdefault('MAX_SHIFT_MS', '50')
            env.setdefault('SPAN_STRICT_PAIR', '1')
            env.setdefault('MIN_URB_BYTES', '65536')
            env.setdefault('CLUSTER_GAP_MS', '0.1')
            res = subprocess.run([SYS_PY, ANALYZE_ACTIVE, usbmon_file, invokes_file, time_map_file],
                                 capture_output=True, text=True, env=env, check=False)
            if res.returncode == 0 and res.stdout:
                try:
                    J = json.loads(res.stdout)
                except Exception:
                    # 有些版本直接打印 JSON
                    # 也可能包含前缀行，尝试从最后一个 { 开始解析
                    s = res.stdout
                    idx = s.rfind('{')
                    if idx >= 0:
                        J = json.loads(s[idx:])
                    else:
                        J = None
                if J and isinstance(J, dict):
                    per = J.get('per_invoke') or []
                    if per:
                        def safe_vals(key):
                            vals = []
                            for r in per:
                                v = r.get(key)
                                if isinstance(v, (int, float)):
                                    vals.append(float(v))
                            return vals
                        in_spans = safe_vals('in_span_sc_ms')
                        out_spans = safe_vals('out_span_sc_ms')
                        pure_gaps = safe_vals('pure_gap_lastCo_to_nextBi_ms')
                        import statistics as _st
                        active_spans_summary = {
                            'mean_in_span_sc_ms': (_st.mean(in_spans) if in_spans else 0.0),
                            'mean_out_span_sc_ms': (_st.mean(out_spans) if out_spans else 0.0),
                            'mean_pure_gap_lastCo_to_nextBi_ms': (_st.mean(pure_gaps) if pure_gaps else 0.0),
                            'count': len(per),
                        }
                        # 保存原始分析输出，便于回溯
                        try:
                            with open(active_file, 'w') as af:
                                json.dump(J, af, ensure_ascii=False, indent=2)
                        except Exception:
                            pass
    except Exception:
        active_spans_summary = None

    summary = {
        'model_name': model_name,
        'segment_label': seg_label,
        'segment_number': seg_num,
        'inference_performance': {
            'invoke_times': {**stat(invoke_ms), 'all_times_ms': invoke_ms},
            'pure_invoke_times': {**stat(pure_ms), 'all_times_ms': pure_ms},
        },
        'io_performance': {
            'strict_window': {
                'overall_avg': overall_avg
            }
        },
        'io_active_union_avg': active_union_avg,
        'active_spans_summary': active_spans_summary,
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary

def main():
    parser = argparse.ArgumentParser(description='组合批量测试（模拟链式或真实链式）')
    parser.add_argument('--mode', choices=['sim','chain'], default='sim', help='sim: 模拟链式; chain: 真实链式')
    parser.add_argument('--model', help='仅测试某个模型名（可选）')
    parser.add_argument('--k', help='仅测试某个K（2-8，可选）')
    args = parser.parse_args()

    mode = args.mode
    print("==========================================")
    if mode == 'sim':
        print("开始：K=2..8 组合，链式模拟（无预热、每段1次）")
    else:
        print("开始：K=2..8 组合，真实链式（无预热、每段1次）")
    print("==========================================")

    check_deps(mode)
    results_base = RESULTS_BASE_SIM if mode == 'sim' else RESULTS_BASE_CHAIN
    os.makedirs(results_base, exist_ok=True)
    bus = get_bus()
    if not bus:
        print("未检测到 USB EdgeTPU 总线号")
        sys.exit(1)
    print("使用 BUS=", bus)

    targets = [args.model] if args.model else MODELS
    for model_name in targets:
        print("==================================================")
        print("开始模型:", model_name)
        print("==================================================")
        model_root = os.path.join(results_base, model_name)
        os.makedirs(model_root, exist_ok=True)

        # 遍历 K=2..7 组合目录
        combos = []
        # K=2..7: 从 combos_K{K}_run1/tpu 读取
        for k in range(2, 8):
            tp = os.path.join(MODELS_BASE, model_name, f"combos_K{k}_run1", "tpu")
            if os.path.isdir(tp):
                combos.append((k, tp))
        # K=8: 使用 full_split_pipeline_local/tpu 作为“全部拆分”的组合
        full_tp = os.path.join(MODELS_BASE, model_name, "full_split_pipeline_local", "tpu")
        if os.path.isdir(full_tp):
            combos.append((8, full_tp))
        if not combos:
            print("(跳过：未找到任何组合目录，含 K=8 full split)")
            continue

        for k, tpu_dir in combos:
            if args.k and str(k) != str(args.k):
                continue
            combo_out = os.path.join(model_root, f"K{k}")
            print(f"=== 组合 K{k}: {tpu_dir} -> {combo_out}")
            if mode == 'sim':
                r = run_sim_chain(tpu_dir, f"{model_name}_K{k}", combo_out, bus)
            else:
                r = run_real_chain(tpu_dir, f"{model_name}_K{k}", combo_out, bus)
            if r.returncode != 0:
                print("采集失败:", r.stderr[-200:])
                continue

            # 分析各段（动态发现以 seg 开头的目录，支持 segXtoY）
            import re as _re
            seg_dirs = [d for d in sorted(os.listdir(combo_out)) if d.startswith('seg') and os.path.isdir(os.path.join(combo_out, d))]
            if not seg_dirs:
                print("(分析失败) 未发现任何 seg*/invokes.json")
            # 依据起始段号排序（segX 或 segXtoY 的 X）
            def seg_key(lbl: str):
                m = _re.match(r"seg(\d+)", lbl)
                return int(m.group(1)) if m else 999
            seg_dirs.sort(key=seg_key)
            # 仅保留本K应有的段
            # - K=2..7：seg1..seg{k-1} + seg{k}to8（若存在）
            # - K=8：完整拆分，使用 seg1..seg8（目录名为 seg1, seg2, ..., seg8）
            if k == 8:
                filtered = [lbl for lbl in seg_dirs if _re.fullmatch(r"seg[1-8]", lbl)]
            else:
                expected = {f"seg{i}": True for i in range(1, k)}
                tail_label = f"seg{k}to8"
                filtered = []
                for lbl in seg_dirs:
                    if lbl in expected or lbl == tail_label:
                        filtered.append(lbl)
            seg_dirs = filtered
            for seg_lbl in seg_dirs:
                seg_dir = os.path.join(combo_out, seg_lbl)
                invp = os.path.join(seg_dir, 'invokes.json')
                if not os.path.isfile(invp):
                    continue
                s = analyze_performance(combo_out, seg_dir, f"{model_name}_K{k}", seg_lbl)
                if s:
                    inv = s['inference_performance']['invoke_times']
                    pure = s['inference_performance']['pure_invoke_times']
                    avg = s.get('io_performance',{}).get('strict_window',{}).get('overall_avg',{})
                    print(f"--- {model_name} K{k} {seg_lbl} ---")
                    print(f"Invoke均值: {inv.get('mean_ms',0):.2f}ms, 纯Invoke均值: {pure.get('mean_ms',0):.2f}ms")
                    print(f"IN/OUT字节: {avg.get('avg_bytes_in_per_invoke',0)/1024/1024:.3f}MiB/{avg.get('avg_bytes_out_per_invoke',0)/1024/1024:.3f}MiB")
                    act = s.get('active_spans_summary') or {}
                    if act:
                        print(f"in_span均值: {act.get('mean_in_span_sc_ms',0):.2f}ms, out_span均值: {act.get('mean_out_span_sc_ms',0):.2f}ms, 纯间隙(lastCo->nextBi)均值: {act.get('mean_pure_gap_lastCo_to_nextBi_ms',0):.2f}ms")
                else:
                    print(f"(分析失败) {model_name} K{k} {seg_lbl}")

    print("==========================================")
    print("完成：组合模拟链式批量测试")
    print("==========================================")

if __name__ == "__main__":
    main()
