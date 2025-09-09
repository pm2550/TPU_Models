#!/usr/bin/env python3
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
    # 无预热，脚本内部固定每段一次、循环100次
    cmd = [SIM_CHAIN_CAPTURE_SCRIPT, tpu_dir, model_name, out_dir, bus, "15"]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)

def run_real_chain(tpu_dir: str, model_name: str, out_dir: str, bus: str):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    # 真实链式（combo）：默认不预热、每段1次；若外部已提供 WARMUP/COUNT，则尊重外部
    env.setdefault('WARMUP', '0')
    env.setdefault('COUNT', '1')
    # 默认持续 20s，可通过 CAP_DUR 覆盖
    dur = os.environ.get('CAP_DUR', '20')
    cmd = [CHAIN_CAPTURE_SCRIPT, tpu_dir, model_name, out_dir, bus, dur]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)

def analyze_performance(combo_root: str, seg_dir: str, model_name: str, seg_num: int):
    usbmon_file = os.path.join(combo_root, "usbmon.txt")
    time_map_file = os.path.join(combo_root, "time_map.json")
    invokes_file = os.path.join(seg_dir, "invokes.json")
    io_file = os.path.join(seg_dir, "io_split_bt.json")
    active_file = os.path.join(seg_dir, "active_analysis.json")
    summary_file = os.path.join(seg_dir, "performance_summary.json")

    if not (os.path.exists(usbmon_file) and os.path.exists(time_map_file) and os.path.exists(invokes_file) and os.path.exists(io_file)):
        return None

    # 活跃IO分析：统一改用 show_overlap_positions.py（严格窗口、事件精配）
    SHOW_OVERLAP = "/home/10210/Desktop/OS/show_overlap_positions.py"
    try:
        res = subprocess.run([VENV_PY, SHOW_OVERLAP, usbmon_file, invokes_file, time_map_file], capture_output=True, text=True, check=True)
        with open(active_file, 'w') as f:
            f.write(res.stdout)
        # 复用 local_batch 的解析/整形逻辑：按文本解析关键数，再生成 per_invoke
        from run_models_local_batch_usbmon import parse_overlap_analysis_output, generate_active_analysis_from_overlap
        overlap_data = parse_overlap_analysis_output(res.stdout, 0)
        active_data = generate_active_analysis_from_overlap(overlap_data, json.load(open(invokes_file)).get('spans', []))
    except Exception:
        active_data = {}

    # 读取invoke spans
    inv = json.load(open(invokes_file))
    spans = inv.get('spans', [])
    if not spans:
        return None

    # 计算invoke与纯invoke
    invoke_ms = [(s['end']-s['begin'])*1000.0 for s in spans]
    pure_ms = []
    per = active_data.get('per_invoke', [])
    for i in range(len(spans)):
        io_ms = (per[i].get('union_active_span_s', 0.0)*1000.0) if i < len(per) else 0.0
        pure_ms.append(max(0.0, invoke_ms[i]-io_ms))

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
        if not os.path.exists(merged_file):
            # 合并 combo_root 下所有以 seg 开头的目录（支持 seg2to8 等标签）
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
            # 排序保证时间顺序
            merged['spans'].sort(key=lambda x: x['begin'])
            json.dump(merged, open(merged_file, 'w'))

        # 运行权威统计，使用 overlap + 最大重叠分配（脚本内实现），一次性覆盖所有窗口
        if not os.path.exists(CORRECT_PER_INVOKE):
            raise FileNotFoundError(CORRECT_PER_INVOKE)
        res_corr = subprocess.run([VENV_PY, CORRECT_PER_INVOKE, usbmon_file, merged_file, time_map_file, "--extra", "0.010", "--mode", "bulk_complete", "--include", "overlap"], capture_output=True, text=True, check=True)
        txt = res_corr.stdout or ""
        import re as _re, json as _json
        # 解析 per-invoke 明细（JSON_PER_INVOKE），按 seg 聚合出本段平均字节
        avg_in_bytes = avg_out_bytes = 0.0
        mj = _re.search(r"JSON_PER_INVOKE:\s*(\[.*\])", txt)
        if mj:
            try:
                arr = _json.loads(mj.group(1))
                # 按 seg_label 聚合：从 merged_invokes.json 中筛出当前段的窗口索引
                merged = _json.loads(open(merged_file).read())
                all_spans = merged.get('spans', [])
                seg_label = os.path.basename(seg_dir)
                idxs = [i for i, ss in enumerate(all_spans) if ss.get('seg_label') == seg_label]
                vals_in = [arr[i]['bytes_in'] for i in idxs if 0 <= i < len(arr)]
                vals_out = [arr[i]['bytes_out'] for i in idxs if 0 <= i < len(arr)]
                if vals_in:
                    avg_in_bytes = sum(vals_in)/len(vals_in)
                if vals_out:
                    avg_out_bytes = sum(vals_out)/len(vals_out)
            except Exception:
                avg_in_bytes = avg_out_bytes = 0.0
        # 用本段的窗口均值换算速率
        avg_span_s = (sum(invoke_ms)/len(invoke_ms)/1000.0) if invoke_ms else 0.0
        to_MiB = lambda b: (b/(1024.0*1024.0))
        overall_avg = {
            'span_s': avg_span_s,
            'avg_bytes_in_per_invoke': avg_in_bytes,
            'avg_bytes_out_per_invoke': avg_out_bytes,
            'MiBps_in': (to_MiB(avg_in_bytes)/avg_span_s) if avg_span_s>0 else 0.0,
            'MiBps_out': (to_MiB(avg_out_bytes)/avg_span_s) if avg_span_s>0 else 0.0,
        }
    except Exception as _:
        pass

    # 活跃IO均值（严格窗口，按并集，仅展示占用，不再用于速率）
    active_union_avg = None
    if per:
        avg_active_ms = statistics.mean([(x.get('union_active_span_s',0.0)*1000.0) for x in per])
        avg_in_bytes = statistics.mean([x.get('bytes_in',0) for x in per])
        avg_out_bytes = statistics.mean([x.get('bytes_out',0) for x in per])
        active_union_avg = {
            'avg_active_ms': avg_active_ms,
            'avg_bytes_in_per_invoke': avg_in_bytes,
            'avg_bytes_out_per_invoke': avg_out_bytes,
        }

    summary = {
        'model_name': model_name,
        'segment_number': seg_num,
        'inference_performance': {
            'invoke_times': stat(invoke_ms),
            'pure_invoke_times': stat(pure_ms),
        },
        'io_performance': {
            'strict_window': {
                'overall_avg': overall_avg
            }
        },
        'io_active_union_avg': active_union_avg,
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

            # 分析各段（依据实际产生的 seg*/invokes.json 动态决定阶段数）
            available_segs = []
            for seg in range(1, 9):
                invp = os.path.join(combo_out, f"seg{seg}", "invokes.json")
                if os.path.isfile(invp):
                    available_segs.append(seg)
            if not available_segs:
                print("(分析失败) 未发现任何 seg*/invokes.json")
            for seg in available_segs:
                seg_dir = os.path.join(combo_out, f"seg{seg}")
                s = analyze_performance(combo_out, seg_dir, f"{model_name}_K{k}", seg)
                if s:
                    inv = s['inference_performance']['invoke_times']
                    pure = s['inference_performance']['pure_invoke_times']
                    avg = s.get('io_performance',{}).get('strict_window',{}).get('overall_avg',{})
                    print(f"--- {model_name} K{k} seg{seg} ---")
                    print(f"Invoke均值: {inv.get('mean_ms',0):.2f}ms, 纯Invoke均值: {pure.get('mean_ms',0):.2f}ms")
                    print(f"IN/OUT字节: {avg.get('avg_bytes_in_per_invoke',0)/1024/1024:.3f}MiB/{avg.get('avg_bytes_out_per_invoke',0)/1024/1024:.3f}MiB")
                else:
                    print(f"(分析失败) {model_name} K{k} seg{seg}")

    print("==========================================")
    print("完成：组合模拟链式批量测试")
    print("==========================================")

if __name__ == "__main__":
    main()


