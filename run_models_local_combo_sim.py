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
    # 无预热，脚本内部固定每段一次、循环100次
    cmd = [SIM_CHAIN_CAPTURE_SCRIPT, tpu_dir, model_name, out_dir, bus, "15"]
    return subprocess.run(cmd, capture_output=True, text=True)

def run_real_chain(tpu_dir: str, model_name: str, out_dir: str, bus: str):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    # 真实链式：脚本内部预热10次，每段记录100次 invoke；默认持续 20s，可通过 CAP_DUR 覆盖
    dur = os.environ.get('CAP_DUR', '20')
    cmd = [CHAIN_CAPTURE_SCRIPT, tpu_dir, model_name, out_dir, bus, dur]
    return subprocess.run(cmd, capture_output=True, text=True)

def analyze_performance(combo_root: str, seg_dir: str, model_name: str, seg_num: int):
    usbmon_file = os.path.join(combo_root, "usbmon.txt")
    time_map_file = os.path.join(combo_root, "time_map.json")
    invokes_file = os.path.join(seg_dir, "invokes.json")
    io_file = os.path.join(seg_dir, "io_split_bt.json")
    active_file = os.path.join(seg_dir, "active_analysis.json")
    summary_file = os.path.join(seg_dir, "performance_summary.json")

    if not (os.path.exists(usbmon_file) and os.path.exists(time_map_file) and os.path.exists(invokes_file) and os.path.exists(io_file)):
        return None

    # 活跃IO分析（严格窗口；ACTIVE_EXPAND_MS=0），仅用于纯invoke扣除
    try:
        env = os.environ.copy(); env['ACTIVE_EXPAND_MS'] = '0'
        res = subprocess.run([SYS_PY, ANALYZE_ACTIVE, usbmon_file, invokes_file, time_map_file], capture_output=True, text=True, check=True, env=env)
        with open(active_file, 'w') as f:
            f.write(res.stdout)
        active_data = json.loads(res.stdout or "{}")
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

    # IO 统计（全局最大重叠分配）：一次性处理整段窗口，避免跨段重复
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
            # 合并 seg1..seg8 的 invokes.json
            merged = {'spans': []}
            for s in range(1, 9):
                seg_inv = os.path.join(combo_root, f"seg{s}", "invokes.json")
                if not os.path.exists(seg_inv):
                    continue
                J = json.load(open(seg_inv))
                for sp in (J.get('spans') or []):
                    merged['spans'].append({'begin': sp['begin'], 'end': sp['end'], 'seg': s})
            # 排序保证时间顺序
            merged['spans'].sort(key=lambda x: x['begin'])
            json.dump(merged, open(merged_file, 'w'))

        # 运行权威统计，使用 overlap + 最大重叠分配（脚本内实现），一次性覆盖所有窗口
        if not os.path.exists(CORRECT_PER_INVOKE):
            raise FileNotFoundError(CORRECT_PER_INVOKE)
        res_corr = subprocess.run([VENV_PY, CORRECT_PER_INVOKE, usbmon_file, merged_file, time_map_file, "--extra", "0.010", "--mode", "bulk_complete", "--include", "overlap"], capture_output=True, text=True, check=True)
        txt = res_corr.stdout or ""
        import re as _re, json as _json
        # 解析 JSON_SUMMARY（全局平均）
        avg_in_bytes = avg_out_bytes = 0.0
        mjson = _re.search(r"JSON_SUMMARY:\s*(\{.*\})", txt)
        if mjson:
            try:
                js = _json.loads(mjson.group(1))
                avg_in_bytes = float(js.get('warm_avg_in_bytes', 0.0) or 0.0)
                avg_out_bytes = float(js.get('warm_avg_out_bytes', 0.0) or 0.0)
            except Exception:
                avg_in_bytes = avg_out_bytes = 0.0
        # 用本段的窗口均值换算速率（避免用全局 span）
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
        print("开始：K=2..8 组合，链式模拟（无预热、每段一次、循环100次）")
    else:
        print("开始：K=2..8 组合，真实链式（预热10次、循环100次）")
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

            # 分析各段
            for seg in range(1, 9):
                seg_dir = os.path.join(combo_out, f"seg{seg}")
                os.makedirs(seg_dir, exist_ok=True)
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


