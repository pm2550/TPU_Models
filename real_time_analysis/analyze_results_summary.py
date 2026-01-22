#!/usr/bin/env python3
"""
分析 real_time_analysis/results 目录下不同情况的 MN7 和 DeepLab 的统计数据：
- invoke 时间
- 传输时间 (IN/OUT)
- 传输数据量 (bytes_in/bytes_out)
"""

import json
import os
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

def load_json(filepath):
    """加载 JSON 文件"""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"警告: 无法加载 {filepath}: {e}", file=sys.stderr)
        return None

def calc_stats(values):
    """计算统计数据：平均值、最小值、最大值"""
    if not values:
        return {'avg': 0, 'min': 0, 'max': 0, 'count': 0}
    return {
        'avg': sum(values) / len(values),
        'min': min(values),
        'max': max(values),
        'count': len(values)
    }

def analyze_single_scenario(scenario_name, analysis_file, model_name=None, skip_warmup=1, min_bytes_out=100000):
    """分析单个场景的数据"""
    data = load_json(analysis_file)
    if data is None:
        return None
    invokes = data.get('per_invoke', [])
    
    # 跳过预热invoke
    invokes = invokes[skip_warmup:] if len(invokes) > skip_warmup else invokes
    
    # 只保留有足够数据的样本
    invokes = [inv for inv in invokes if (inv.get('bytes_out') or 0) > min_bytes_out]
    
    if not invokes:
        return None
    
    # 提取数据 - 使用 out_union_ms / in_union_ms（并集时长）作为传输时间
    invoke_times_ms = [inv.get('invoke_span_s', 0) * 1000 for inv in invokes]
    # 使用 out_union_ms 作为 OUT 传输时间（并集时长口径）
    out_union_ms = [inv.get('out_union_ms', 0) or 0 for inv in invokes]
    # 使用 in_union_ms 作为 IN 传输时间
    in_union_ms = [inv.get('in_union_ms', 0) or 0 for inv in invokes]
    bytes_in = [inv.get('bytes_in', 0) or 0 for inv in invokes]
    bytes_out = [inv.get('bytes_out', 0) or 0 for inv in invokes]
    pure_ms = [inv.get('pure_compute_ms', 0) or 0 for inv in invokes]
    # 使用 out_speed_mibps / in_speed_mibps（基于 union 时长的速率）
    out_speed = [inv.get('out_speed_mibps', 0) or 0 for inv in invokes]
    in_speed = [inv.get('in_speed_mibps', 0) or 0 for inv in invokes]
    
    return {
        'scenario': scenario_name,
        'model': model_name or data.get('model_name', 'unknown'),
        'total_invokes': data.get('total_invokes', 0),
        'analyzed_invokes': len(invokes),
        'invoke_time_ms': calc_stats(invoke_times_ms),
        'out_transfer_ms': calc_stats(out_union_ms),
        'in_transfer_ms': calc_stats(in_union_ms),
        'bytes_in': calc_stats(bytes_in),
        'bytes_out': calc_stats(bytes_out),
        'pure_compute_ms': calc_stats(pure_ms),
        'out_speed_mibps': calc_stats([s for s in out_speed if s > 0]),
        'in_speed_mibps': calc_stats([s for s in in_speed if s > 0]),
    }

def format_bytes(b):
    """格式化字节数为可读格式"""
    if b >= 1024*1024:
        return f"{b/(1024*1024):.2f} MiB"
    elif b >= 1024:
        return f"{b/1024:.2f} KiB"
    else:
        return f"{b:.0f} B"

def print_scenario_stats(stats, label=""):
    """打印单个场景统计"""
    print(f"\n{'='*60}")
    print(f"场景: {label or stats['scenario']}")
    print(f"模型: {stats['model']}")
    print(f"分析 Invoke 数量: {stats['analyzed_invokes']} (总共: {stats['total_invokes']})")
    print(f"-"*60)
    
    inv = stats['invoke_time_ms']
    print(f"Invoke 时间 (ms):")
    print(f"  平均: {inv['avg']:.3f}  最小: {inv['min']:.3f}  最大: {inv['max']:.3f}")
    
    out_t = stats['out_transfer_ms']
    print(f"OUT 传输时间 (ms):")
    print(f"  平均: {out_t['avg']:.3f}  最小: {out_t['min']:.3f}  最大: {out_t['max']:.3f}")
    
    in_t = stats['in_transfer_ms']
    print(f"IN 传输时间 (ms):")
    print(f"  平均: {in_t['avg']:.3f}  最小: {in_t['min']:.3f}  最大: {in_t['max']:.3f}")
    
    pure = stats['pure_compute_ms']
    print(f"纯计算时间 (ms):")
    print(f"  平均: {pure['avg']:.3f}  最小: {pure['min']:.3f}  最大: {pure['max']:.3f}")
    
    print(f"-"*60)
    bout = stats['bytes_out']
    print(f"OUT 传输数据量:")
    print(f"  平均: {format_bytes(bout['avg'])}  最小: {format_bytes(bout['min'])}  最大: {format_bytes(bout['max'])}")
    
    bin_ = stats['bytes_in']
    print(f"IN 传输数据量:")
    print(f"  平均: {format_bytes(bin_['avg'])}  最小: {format_bytes(bin_['min'])}  最大: {format_bytes(bin_['max'])}")
    
    print(f"-"*60)
    out_s = stats['out_speed_mibps']
    if out_s['count'] > 0:
        print(f"OUT 传输速率 (MiB/s):")
        print(f"  平均: {out_s['avg']:.2f}  最小: {out_s['min']:.2f}  最大: {out_s['max']:.2f}")
    
    in_s = stats['in_speed_mibps']
    if in_s['count'] > 0:
        print(f"IN 传输速率 (MiB/s):")
        print(f"  平均: {in_s['avg']:.2f}  最小: {in_s['min']:.2f}  最大: {in_s['max']:.2f}")

def main():
    print("=" * 80)
    print("real_time_analysis/results 目录统计分析")
    print("统计内容: MN7 和 DeepLab 的 invoke时间、传输时间、传输数据量")
    print("=" * 80)
    
    all_stats = {}
    
    # 1. baseline_single_mn7
    mn7_single_file = RESULTS_DIR / "baseline_single_mn7" / "active_analysis_aligned.json"
    if mn7_single_file.exists():
        stats = analyze_single_scenario("baseline_single_mn7", mn7_single_file, "MN7 (单模型)")
        all_stats['mn7_single'] = stats
        print_scenario_stats(stats, "MN7 单模型基线")
    
    # 2. baseline_single_deeplab
    deeplab_single_file = RESULTS_DIR / "baseline_single_deeplab" / "active_analysis_aligned.json"
    if deeplab_single_file.exists():
        stats = analyze_single_scenario("baseline_single_deeplab", deeplab_single_file, "DeepLab (单模型)")
        all_stats['deeplab_single'] = stats
        print_scenario_stats(stats, "DeepLab 单模型基线")
    
    # 3. dual_mn7_mn7_sync - 双TPU都运行MN7
    dual_mn7_dir = RESULTS_DIR / "dual_mn7_mn7_sync"
    if dual_mn7_dir.exists():
        # 检查是否有设备分开的分析文件
        dev3_file = dual_mn7_dir / "active_analysis_dev3.json"
        dev4_file = dual_mn7_dir / "active_analysis_dev4.json"
        aligned_file = dual_mn7_dir / "active_analysis_aligned.json"
        
        if dev3_file.exists():
            stats = analyze_single_scenario("dual_mn7_mn7_sync_dev3", dev3_file, "MN7 (TPU#3)")
            if stats:
                all_stats['dual_mn7_dev3'] = stats
                print_scenario_stats(stats, "双MN7同步 - TPU设备#3")
        
        if dev4_file.exists():
            stats = analyze_single_scenario("dual_mn7_mn7_sync_dev4", dev4_file, "MN7 (TPU#4)")
            if stats:
                all_stats['dual_mn7_dev4'] = stats
                print_scenario_stats(stats, "双MN7同步 - TPU设备#4")
        
        if aligned_file.exists() and not (dev3_file.exists() or dev4_file.exists()):
            stats = analyze_single_scenario("dual_mn7_mn7_sync", aligned_file, "MN7x2 (合并)")
            if stats:
                all_stats['dual_mn7_merged'] = stats
                print_scenario_stats(stats, "双MN7同步 - 合并统计")
    
    # 4. dual_deeplab_deeplab_sync - 双TPU都运行DeepLab
    dual_deeplab_dir = RESULTS_DIR / "dual_deeplab_deeplab_sync"
    if dual_deeplab_dir.exists():
        dev3_file = dual_deeplab_dir / "active_analysis_dev3.json"
        dev4_file = dual_deeplab_dir / "active_analysis_dev4.json"
        aligned_file = dual_deeplab_dir / "active_analysis_aligned.json"
        
        if dev3_file.exists():
            stats = analyze_single_scenario("dual_deeplab_deeplab_sync_dev3", dev3_file, "DeepLab (TPU#3)")
            if stats:
                all_stats['dual_deeplab_dev3'] = stats
                print_scenario_stats(stats, "双DeepLab同步 - TPU设备#3")
        
        if dev4_file.exists():
            stats = analyze_single_scenario("dual_deeplab_deeplab_sync_dev4", dev4_file, "DeepLab (TPU#4)")
            if stats:
                all_stats['dual_deeplab_dev4'] = stats
                print_scenario_stats(stats, "双DeepLab同步 - TPU设备#4")
        
        if aligned_file.exists() and not (dev3_file.exists() or dev4_file.exists()):
            stats = analyze_single_scenario("dual_deeplab_deeplab_sync", aligned_file, "DeepLabx2 (合并)")
            if stats:
                all_stats['dual_deeplab_merged'] = stats
                print_scenario_stats(stats, "双DeepLab同步 - 合并统计")
    
    # 5. dual_deeplab_mn7_sync - 混合场景
    # 根据数据特征判断模型：MN7 ~8ms invoke，DeepLab ~50ms invoke
    dual_mixed_dir = RESULTS_DIR / "dual_deeplab_mn7_sync"
    if dual_mixed_dir.exists():
        dev3_file = dual_mixed_dir / "active_analysis_dev3.json"
        dev4_file = dual_mixed_dir / "active_analysis_dev4.json"
        aligned_file = dual_mixed_dir / "active_analysis_aligned.json"
        
        if dev3_file.exists():
            stats = analyze_single_scenario("dual_deeplab_mn7_sync_dev3", dev3_file, "")
            if stats:
                # 根据 invoke 时间判断模型类型
                avg_invoke = stats['invoke_time_ms']['avg']
                if avg_invoke < 15:  # MN7 特征
                    model_label = "混合-MN7 (dev3)"
                    all_stats['dual_mixed_mn7'] = stats
                else:  # DeepLab 特征
                    model_label = "混合-DeepLab (dev3)"
                    all_stats['dual_mixed_deeplab'] = stats
                stats['model'] = model_label
                print_scenario_stats(stats, f"混合模式 - {model_label}")
        
        if dev4_file.exists():
            stats = analyze_single_scenario("dual_deeplab_mn7_sync_dev4", dev4_file, "")
            if stats:
                avg_invoke = stats['invoke_time_ms']['avg']
                if avg_invoke < 15:  # MN7 特征
                    model_label = "混合-MN7 (dev4)"
                    if 'dual_mixed_mn7' not in all_stats:
                        all_stats['dual_mixed_mn7'] = stats
                else:  # DeepLab 特征
                    model_label = "混合-DeepLab (dev4)"
                    if 'dual_mixed_deeplab' not in all_stats:
                        all_stats['dual_mixed_deeplab'] = stats
                stats['model'] = model_label
                print_scenario_stats(stats, f"混合模式 - {model_label}")
        
        if aligned_file.exists() and not (dev3_file.exists() or dev4_file.exists()):
            stats = analyze_single_scenario("dual_deeplab_mn7_sync", aligned_file, "混合模式 (合并)")
            if stats:
                all_stats['dual_mixed_merged'] = stats
                print_scenario_stats(stats, "DeepLab+MN7混合 - 合并统计")
    
    # 打印对比汇总表
    print("\n" + "=" * 80)
    print("对比汇总表")
    print("=" * 80)
    
    # 创建汇总表格
    print(f"\n{'场景':<30} {'Invoke(ms)':<12} {'OUT传输(ms)':<12} {'IN传输(ms)':<12} {'纯计算(ms)':<12}")
    print("-" * 80)
    
    for key, stats in all_stats.items():
        inv = stats['invoke_time_ms']['avg']
        out_t = stats['out_transfer_ms']['avg']
        in_t = stats['in_transfer_ms']['avg']
        pure = stats['pure_compute_ms']['avg']
        name = f"{stats['model']}"[:28]
        print(f"{name:<30} {inv:<12.3f} {out_t:<12.3f} {in_t:<12.3f} {pure:<12.3f}")
    
    print("\n" + "-" * 80)
    print(f"\n{'场景':<30} {'OUT数据量':<15} {'IN数据量':<15} {'OUT速率(MiB/s)':<15}")
    print("-" * 80)
    
    for key, stats in all_stats.items():
        bout = stats['bytes_out']['avg']
        bin_ = stats['bytes_in']['avg']
        out_s = stats['out_speed_mibps']['avg']
        name = f"{stats['model']}"[:28]
        print(f"{name:<30} {format_bytes(bout):<15} {format_bytes(bin_):<15} {out_s:<15.2f}")
    
    print("\n" + "=" * 80)
    
    return all_stats

if __name__ == '__main__':
    main()

