#!/usr/bin/env python3

import json
import sys
import argparse

def parse_usbmon_line(line):
    """解析 usbmon 行，返回 (timestamp, event, direction, nbytes) 或 None"""
    parts = line.split()
    if len(parts) < 8:
        return None
    
    try:
        ts_us = int(parts[1])
        ts = ts_us / 1000000.0  # 转换为秒
        ev = parts[2]  # S 或 C
        endpoint_info = parts[3]  # 如 Bi:2:007:1
        
        if ':' in endpoint_info:
            d = endpoint_info.split(':')[0]  # 提取 Bi, Bo, Ci, Co
        else:
            return None
            
        # 提取字节数（usbmon 格式：第6列是字节数）
        nb = 0
        if len(parts) > 5:
            try:
                nb = int(parts[5])
            except ValueError:
                nb = 0
                
        return (ts, ev, d, nb)
    except (ValueError, IndexError):
        return None

def calculate_active_time_density(timestamps, window_start, window_end, bin_size_ms=1.0):
    """
    计算活跃时间密度：将窗口分成小段，统计有传输的时间段
    
    Args:
        timestamps: 传输时间戳列表
        window_start, window_end: 窗口范围
        bin_size_ms: 时间段大小（毫秒）
    
    Returns:
        活跃时间（秒）
    """
    if not timestamps:
        return 0.0
    
    bin_size = bin_size_ms / 1000.0  # 转换为秒
    window_duration = window_end - window_start
    
    # 创建时间段
    num_bins = int(window_duration / bin_size) + 1
    active_bins = set()
    
    # 标记有传输的时间段
    for ts in timestamps:
        if window_start <= ts <= window_end:
            bin_idx = int((ts - window_start) / bin_size)
            active_bins.add(bin_idx)
    
    # 返回活跃时间段的总时长
    return len(active_bins) * bin_size

def stat_window(usbmon_file, b0, e0, mode='bulk_complete', expand_s=0.0):
    """统计窗口内的传输数据"""
    
    # 扩展窗口
    b0 -= expand_s
    e0 += expand_s
    
    bi = bo = ci = co = 0
    binb = boutb = 0
    in_times = []
    out_times = []
    
    with open(usbmon_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            parsed = parse_usbmon_line(line)
            if not parsed:
                continue
                
            ts, ev, d, nb = parsed
            
            # 检查是否在窗口内
            if b0 <= ts <= e0:
                if mode == 'bulk_complete':
                    # 仅统计完成事件 C，且仅统计 Bulk（Bi/Bo）
                    if ev == 'C' and d in ('Bi', 'Bo'):
                        if d == 'Bi':
                            bi += 1
                            binb += nb
                            in_times.append(ts)
                        elif d == 'Bo':
                            bo += 1
                            boutb += nb
                            out_times.append(ts)
                    elif d == 'Ci':
                        ci += 1
                    elif d == 'Co':
                        co += 1
    
    span = e0 - b0
    toMB = lambda x: x / (1000 * 1000.0)  # 转换为 MB (1000^2)
    
    # 使用新的活跃时间计算方法
    window_start = b0 + expand_s  # 原始窗口开始
    window_end = e0 - expand_s    # 原始窗口结束
    
    in_active_s = calculate_active_time_density(in_times, window_start, window_end, bin_size_ms=0.5)
    out_active_s = calculate_active_time_density(out_times, window_start, window_end, bin_size_ms=0.5)
    
    # 计算传输速度
    in_active_MBps = toMB(binb) / in_active_s if in_active_s > 0 else 0.0
    out_active_MBps = toMB(boutb) / out_active_s if out_active_s > 0 else 0.0
    
    return {
        'span_s': span,
        'Bi': bi, 'Bo': bo, 'Ci': ci, 'Co': co,
        'bytes_in': binb, 'bytes_out': boutb,
        'in_active_s': in_active_s,
        'out_active_s': out_active_s,
        'in_active_MBps': in_active_MBps,
        'out_active_MBps': out_active_MBps,
        'MBps_in': toMB(binb) / span if span > 0 else 0.0,
        'MBps_out': toMB(boutb) / span if span > 0 else 0.0
    }

def main():
    parser = argparse.ArgumentParser(description='修正的 per-invoke usbmon 统计')
    parser.add_argument('usbmon_file', help='usbmon.txt 文件路径')
    parser.add_argument('invokes_file', help='invokes.json 文件路径')
    parser.add_argument('time_map_file', help='time_map.json 文件路径')
    parser.add_argument('--extra', type=float, default=0.0, help='窗口扩展时间（秒）')
    parser.add_argument('--mode', default='bulk_complete', help='统计模式')
    
    args = parser.parse_args()
    
    # 读取 invoke 时间
    with open(args.invokes_file, 'r') as f:
        invokes_data = json.load(f)
    
    # 读取时间映射
    try:
        with open(args.time_map_file, 'r') as f:
            time_map = json.load(f)
        usbmon_ref = time_map['usbmon_ref']
        boottime_ref = time_map['boottime_ref']
        time_map_ready = True
    except (FileNotFoundError, KeyError):
        time_map_ready = False
        print("时间映射不可用，假设 EPOCH 时间对齐")
    
    spans = invokes_data.get('spans', [])
    print(f"总 invoke 数: {len(spans)}")
    
    # 统计每个 invoke
    per_invoke_stats = []
    warm_stats = []
    
    for i, span in enumerate(spans):
        # 转换时间
        if time_map_ready:
            b0 = span['begin'] - boottime_ref + usbmon_ref
            e0 = span['end'] - boottime_ref + usbmon_ref
        else:
            b0 = span['begin']
            e0 = span['end']
        
        # 统计这个窗口
        stats = stat_window(args.usbmon_file, b0, e0, args.mode, args.extra)
        per_invoke_stats.append(stats)
        
        # 跳过第一次（cold）
        if i > 0:
            warm_stats.append(stats)
    
    # 计算 warm 平均值
    if warm_stats:
        n = len(warm_stats)
        
        # 累计统计
        total_bytes_in = sum(s['bytes_in'] for s in warm_stats)
        total_bytes_out = sum(s['bytes_out'] for s in warm_stats)
        total_in_active_s = sum(s['in_active_s'] for s in warm_stats)
        total_out_active_s = sum(s['out_active_s'] for s in warm_stats)
        
        # 计算平均值
        avg_bytes_in = total_bytes_in / n
        avg_bytes_out = total_bytes_out / n
        avg_in_active_s = total_in_active_s / n
        avg_out_active_s = total_out_active_s / n
        
        # 计算平均传输速度
        avg_in_MBps = (total_bytes_in / (1000*1000)) / total_in_active_s if total_in_active_s > 0 else 0
        avg_out_MBps = (total_bytes_out / (1000*1000)) / total_out_active_s if total_out_active_s > 0 else 0
        
        # 统计非零次数
        nonzero_in = sum(1 for s in warm_stats if s['bytes_in'] > 0)
        nonzero_out = sum(1 for s in warm_stats if s['bytes_out'] > 0)
        
        print(f"\n=== Warm 摘要 (跳过第1次) ===")
        print(f"Warm invoke 数: {n}")
        print(f"IN:  总字节={total_bytes_in:,}, 平均={avg_bytes_in:.0f}, 非零次数={nonzero_in}")
        print(f"OUT: 总字节={total_bytes_out:,}, 平均={avg_bytes_out:.0f}, 非零次数={nonzero_out}")
        print(f"平均传输: IN={avg_bytes_in/(1000*1000):.3f}MB, OUT={avg_bytes_out/(1000*1000):.3f}MB")
        print(f"平均活跃时长: IN={avg_in_active_s:.4f}s, OUT={avg_out_active_s:.4f}s")
        print(f"平均活跃速度: IN={avg_in_MBps:.1f}MB/s, OUT={avg_out_MBps:.1f}MB/s")

if __name__ == '__main__':
    main()
