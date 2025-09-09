#!/usr/bin/env python3

import json
import sys
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: script.py usbmon_file invokes_file time_map_file")
        sys.exit(1)
    
    usbmon_file, invokes_file, time_map_file = sys.argv[1:4]
    
    # 读取数据
    with open(invokes_file, 'r') as f:
        invokes_data = json.load(f)
    
    with open(time_map_file, 'r') as f:
        time_map = json.load(f)
    
    usbmon_ref = time_map['usbmon_ref']
    boottime_ref = time_map['boottime_ref']
    spans = invokes_data.get('spans', [])
    
    print(f"总 invoke 数: {len(spans)}")
    
    # 预解析 usbmon 数据
    print("解析 usbmon 数据...")
    usbmon_events = []
    
    with open(usbmon_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
                
            try:
                ts = int(parts[1]) / 1000000.0  # 转换为秒
                ev = parts[2]  # S 或 C
                endpoint_info = parts[3]  # 如 Bi:2:007:1
                
                if ':' in endpoint_info:
                    d = endpoint_info.split(':')[0]  # 提取 Bi, Bo, Ci, Co
                else:
                    continue
                
                if ev == 'C' and d in ('Bi', 'Bo'):
                    nb = int(parts[5])  # 字节数
                    usbmon_events.append((ts, d, nb))
            except:
                continue
    
    print(f"解析到 {len(usbmon_events)} 个相关事件")
    
    # 统计每个 invoke
    warm_stats = []
    
    for i, span in enumerate(spans):
        if i == 0:  # 跳过第一次（cold）
            continue
            
        # 转换时间
        b0 = span['begin'] - boottime_ref + usbmon_ref
        e0 = span['end'] - boottime_ref + usbmon_ref
        
        # 统计这个窗口内的事件
        bi_bytes = 0
        bo_bytes = 0
        bi_times = []
        bo_times = []
        
        for ts, d, nb in usbmon_events:
            if b0 <= ts <= e0:
                if d == 'Bi':
                    bi_bytes += nb
                    bi_times.append(ts)
                elif d == 'Bo':
                    bo_bytes += nb
                    bo_times.append(ts)
        
        # 计算活跃时间（使用时间密度，0.5ms 为一个时间段）
        bin_size = 0.0005  # 0.5ms
        
        def calc_active_time(times, window_start, window_end):
            if not times:
                return 0.0
            active_bins = set()
            for ts in times:
                if window_start <= ts <= window_end:
                    bin_idx = int((ts - window_start) / bin_size)
                    active_bins.add(bin_idx)
            return len(active_bins) * bin_size
        
        bi_active_s = calc_active_time(bi_times, b0, e0)
        bo_active_s = calc_active_time(bo_times, b0, e0)
        
        warm_stats.append({
            'bi_bytes': bi_bytes,
            'bo_bytes': bo_bytes,
            'bi_active_s': bi_active_s,
            'bo_active_s': bo_active_s
        })
    
    # 计算平均值
    if warm_stats:
        n = len(warm_stats)
        
        total_bi_bytes = sum(s['bi_bytes'] for s in warm_stats)
        total_bo_bytes = sum(s['bo_bytes'] for s in warm_stats)
        total_bi_active_s = sum(s['bi_active_s'] for s in warm_stats)
        total_bo_active_s = sum(s['bo_active_s'] for s in warm_stats)
        
        avg_bi_mb = (total_bi_bytes / n) / (1000 * 1000)
        avg_bo_mb = (total_bo_bytes / n) / (1000 * 1000)
        avg_bi_active_s = total_bi_active_s / n
        avg_bo_active_s = total_bo_active_s / n
        
        # 计算活跃速度
        avg_bi_speed = (total_bi_bytes / (1000*1000)) / total_bi_active_s if total_bi_active_s > 0 else 0
        avg_bo_speed = (total_bo_bytes / (1000*1000)) / total_bo_active_s if total_bo_active_s > 0 else 0
        
        nonzero_bi = sum(1 for s in warm_stats if s['bi_bytes'] > 0)
        nonzero_bo = sum(1 for s in warm_stats if s['bo_bytes'] > 0)
        
        print(f"\n=== Warm 摘要 (跳过第1次) ===")
        print(f"Warm invoke 数: {n}")
        print(f"IN:  总字节={total_bi_bytes:,}, 平均={total_bi_bytes//n}, 非零次数={nonzero_bi}")
        print(f"OUT: 总字节={total_bo_bytes:,}, 平均={total_bo_bytes//n}, 非零次数={nonzero_bo}")
        print(f"平均传输: IN={avg_bi_mb:.3f}MB, OUT={avg_bo_mb:.3f}MB")
        print(f"平均活跃时长: IN={avg_bi_active_s:.4f}s, OUT={avg_bo_active_s:.4f}s")
        print(f"平均活跃速度: IN={avg_bi_speed:.1f}MB/s, OUT={avg_bo_speed:.1f}MB/s")

if __name__ == '__main__':
    main()
