#!/usr/bin/env python3
"""
快速检查IN/OUT重叠情况
"""

import json
import sys
import re
from collections import defaultdict

def parse_usbmon_simple(usbmon_file, invoke_windows):
    """简单解析usbmon，找到IN/OUT事件"""
    
    # 解析所有USB事件
    usb_events = []
    with open(usbmon_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
                
            try:
                ts = float(parts[1])
                ts = ts / 1e6 if ts > 1e6 else ts
            except:
                continue
                
            # 查找方向
            direction = None
            for token in parts:
                if re.match(r'^[BC][io]:\d+:\d+:\d+', token):
                    direction = token[:2]
                    break
                    
            if direction in ['Bi', 'Bo']:  # 只看Bulk传输
                # 查找字节数
                bytes_count = 0
                for token in parts:
                    if 'len=' in token:
                        try:
                            bytes_count = int(token.split('=')[1])
                            break
                        except:
                            pass
                            
                if bytes_count > 0:
                    usb_events.append((ts, direction, bytes_count))
    
    print(f"总USB事件数: {len(usb_events)}")
    
    # 为每个invoke窗口分析IN/OUT分布
    results = []
    for i, (start_time, end_time) in enumerate(invoke_windows):
        if i == 0:  # 跳过cold run
            continue
            
        # 找到此窗口内的事件
        window_events = [(ts, direction, bytes_count) for ts, direction, bytes_count in usb_events 
                        if start_time <= ts <= end_time]
        
        in_events = [(ts, bytes_count) for ts, direction, bytes_count in window_events if direction == 'Bi']
        out_events = [(ts, bytes_count) for ts, direction, bytes_count in window_events if direction == 'Bo']
        
        window_span = (end_time - start_time) * 1000  # ms
        
        results.append({
            'invoke': i + 1,
            'window_ms': window_span,
            'in_events': len(in_events),
            'out_events': len(out_events),
            'in_times': [(ts - start_time) * 1000 for ts, _ in in_events[:10]],  # 相对时间ms
            'out_times': [(ts - start_time) * 1000 for ts, _ in out_events[:10]],
            'in_bytes_total': sum(b for _, b in in_events),
            'out_bytes_total': sum(b for _, b in out_events)
        })
        
        if len(results) >= 4:  # 只看前4次
            break
    
    return results

def main():
    if len(sys.argv) != 4:
        print("用法: python3 quick_overlap_check.py <usbmon.txt> <invokes.json> <time_map.json>")
        sys.exit(1)
        
    usbmon_file = sys.argv[1]
    invokes_file = sys.argv[2]
    time_map_file = sys.argv[3]
    
    # 读取invoke窗口
    with open(invokes_file) as f:
        invokes_data = json.load(f)
    
    with open(time_map_file) as f:
        time_map = json.load(f)
    
    # 处理invokes格式
    if 'spans' in invokes_data:
        spans = invokes_data['spans']
    else:
        spans = invokes_data
    
    # 转换为绝对时间
    invoke_windows = []
    for span in spans:
        abs_start = span['begin'] + time_map['base_time']
        abs_end = span['end'] + time_map['base_time']
        invoke_windows.append((abs_start, abs_end))
    
    print(f"找到 {len(invoke_windows)} 个invoke窗口")
    
    # 分析
    results = parse_usbmon_simple(usbmon_file, invoke_windows)
    
    # 输出结果
    for r in results:
        print(f"\n=== Invoke #{r['invoke']} ===")
        print(f"窗口长度: {r['window_ms']:.2f}ms")
        print(f"IN事件: {r['in_events']}个, 总字节: {r['in_bytes_total']}")
        print(f"OUT事件: {r['out_events']}个, 总字节: {r['out_bytes_total']}")
        
        print("IN事件时间点(ms):", [f"{t:.2f}" for t in r['in_times']])
        print("OUT事件时间点(ms):", [f"{t:.2f}" for t in r['out_times']])
        
        # 简单重叠检查：看时间点是否交替出现
        all_events = []
        for t in r['in_times']:
            all_events.append((t, 'IN'))
        for t in r['out_times']:
            all_events.append((t, 'OUT'))
        all_events.sort()
        
        print("时间序列:", ' -> '.join([f"{t:.1f}({typ})" for t, typ in all_events[:15]]))
        
        # 检查是否有重叠模式
        has_overlap = False
        for i in range(len(all_events) - 1):
            if all_events[i][1] != all_events[i+1][1]:  # 不同类型相邻
                if abs(all_events[i][0] - all_events[i+1][0]) < 1.0:  # 时间很近
                    has_overlap = True
                    break
        
        print(f"疑似重叠: {'是' if has_overlap else '否'}")

if __name__ == '__main__':
    main()
