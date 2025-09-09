#!/usr/bin/env python3
"""
显示IN/OUT重叠的具体位置
"""

import json
import sys
import argparse
import re
from collections import defaultdict

def parse_usbmon_events(usbmon_file, dev_filter=None):
    """解析usbmon文件，返回所有USB事件的详细信息"""
    
    events = []
    with open(usbmon_file, 'r', errors='ignore') as f:  # 忽略二进制字符
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
                
            # 兼容两种常见格式："TIMESTAMP URB_ID ..." 和 "URB_ID TIMESTAMP ..."
            ts = None
            try:
                ts = float(parts[0])
            except Exception:
                try:
                    ts = float(parts[1])
                except Exception:
                    ts = None
            if ts is not None and ts > 1e6:
                ts = ts/1e6
            if ts is None:
                continue
                
            # 查找方向和设备号
            direction = None
            dev_num = None
            dir_idx = None
            for token in parts:
                if re.match(r'^[BC][io]:\d+:\d+:\d+', token):
                    direction = token[:2]
                    try:
                        maddr = re.match(r'^[BC][io]:(\d+):(\d+):(\d+)', token)
                        if maddr:
                            dev_num = int(maddr.group(2))
                    except:
                        pass
                    # 记录方向字段位置，便于后备解析字节数
                    try:
                        dir_idx = parts.index(token)
                    except ValueError:
                        dir_idx = None
                    break
                    
            if not direction or direction not in ['Bi', 'Bo']:
                continue
                
            # 设备过滤
            if dev_filter and dev_num != dev_filter:
                continue
                
            # 查找字节数：优先len=，否则从方向字段后两列取数字（与 usbmon 固定列对齐）
            bytes_count = 0
            for token in parts:
                if 'len=' in token:
                    try:
                        bytes_count = int(token.split('=')[1])
                        break
                    except:
                        pass

            # 如果没有len=，尝试基于方向字段位置的后备列解析
            if bytes_count == 0 and dir_idx is not None and len(parts) > dir_idx + 2:
                try:
                    clean_token = re.sub(r'[^\d]', '', parts[dir_idx + 2])
                    if clean_token:
                        bytes_count = int(clean_token)
                except:
                    pass
                        
            if bytes_count > 0:
                events.append({
                    'timestamp': ts,
                    'direction': direction,
                    'bytes': bytes_count,
                    'dev': dev_num
                })
    
    return events

def find_active_intervals(events, window_start, window_end, direction):
    """找到指定方向在窗口内的活跃区间（使用原始方法：首次到末次传输的总跨度）"""
    
    # 筛选窗口内的指定方向事件
    window_events = [e for e in events 
                    if window_start <= e['timestamp'] <= window_end 
                    and e['direction'] == direction]
    
    if not window_events:
        return []
    
    # 按时间排序
    window_events.sort(key=lambda x: x['timestamp'])
    
    # 使用原始方法：活跃时间 = 第一个到最后一个传输的总跨度
    first_time = window_events[0]['timestamp']
    last_time = window_events[-1]['timestamp']
    
    # 返回单个连续区间（从首次到末次传输）
    return [(first_time, last_time)]

def find_overlaps(in_intervals, out_intervals):
    """找到IN和OUT区间的重叠部分"""
    overlaps = []
    
    for i, (in_start, in_end) in enumerate(in_intervals):
        for j, (out_start, out_end) in enumerate(out_intervals):
            overlap_start = max(in_start, out_start)
            overlap_end = min(in_end, out_end)
            
            if overlap_start < overlap_end:
                overlaps.append({
                    'in_interval': i + 1,
                    'out_interval': j + 1,
                    'start': overlap_start,
                    'end': overlap_end,
                    'duration': overlap_end - overlap_start
                })
    
    return overlaps

def main():
    parser = argparse.ArgumentParser(description="显示IN/OUT重叠的具体位置")
    parser.add_argument('usbmon_file', help='usbmon.txt 文件路径')
    parser.add_argument('invokes_file', help='invokes.json 文件路径')
    parser.add_argument('time_map_file', help='time_map.json 文件路径')
    parser.add_argument('--start', type=int, default=2, help='起始invoke编号(从1开始，默认2以跳过cold run)')
    parser.add_argument('--end', type=int, default=None, help='结束invoke编号(包含，默认到末尾或起始后3个)')
    parser.add_argument('--dev', type=int, default=None, help='仅统计指定设备号(DEV)，默认自动探测')
    args = parser.parse_args()
    
    usbmon_file = args.usbmon_file
    invokes_file = args.invokes_file
    time_map_file = args.time_map_file
    
    # 读取文件
    with open(invokes_file) as f:
        invokes_data = json.load(f)
    
    with open(time_map_file) as f:
        time_map = json.load(f)
    
    # 处理invokes格式
    if 'spans' in invokes_data:
        spans = invokes_data['spans']
    else:
        spans = invokes_data
    
    # 解析所有USB事件
    print("解析USB事件...")
    # 自动探测设备号：统计 Bi/Bo 的字节量占比最高的设备
    auto_dev = args.dev
    if auto_dev is None:
        tmp_events = parse_usbmon_events(usbmon_file, dev_filter=None)
        bytes_by_dev = {}
        for e in tmp_events:
            if e['dev'] is not None:
                bytes_by_dev[e['dev']] = bytes_by_dev.get(e['dev'], 0) + e['bytes']
        if bytes_by_dev:
            auto_dev = max(bytes_by_dev.items(), key=lambda x: x[1])[0]
    all_events = parse_usbmon_events(usbmon_file, dev_filter=auto_dev)
    print(f"找到 {len(all_events)} 个有效USB事件 (DEV={auto_dev})")
    print(f"找到 {len(all_events)} 个有效USB事件")
    
    # 计算要分析的范围（将用户输入范围裁剪到有效边界内）
    total_invokes = len(spans)
    start_idx_1based = max(1, args.start)
    if args.end is None:
        # 默认展示起点开始的4个（与旧行为兼容）
        end_idx_1based = min(total_invokes, start_idx_1based + 3)
    else:
        end_idx_1based = min(total_invokes, max(args.start, args.end))
    
    # 将1-based转换为内部索引（0-based）
    start_idx = start_idx_1based - 1
    end_idx = end_idx_1based - 1
    
    for i in range(start_idx, end_idx + 1):
        span = spans[i]
        # 将boottime转换为usbmon时间
        boottime_start = span['begin']
        boottime_end = span['end']
        
        # 转换公式: usbmon_time = (boottime - boottime_ref) + usbmon_ref
        # 时间对齐：支持绝对时间戳
        if time_map.get('usbmon_ref') is not None:
            # 有usbmon_ref：转换到usbmon时间轴
            window_start = (boottime_start - time_map['boottime_ref']) + time_map['usbmon_ref']
            window_end = (boottime_end - time_map['boottime_ref']) + time_map['usbmon_ref']
        else:
            # 无usbmon_ref：直接使用绝对时间戳
            window_start = boottime_start
            window_end = boottime_end
        window_duration = (window_end - window_start) * 1000  # ms
        
        print(f"\n=== Invoke #{i+1} ===")
        print(f"窗口: {window_duration:.2f}ms")
        
        # 找到IN和OUT活跃区间
        in_intervals = find_active_intervals(all_events, window_start, window_end, 'Bi')
        out_intervals = find_active_intervals(all_events, window_start, window_end, 'Bo')
        
        # 转换为相对时间(ms)
        in_intervals_ms = [((start - window_start) * 1000, (end - window_start) * 1000) 
                          for start, end in in_intervals]
        out_intervals_ms = [((start - window_start) * 1000, (end - window_start) * 1000) 
                           for start, end in out_intervals]
        
        print(f"IN区间 ({len(in_intervals_ms)}个):")
        for j, (start, end) in enumerate(in_intervals_ms):
            print(f"  IN{j+1}: [{start:.2f}ms, {end:.2f}ms] 长度={end-start:.2f}ms")
        
        print(f"OUT区间 ({len(out_intervals_ms)}个):")
        for j, (start, end) in enumerate(out_intervals_ms):
            print(f"  OUT{j+1}: [{start:.2f}ms, {end:.2f}ms] 长度={end-start:.2f}ms")
        
        # 找重叠
        overlaps = find_overlaps(in_intervals, out_intervals)
        
        # 重叠统计与摘要
        total_overlap = 0.0
        if overlaps:
            print(f"重叠区间 ({len(overlaps)}个):")
            for overlap in overlaps:
                start_ms = (overlap['start'] - window_start) * 1000
                end_ms = (overlap['end'] - window_start) * 1000
                duration_ms = overlap['duration'] * 1000
                total_overlap += duration_ms
                print(f"  IN{overlap['in_interval']} ∩ OUT{overlap['out_interval']}: "
                      f"[{start_ms:.2f}ms, {end_ms:.2f}ms] 重叠={duration_ms:.2f}ms")
        else:
            print("重叠区间: 无")

        # 计算并报告IN/OUT总时长、总重叠与联合活跃
        in_total = sum(end - start for start, end in in_intervals_ms)
        out_total = sum(end - start for start, end in out_intervals_ms)
        union_time = in_total + out_total - total_overlap
        gap_time = window_duration - union_time
        print(f"总重叠: {total_overlap:.2f}ms")
        print(f"IN总时长: {in_total:.2f}ms, OUT总时长: {out_total:.2f}ms")
        print(f"活跃联合: {union_time:.2f}ms")
        print(f"剩余空白: {gap_time:.2f}ms")

if __name__ == '__main__':
    main()
