#!/usr/bin/env python3
"""
使用原来正确的方法分析 usbmon 数据
基于 run_usbmon_capture_offline.sh 中的分析逻辑
"""
import json
import sys
import re
import os
from typing import List, Dict, Optional, Tuple

def parse_usbmon_simple(cap_file: str) -> List[Tuple[float, str, int, int, int, int]]:
    """解析 usbmon 文件，返回 (ts, direction, bytes, bus, dev, ep) 列表"""
    records: List[Tuple[float, str, int, int, int, int]] = []
    
    with open(cap_file, 'r', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
                
            # 解析时间戳
            try:
                ts = float(parts[1])
                ts = ts / 1e6 if ts > 1e6 else ts  # 转换微秒到秒
            except:
                continue
                
            # 查找方向token (Bi:, Bo:, Ci:, Co:)
            token = None
            token_idx = None
            bus = dev = ep = -1
            for i, part in enumerate(parts):
                m = re.match(r'^([BC][io]):(\d+):(\d+):(\d+)', part)
                if m:
                    token = m.group(1)  # Bi, Bo, Ci, Co
                    token_idx = i
                    try:
                        bus = int(m.group(2)); dev = int(m.group(3)); ep = int(m.group(4))
                    except Exception:
                        bus = dev = ep = -1
                    break
                    
            if not token:
                continue
                
            # 解析字节数
            bytes_count = 0
            
            # 方法1: 查找 len=(\d+)
            len_match = re.search(r'len=(\d+)', line)
            if len_match:
                bytes_count = int(len_match.group(1))
            # 方法2: token后第3个字段
            elif token_idx is not None and token_idx + 2 < len(parts):
                try:
                    bytes_count = int(parts[token_idx + 2])
                except:
                    bytes_count = 0
                    
            records.append((ts, token, bytes_count, bus, dev, ep))
            
    return records

def analyze_window(records: List[Tuple[float, str, int, int, int, int]], window: Dict, time_map: Dict, target_dev: Optional[int], expand_s: float = 0.0) -> Dict:
    """分析单个时间窗口的USB流量"""
    
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    
    if usb_ref is None or bt_ref is None:
        return {
            'note': 'no_ref',
            'span_s': window['end'] - window['begin'],
            'Bi': 0, 'Bo': 0, 'Ci': 0, 'Co': 0,
            'bytes_in': 0, 'bytes_out': 0,
            'MBps_in': 0.0, 'MBps_out': 0.0
        }
    
    # 转换时间窗口到usbmon时间戳
    b0 = window['begin'] - bt_ref + usb_ref - expand_s
    e0 = window['end'] - bt_ref + usb_ref + expand_s
    
    # 统计计数器
    bi = bo = ci = co = 0
    bytes_in = bytes_out = 0
    
    # 遍历记录
    for ts, direction, nb, _bus, _dev, _ep in records:
        if b0 <= ts <= e0:
            if target_dev is not None and _dev != target_dev:
                continue
            if direction == 'Bi':
                bi += 1
                # 仅在没有 Ci 的日志时计入 Bi
                if co == 0 and ci == 0:
                    bytes_in += nb
            elif direction == 'Bo':
                bo += 1
                if co == 0 and ci == 0:
                    bytes_out += nb
            elif direction == 'Ci':
                ci += 1
                bytes_in += nb
            elif direction == 'Co':
                co += 1
                bytes_out += nb
    
    span = e0 - b0
    to_MB = lambda x: x / (1024 * 1024.0)
    
    return {
        'span_s': span,
        'Bi': bi, 'Bo': bo, 'Ci': ci, 'Co': co,
        'bytes_in': bytes_in,
        'bytes_out': bytes_out,
        'MBps_in': (to_MB(bytes_in) / span) if span > 0 else 0.0,
        'MBps_out': (to_MB(bytes_out) / span) if span > 0 else 0.0
    }

def aggregate_stats(stats_list: List[Dict]) -> Dict:
    """聚合多个窗口的统计数据"""
    total = {'bytes_in': 0, 'bytes_out': 0, 'span_s': 0.0}
    
    for stat in stats_list:
        total['bytes_in'] += stat['bytes_in']
        total['bytes_out'] += stat['bytes_out']
        total['span_s'] += stat['span_s']
    
    to_MB = lambda x: x / (1024 * 1024.0)
    total['MBps_in'] = (to_MB(total['bytes_in']) / total['span_s']) if total['span_s'] > 0 else 0.0
    total['MBps_out'] = (to_MB(total['bytes_out']) / total['span_s']) if total['span_s'] > 0 else 0.0
    
    return total

def detect_target_dev(records: List[Tuple[float, str, int, int, int, int]], first_window: Dict, time_map: Dict) -> Optional[int]:
    """在首个窗口内按 dev 聚合字节数，选出字节最多的设备作为 Edge TPU 设备。"""
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        return None
    b0 = first_window['begin'] - bt_ref + usb_ref
    e0 = first_window['end'] - bt_ref + usb_ref
    dev_to_bytes: Dict[int, int] = {}
    for ts, d, nb, _bus, _dev, _ep in records:
        if b0 <= ts <= e0:
            dev_to_bytes[_dev] = dev_to_bytes.get(_dev, 0) + nb
    if not dev_to_bytes:
        return None
    # 选择 bytes 最大的 dev
    return max(dev_to_bytes.items(), key=lambda x: x[1])[0]

def main():
    if len(sys.argv) != 4:
        print("用法: python analyze_usbmon_simple.py <usbmon.txt> <invokes.json> <time_map.json>")
        sys.exit(1)
    
    cap_file = sys.argv[1]
    invokes_file = sys.argv[2]
    time_map_file = sys.argv[3]
    
    # 加载数据
    with open(time_map_file, 'r') as f:
        time_map = json.load(f) if os.path.exists(time_map_file) else {'usbmon_ref': None, 'boottime_ref': None}
    
    with open(invokes_file, 'r') as f:
        invokes_data = json.load(f)
    
    # 提取窗口
    if 'spans' in invokes_data:
        windows = invokes_data['spans']
    else:
        windows = invokes_data  # 假设是窗口列表
    
    # 解析usbmon记录
    print(f"解析 usbmon 文件: {cap_file}", file=sys.stderr)
    records = parse_usbmon_simple(cap_file)
    print(f"找到 {len(records)} 条记录", file=sys.stderr)
    
    # 设备过滤：优先环境变量 USBMON_DEV，其次自动检测首窗口内字节最多的设备
    target_dev: Optional[int] = None
    env_dev = os.environ.get('USBMON_DEV')
    if env_dev is not None and str(env_dev).strip() != '':
        try:
            target_dev = int(str(env_dev).strip())
            print(f"使用 USBMON_DEV={target_dev} 进行过滤", file=sys.stderr)
        except Exception:
            target_dev = None
    if target_dev is None and windows:
        target_dev = detect_target_dev(records, windows[0], time_map)
        if target_dev is not None:
            print(f"自动选择 dev={target_dev} 作为目标设备", file=sys.stderr)
        else:
            print("未能确定目标设备，将不做设备过滤", file=sys.stderr)
    
    # 分析每个窗口
    strict_stats = []
    loose_stats = []
    
    for i, window in enumerate(windows):
        strict = analyze_window(records, window, time_map, target_dev, expand_s=0.0)
        loose = analyze_window(records, window, time_map, target_dev, expand_s=0.010)
        
        strict_stats.append(strict)
        loose_stats.append(loose)
        
        if i < 3:  # 打印前3个窗口的详细信息
            print(f"窗口 {i}: 严格 IN={strict['bytes_in']}B OUT={strict['bytes_out']}B", file=sys.stderr)
    
    # 聚合结果
    strict_overall = aggregate_stats(strict_stats)
    loose_overall = aggregate_stats(loose_stats)
    
    # 输出结果
    result = {
        'method': 'simple_usbmon_analysis',
        'total_windows': len(windows),
        'strict_first': strict_stats[0] if strict_stats else {},
        'loose_first': loose_stats[0] if loose_stats else {},
        'strict_overall': strict_overall,
        'loose_overall': loose_overall,
        'per_invoke_strict': strict_stats,
        'per_invoke_loose': loose_stats
    }
    
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
