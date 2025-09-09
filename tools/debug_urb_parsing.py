#!/usr/bin/env python3
"""
调试 URB 解析，检查字节计算是否正确
"""
import json
import sys
import argparse
from pathlib import Path


def debug_parse_urbs(usbmon_txt, start_time, end_time, debug_lines=20):
    """调试解析 usbmon，显示详细信息"""
    print(f"调试时间窗口: {start_time:.6f} - {end_time:.6f} ({end_time-start_time:.6f}s)")
    
    in_urbs = []
    out_urbs = []
    total_lines = 0
    in_window_lines = 0
    
    with open(usbmon_txt, 'r', errors='ignore') as f:
        for line_num, line in enumerate(f):
            total_lines += 1
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            try:
                timestamp = float(parts[1]) / 1e6  # 转换为秒
                event_type = parts[2]
                
                # 检查是否在时间窗口内
                if start_time <= timestamp <= end_time:
                    in_window_lines += 1
                    
                    # 查找字节数
                    nbytes = 0
                    byte_source = "未找到"
                    
                    for i, part in enumerate(parts):
                        if part.startswith('len='):
                            nbytes = int(part[4:])
                            byte_source = f"len={part[4:]}"
                            break
                        if i > 4 and part.isdigit() and int(part) > 0:
                            nbytes = int(part)
                            byte_source = f"位置{i}={part}"
                            break
                    
                    # 只关心 Bulk 传输的完成事件
                    if event_type.startswith('C'):
                        if 'Bo:' in line:
                            out_urbs.append({
                                'timestamp': timestamp,
                                'bytes': nbytes,
                                'line': line.strip(),
                                'source': byte_source
                            })
                        elif 'Bi:' in line:
                            in_urbs.append({
                                'timestamp': timestamp,
                                'bytes': nbytes,
                                'line': line.strip(),
                                'source': byte_source
                            })
                
                # 显示前几条匹配的行
                if len(in_urbs) + len(out_urbs) <= debug_lines and in_window_lines <= debug_lines:
                    if start_time <= timestamp <= end_time and ('Bo:' in line or 'Bi:' in line):
                        direction = "OUT" if 'Bo:' in line else "IN"
                        print(f"  {direction}: {timestamp:.6f} {nbytes}B ({byte_source}) - {line.strip()}")
                        
            except (ValueError, IndexError):
                continue
    
    print(f"\n窗口统计:")
    print(f"  总行数: {total_lines}")
    print(f"  窗口内行数: {in_window_lines}")
    print(f"  IN URBs: {len(in_urbs)} 个, 总字节: {sum(u['bytes'] for u in in_urbs):,}")
    print(f"  OUT URBs: {len(out_urbs)} 个, 总字节: {sum(u['bytes'] for u in out_urbs):,}")
    
    # 检查异常大的传输
    large_in = [u for u in in_urbs if u['bytes'] > 10000]
    large_out = [u for u in out_urbs if u['bytes'] > 10000]
    
    if large_in:
        print(f"\n大的 IN 传输 (>10KB):")
        for u in large_in[:5]:
            print(f"  {u['timestamp']:.6f}: {u['bytes']:,}B ({u['source']}) - {u['line'][:80]}...")
    
    if large_out:
        print(f"\n大的 OUT 传输 (>10KB):")
        for u in large_out[:5]:
            print(f"  {u['timestamp']:.6f}: {u['bytes']:,}B ({u['source']}) - {u['line'][:80]}...")
    
    return sum(u['bytes'] for u in in_urbs), sum(u['bytes'] for u in out_urbs)


def main():
    parser = argparse.ArgumentParser(description='调试 URB 解析')
    parser.add_argument('usbmon_txt', help='usbmon.txt 文件路径')
    parser.add_argument('invokes_json', help='invokes.json 文件路径')
    parser.add_argument('--time_map', help='time_map.json 文件路径')
    parser.add_argument('--invoke_idx', type=int, default=1, help='调试第几个 invoke (0-based)')
    
    args = parser.parse_args()
    
    # 解析 invoke 时间窗口
    with open(args.invokes_json, 'r') as f:
        invokes_data = json.load(f)
    
    spans = invokes_data.get('spans', [])
    if args.invoke_idx >= len(spans):
        print(f"错误: invoke {args.invoke_idx} 不存在，总共只有 {len(spans)} 个")
        return
    
    span = spans[args.invoke_idx]
    
    # 尝试加载时间映射
    epoch_offset = 0
    if args.time_map and Path(args.time_map).exists():
        with open(args.time_map, 'r') as f:
            time_map = json.load(f)
        print(f"时间映射: {time_map}")
    
    # 检查是否为 EPOCH 时间
    if span['begin'] > 1e9:
        print("检测到 EPOCH 时间，需要计算偏移...")
        # 简单估算偏移
        usbmon_start = 1640000000  # 大概的 usbmon 起始时间（微秒）
        epoch_offset = span['begin'] - usbmon_start
        print(f"计算 EPOCH 偏移: {epoch_offset:.6f}s")
    
    begin = span['begin'] - epoch_offset
    end = span['end'] - epoch_offset
    
    print(f"调试 invoke {args.invoke_idx}:")
    print(f"  原始窗口: {span['begin']:.6f} - {span['end']:.6f}")
    print(f"  调整窗口: {begin:.6f} - {end:.6f}")
    
    in_bytes, out_bytes = debug_parse_urbs(args.usbmon_txt, begin, end)
    
    print(f"\n最终结果:")
    print(f"  IN: {in_bytes:,} bytes ({in_bytes/1024:.1f} KB)")
    print(f"  OUT: {out_bytes:,} bytes ({out_bytes/1024:.1f} KB)")


if __name__ == '__main__':
    main()

