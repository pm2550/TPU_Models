#!/usr/bin/env python3
"""
分析 usbmon 逐次 invoke 数据，使用 URB 区间与 invoke 窗口交集的方法
"""
import json
import sys
import argparse
from pathlib import Path


def parse_urbs(usbmon_txt):
    """解析 usbmon.txt，返回 URB 列表"""
    urbs = []
    with open(usbmon_txt, 'r', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            
            # usbmon 格式: <urb_id> <timestamp> <event_type> <device:endpoint> ...
            try:
                timestamp = float(parts[1]) / 1e6  # 转换为秒
                event_type = parts[2]
                
                # 查找字节数
                nbytes = 0
                for i, part in enumerate(parts):
                    if part.startswith('len='):
                        nbytes = int(part[4:])
                        break
                    if i > 4 and part.isdigit():
                        nbytes = int(part)
                        break
                
                # 只关心 Bulk 传输的完成事件
                if event_type.startswith('C') and ('Bo:' in line or 'Bi:' in line):
                    direction = 'out' if 'Bo:' in line else 'in'
                    urbs.append({
                        'timestamp': timestamp,
                        'direction': direction,
                        'bytes': nbytes
                    })
            except (ValueError, IndexError):
                continue
    
    return urbs


def per_invoke_intersection(usbmon_txt, invokes_json, time_map_json=None, expand_s=0.0):
    """
    计算每次 invoke 的 IN/OUT 字节数和活跃时长
    expand_s: 扩展 invoke 窗口的秒数（前后各扩展）
    """
    # 解析 URB 数据
    urbs = parse_urbs(usbmon_txt)
    
    # 解析 invoke 时间窗口
    with open(invokes_json, 'r') as f:
        invokes_data = json.load(f)
    
    spans = invokes_data.get('spans', [])
    
    # 尝试加载时间映射
    usbmon_offset = 0.0
    if time_map_json and Path(time_map_json).exists():
        try:
            with open(time_map_json, 'r') as f:
                time_map = json.load(f)
            if 'usbmon_ref' in time_map and 'boottime_ref' in time_map:
                usbmon_offset = time_map['boottime_ref'] - time_map['usbmon_ref']
        except:
            pass
    
    # 检查 spans 是否为 EPOCH 时间（数值很大）
    if spans and spans[0]['begin'] > 1e9:
        # EPOCH 时间，需要转换为与 usbmon 相同的时间基准
        # 假设第一个 URB 时间对应第一个 invoke 开始前的某个时刻
        if urbs:
            first_urb_time = min(urb['timestamp'] for urb in urbs)
            first_invoke_time = spans[0]['begin']
            epoch_offset = first_invoke_time - first_urb_time
            print(f"检测到 EPOCH 时间，计算偏移: {epoch_offset:.6f}s", file=sys.stderr)
        else:
            epoch_offset = 0
    else:
        epoch_offset = 0
    
    results = []
    
    for i, span in enumerate(spans):
        # 如果是 EPOCH 时间，转换为 usbmon 时间基准
        if epoch_offset > 0:
            begin = span['begin'] - epoch_offset - expand_s
            end = span['end'] - epoch_offset + expand_s
        else:
            begin = span['begin'] + usbmon_offset - expand_s
            end = span['end'] + usbmon_offset + expand_s
        
        in_bytes = 0
        out_bytes = 0
        in_times = []
        out_times = []
        
        # 找到在此窗口内的 URB
        for urb in urbs:
            if begin <= urb['timestamp'] <= end:
                if urb['direction'] == 'in':
                    in_bytes += urb['bytes']
                    in_times.append(urb['timestamp'])
                else:
                    out_bytes += urb['bytes']
                    out_times.append(urb['timestamp'])
        
        # 计算活跃时长（第一个到最后一个传输的时间跨度）
        in_active_s = max(in_times) - min(in_times) if len(in_times) > 1 else 0.0
        out_active_s = max(out_times) - min(out_times) if len(out_times) > 1 else 0.0
        
        results.append({
            'invoke': i,
            'window': [begin, end],
            'in_bytes': in_bytes,
            'out_bytes': out_bytes,
            'in_active_s': in_active_s,
            'out_active_s': out_active_s,
            'in_count': len(in_times),
            'out_count': len(out_times)
        })
    
    return results


def print_summary(results):
    """打印摘要统计"""
    if not results:
        print("没有数据")
        return
    
    print(f"总 invoke 数: {len(results)}")
    
    # 跳过第一次（冷启动）
    warm_results = results[1:] if len(results) > 1 else results
    
    if not warm_results:
        print("没有 warm 数据")
        return
    
    total_in = sum(r['in_bytes'] for r in warm_results)
    total_out = sum(r['out_bytes'] for r in warm_results)
    total_in_active = sum(r['in_active_s'] for r in warm_results)
    total_out_active = sum(r['out_active_s'] for r in warm_results)
    
    non_zero_in = len([r for r in warm_results if r['in_bytes'] > 0])
    non_zero_out = len([r for r in warm_results if r['out_bytes'] > 0])
    
    print(f"\n=== Warm 摘要 (跳过第1次) ===")
    print(f"Warm invoke 数: {len(warm_results)}")
    print(f"IN:  总字节={total_in:,}, 平均={total_in/len(warm_results):.0f}, 非零次数={non_zero_in}")
    print(f"OUT: 总字节={total_out:,}, 平均={total_out/len(warm_results):.0f}, 非零次数={non_zero_out}")
    print(f"IN 活跃时长:  总计={total_in_active:.3f}s, 平均={total_in_active/len(warm_results):.4f}s")
    print(f"OUT 活跃时长: 总计={total_out_active:.3f}s, 平均={total_out_active/len(warm_results):.4f}s")


def main():
    parser = argparse.ArgumentParser(description='分析 usbmon 逐次 invoke 统计')
    parser.add_argument('usbmon_txt', help='usbmon.txt 文件路径')
    parser.add_argument('invokes_json', help='invokes.json 文件路径')
    parser.add_argument('--time_map', help='time_map.json 文件路径')
    parser.add_argument('--expand_s', type=float, default=2.0, help='扩展窗口秒数')
    parser.add_argument('--details', action='store_true', help='显示每次详情')
    
    args = parser.parse_args()
    
    results = per_invoke_intersection(
        args.usbmon_txt, 
        args.invokes_json, 
        args.time_map, 
        args.expand_s
    )
    
    if args.details:
        print("=== 逐次详情 ===")
        for r in results:
            print(f"invoke {r['invoke']}: IN={r['in_bytes']:,}B OUT={r['out_bytes']:,}B "
                  f"in_active={r['in_active_s']:.4f}s out_active={r['out_active_s']:.4f}s")
        print()
    
    print_summary(results)


if __name__ == '__main__':
    main()