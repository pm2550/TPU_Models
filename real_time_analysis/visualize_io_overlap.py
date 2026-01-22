#!/usr/bin/env python3
import json
import sys
from analyze_usbmon_active import calculate_active_spans_urbs

def visualize_overlap(usbmon_file, invokes_file, time_map_file, invoke_index=0):
    """展示指定invoke的IN/OUT活跃时间重叠情况"""
    
    # 加载数据
    with open(time_map_file, 'r') as f:
        time_map = json.load(f)
    
    with open(invokes_file, 'r') as f:
        invokes_data = json.load(f)
    
    # 构建窗口
    windows = []
    if 'spans' in invokes_data:
        # 新格式：直接包含时间戳
        for span in invokes_data['spans']:
            windows.append({'begin': span['begin'], 'end': span['end']})
    else:
        # 旧格式：需要通过time_map转换
        for invoke in invokes_data:
            begin_ts = time_map.get(str(invoke['begin_id']), 0.0)
            end_ts = time_map.get(str(invoke['end_id']), 0.0)
            windows.append({'begin': begin_ts, 'end': end_ts})
    
    if invoke_index >= len(windows):
        print(f"错误: invoke_index {invoke_index} 超出范围 (共 {len(windows)} 个invoke)")
        return
    
    # 分析指定的invoke
    target_window = [windows[invoke_index]]
    print(f"调试: 分析窗口 {invoke_index}: [{target_window[0]['begin']:.6f}, {target_window[0]['end']:.6f}]")
    
    results = calculate_active_spans_urbs(usbmon_file, target_window, time_map, expand_s=0.0)
    
    if not results:
        print(f"错误: 无法分析 invoke {invoke_index}")
        return
    
    result = results[0]
    
    print(f"=== DenseNet201 seg1 - Invoke {invoke_index} IN/OUT 重叠分析 ===")
    print(f"Invoke 总时长: {result['invoke_span_s']*1000:.2f}ms")
    print(f"IN  活跃时长: {result['in_active_span_s']*1000:.2f}ms")
    print(f"OUT 活跃时长: {result['out_active_span_s']*1000:.2f}ms") 
    print(f"联合活跃时长: {result['union_active_span_s']*1000:.2f}ms")
    print()
    
    # 检查是否有区间数据
    if 'in_intervals_ms' in result and 'out_intervals_ms' in result:
        in_intervals = result['in_intervals_ms']
        out_intervals = result['out_intervals_ms']
        
        print(f"IN 区间数量: {len(in_intervals)}")
        print(f"OUT 区间数量: {len(out_intervals)}")
        print()
        
        if in_intervals:
            print("前5个 IN 区间 (相对invoke开始时间, ms):")
            for i, (start, end) in enumerate(in_intervals[:5]):
                print(f"  IN-{i+1}: [{start:.2f}, {end:.2f}] 长度={end-start:.2f}ms")
            print()
        
        if out_intervals:
            print("前5个 OUT 区间 (相对invoke开始时间, ms):")
            for i, (start, end) in enumerate(out_intervals[:5]):
                print(f"  OUT-{i+1}: [{start:.2f}, {end:.2f}] 长度={end-start:.2f}ms")
            print()
        
        # 分析重叠情况
        overlap_time = 0.0
        overlaps = []
        
        for in_start, in_end in in_intervals:
            for out_start, out_end in out_intervals:
                # 计算重叠区间
                overlap_start = max(in_start, out_start)
                overlap_end = min(in_end, out_end)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlap_time += overlap_duration
                    overlaps.append((overlap_start, overlap_end, overlap_duration))
        
        print(f"重叠分析:")
        print(f"  重叠区间数量: {len(overlaps)}")
        print(f"  总重叠时间: {overlap_time:.2f}ms")
        
        if overlaps:
            print("  前3个重叠区间:")
            for i, (start, end, duration) in enumerate(overlaps[:3]):
                print(f"    重叠-{i+1}: [{start:.2f}, {end:.2f}] 长度={duration:.2f}ms")
        
        # 计算理论上的简单相加时间
        simple_sum = result['in_active_span_s']*1000 + result['out_active_span_s']*1000
        actual_union = result['union_active_span_s']*1000
        saved_time = simple_sum - actual_union
        
        print()
        print(f"时间对比:")
        print(f"  IN + OUT 简单相加: {simple_sum:.2f}ms")
        print(f"  实际联合时间: {actual_union:.2f}ms")
        if simple_sum > 0:
            print(f"  重叠节省时间: {saved_time:.2f}ms ({saved_time/simple_sum*100:.1f}%)")
        else:
            print(f"  重叠节省时间: {saved_time:.2f}ms (N/A%)")
        
    else:
        print("⚠️  当前版本的分析结果不包含详细区间信息")
        print("   无法展示具体的重叠详情")
    
    print()
    print(f"数据传输:")
    print(f"  IN:  {result['bytes_in']:,} 字节 ({result['bytes_in']/1024:.1f} KB)")
    print(f"  OUT: {result['bytes_out']:,} 字节 ({result['bytes_out']/1024:.1f} KB)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python visualize_io_overlap.py <usbmon.txt> <invokes.json> <time_map.json>")
        sys.exit(1)
    
    usbmon_file = sys.argv[1]
    invokes_file = sys.argv[2]  
    time_map_file = sys.argv[3]
    
    visualize_overlap(usbmon_file, invokes_file, time_map_file, invoke_index=0)
