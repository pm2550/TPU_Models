#!/usr/bin/env python3
"""
显示IN/OUT重叠的具体位置
"""

import json
import sys
import argparse
import re
from collections import defaultdict

def parse_usbmon_records(usbmon_file):
    """解析 usbmon.txt 行，返回事件记录用于 URB S/C 配对。
    字段：urb_id, ts, ev(S/C), dir(Bi/Bo/Ci/Co), dev, ep, len
    """
    import re
    recs = []
    with open(usbmon_file, 'r', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            # 兼容两种列位：TIMESTAMP URB_ID ... 或 URB_ID TIMESTAMP ...
            ts = None; urb_id=None; ev=None
            try:
                # URB_ID TIMESTAMP ...
                urb_id = parts[0]
                ts = float(parts[1])
                ev = parts[2]
            except Exception:
                try:
                    ts = float(parts[0])
                    urb_id = parts[1]
                    ev = parts[2] if len(parts)>2 else 'S'
                except Exception:
                    continue
            if ts is not None and ts > 1e6:
                ts = ts/1e6
            # 查找方向/设备/端点
            direction = None; dev_num=None; ep_num=None; dir_idx=None
            for i, token in enumerate(parts):
                if re.match(r'^[BC][io]:\d+:\d+:\d+', token):
                    direction = token[:2]
                    try:
                        m = re.match(r'^[BC][io]:(\d+):(\d+):(\d+)', token)
                        if m:
                            dev_num = int(m.group(2)); ep_num = int(m.group(3))
                    except Exception:
                        pass
                    dir_idx = i
                    break
            if not direction:
                continue
            # 字节数：优先 len=，否则取方向字段后两列
            nb = 0
            mlen = re.search(r'len=(\d+)', line)
            if mlen:
                try:
                    nb = int(mlen.group(1))
                except Exception:
                    nb = 0
            elif dir_idx is not None and len(parts) > dir_idx + 2:
                try:
                    nb = int(re.sub(r'[^\d]', '', parts[dir_idx+2]) or '0')
                except Exception:
                    nb = 0
            recs.append({'urb_id': urb_id, 'ts': ts, 'ev': ev, 'dir': direction,
                         'dev': dev_num, 'ep': ep_num, 'len': nb})
    return recs

def build_urb_pairs(records):
    """按 urb_id 将 S/C 事件配对，返回包含提交/完成时间与方向等信息的列表。"""
    submit = {}
    pairs = []
    for r in records:
        if r['ev'] == 'S':
            submit[r['urb_id']] = r
        elif r['ev'] == 'C':
            s = submit.pop(r['urb_id'], None)
            if not s:
                continue
            d = s['dir']
            if d not in ('Bi','Bo'):
                continue
            dev = s['dev'] if s['dev'] is not None else r['dev']
            ep = s['ep'] if s['ep'] is not None else r['ep']
            length = s['len'] if s['len'] is not None else 0
            t0 = s['ts']; t1 = r['ts']
            if t1 < t0:
                continue
            dur = t1 - t0
            # 过滤异常小长包
            if length < 64 and dur > 0.002:
                continue
            pairs.append({'dir': d, 'dev': dev, 'ep': ep, 'len': length,
                          'ts_submit': t0, 'ts_complete': t1, 'duration': dur})
    return pairs

def union_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s,e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s,e))
    return merged

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

def find_active_intervals_from_pairs(pairs, window_start, window_end, direction, *, allow_dev=None, allow_ep=None):
    """基于 URB S/C 配对，在窗口内找指定方向的活跃区间（并集）。
    包含规则：任意交叠（overlap），并将区间剪裁到窗口边界，避免因跨窗URB导致0活跃。
    """
    ints = []
    for p in pairs:
        if p['dir'] != direction:
            continue
        if allow_dev is not None and p['dev'] is not None and p['dev'] != allow_dev:
            continue
        if allow_ep is not None and p['ep'] is not None and p['ep'] != allow_ep:
            continue
        t0 = p['ts_submit']; t1 = p['ts_complete']
        # 任意交叠
        if t1 >= window_start and t0 <= window_end:
            s = max(t0, window_start)
            e = min(t1, window_end)
            if e > s:
                ints.append((s, e))
    return union_intervals(ints)

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
    
    # 解析并配对 URB（更精确的活跃区间）
    print("解析USB事件并配对URB...")
    recs = parse_usbmon_records(usbmon_file)
    pairs = build_urb_pairs(recs)
    # 自动探测设备与端点
    auto_dev = args.dev
    ep_in = None; ep_out = None
    if auto_dev is None:
        bytes_by_dev = {}
        for p in pairs:
            if p['dir'] in ('Bi','Bo') and p['dev'] is not None:
                bytes_by_dev[p['dev']] = bytes_by_dev.get(p['dev'],0) + (p['len'] or 0)
        if bytes_by_dev:
            auto_dev = max(bytes_by_dev.items(), key=lambda x:x[1])[0]
    # 端点自动选择
    bytes_by_ep_in = {}; bytes_by_ep_out = {}
    for p in pairs:
        if p['dev'] != auto_dev:
            continue
        if p['dir']=='Bi' and p['ep'] is not None:
            bytes_by_ep_in[p['ep']] = bytes_by_ep_in.get(p['ep'],0)+(p['len'] or 0)
        if p['dir']=='Bo' and p['ep'] is not None:
            bytes_by_ep_out[p['ep']] = bytes_by_ep_out.get(p['ep'],0)+(p['len'] or 0)
    if bytes_by_ep_in:
        ep_in = max(bytes_by_ep_in.items(), key=lambda x:x[1])[0]
    if bytes_by_ep_out:
        ep_out = max(bytes_by_ep_out.items(), key=lambda x:x[1])[0]
    print(f"使用过滤: DEV={auto_dev}, EP_IN={ep_in}, EP_OUT={ep_out}")
    
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
        
        # 找到IN和OUT活跃区间（基于 URB 对）
        in_intervals = find_active_intervals_from_pairs(pairs, window_start, window_end, 'Bi', allow_dev=auto_dev, allow_ep=ep_in)
        out_intervals = find_active_intervals_from_pairs(pairs, window_start, window_end, 'Bo', allow_dev=auto_dev, allow_ep=ep_out)
        
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
