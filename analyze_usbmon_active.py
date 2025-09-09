#!/usr/bin/env python3
"""
分析USB监控数据，计算每次invoke的纯IO活跃时间
基于usbmon数据计算实际的数据传输活跃时间，而不是invoke的总时间。

改进：支持窗口扩展（默认 ±10ms，可通过环境变量 ACTIVE_EXPAND_MS 调整），
用于涵盖 set/get 与 invoke 边界抖动，避免活跃IO被误判为 0。
"""

import json
import os
import sys
import re
from typing import List, Dict, Tuple, Any, Optional


def parse_usbmon_line(line: str) -> Tuple[float, str, int, Optional[int]]:
    """解析usbmon行，返回(时间戳, 方向, 字节数, 设备号dev)"""
    parts = line.split()
    if len(parts) < 3:
        return None, None, 0, None

    # 第二列时间戳，可能是微秒，需要规范化为秒
    try:
        ts = float(parts[1])
        ts = ts / 1e6 if ts > 1e6 else ts
    except Exception:
        return None, None, 0, None

    # 查找方向标记及其索引
    direction = None
    dir_idx = None
    dev_num: Optional[int] = None
    for i, token in enumerate(parts):
        if re.match(r'^[BC][io]:\d+:\d+:\d+', token):
            direction = token[:2]  # Bi/Bo/Ci/Co
            dir_idx = i
            try:
                maddr = re.match(r'^[BC][io]:(\d+):(\d+):(\d+)', token)
                if maddr:
                    dev_num = int(maddr.group(2))
            except Exception:
                dev_num = None
            break

    if not direction:
        return None, None, 0, None

    # 提取字节数：优先 len=，否则回退为方向标记后第2列（与现有脚本一致）
    bytes_count = 0
    m = re.search(r'len=(\d+)', line)
    if m:
        bytes_count = int(m.group(1))
    elif dir_idx is not None and (dir_idx + 2) < len(parts):
        try:
            bytes_count = int(parts[dir_idx + 2])
        except Exception:
            bytes_count = 0

    return ts, direction, bytes_count, dev_num


def calculate_active_spans(usbmon_records: List[Tuple], 
                          invoke_windows: List[Dict], 
                          time_map: Dict,
                          expand_s: float = 0.010) -> List[Dict]:
    """计算每次invoke的IO活跃时间。

    expand_s: 在转换到 usbmon 时间轴后，对每个窗口做前后扩展（单位：秒）。
    """
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    
    if usb_ref is None or bt_ref is None:
        return []
    
    results = []
    
    for i, window in enumerate(invoke_windows):
        # 将boottime窗口转换为usbmon时间，并做前后扩展
        win_start = window['begin'] - bt_ref + usb_ref - (expand_s or 0.0)
        win_end = window['end'] - bt_ref + usb_ref + (expand_s or 0.0)
        
        # 收集窗口内的IO事件
        in_events = []  # (timestamp, bytes)
        out_events = []
        
        for rec in usbmon_records:
            ts, direction, bytes_count = rec[0], rec[1], rec[2]
            if win_start <= ts <= win_end:
                if direction in ['Bi', 'Ci']:  # 输入事件
                    in_events.append((ts, bytes_count))
                elif direction in ['Bo', 'Co']:  # 输出事件
                    out_events.append((ts, bytes_count))
        
        # 计算活跃时间跨度
        in_active_span = 0.0
        out_active_span = 0.0
        total_in_bytes = sum(b for _, b in in_events)
        total_out_bytes = sum(b for _, b in out_events)
        
        if in_events:
            in_active_span = max(ts for ts, _ in in_events) - min(ts for ts, _ in in_events)
            if in_active_span == 0 and len(in_events) == 1:
                in_active_span = 0.001  # 最小1ms活跃时间
        
        if out_events:
            out_active_span = max(ts for ts, _ in out_events) - min(ts for ts, _ in out_events)
            if out_active_span == 0 and len(out_events) == 1:
                out_active_span = 0.001  # 最小1ms活跃时间
        
        # 计算并集活跃时间（IN和OUT的总体活跃时间）
        all_events = in_events + out_events
        union_active_span = 0.0
        if all_events:
            union_active_span = max(ts for ts, _ in all_events) - min(ts for ts, _ in all_events)
            if union_active_span == 0 and len(all_events) == 1:
                union_active_span = 0.001
        
        result = {
            'invoke_index': i,
            'invoke_span_s': window['end'] - window['begin'],
            'bytes_in': total_in_bytes,
            'bytes_out': total_out_bytes,
            'in_active_span_s': in_active_span,
            'out_active_span_s': out_active_span,
            'union_active_span_s': union_active_span,  # 纯IO活跃时间
            'in_events_count': len(in_events),
            'out_events_count': len(out_events)
        }
        results.append(result)
    
    return results


def parse_urbs_from_file(usbmon_file: str, dirs: Tuple[str, str], dev_filter: Optional[int]=None) -> List[Tuple[float, float, int, str, Optional[int]]]:
    """从 usbmon.txt 解析 URB 匹配（S/C 配对），返回 [(start_s, end_s, bytes, dirTok, devNum)]。

    dirs: (submit_dir, complete_dir)，例如 ('Bi','Ci') 或 ('Bo','Co')。
    """
    sub_dir, com_dir = dirs
    pending: Dict[str, Tuple[float, str, Optional[int]]] = {}
    finished: List[Tuple[float, float, int, str, Optional[int]]] = []
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    try:
        with open(usbmon_file, 'r', errors='ignore') as f:
            for ln in f:
                cols = ln.split()
                if len(cols) < 4:
                    continue
                tag = cols[0]
                try:
                    ts = float(cols[1])
                    ts = ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
                sc = cols[2]
                mdir = re_dir.search(ln)
                if not mdir:
                    continue
                dir_tok = mdir.group(1)
                try:
                    dev_num = int(mdir.group(3))
                except Exception:
                    dev_num = None
                if dev_filter is not None and dev_num is not None and dev_num != dev_filter:
                    continue
                if dir_tok not in (sub_dir, com_dir):
                    continue
                if sc == 'S':
                    pending[tag] = (ts, dir_tok, dev_num)
                elif sc == 'C':
                    start = None
                    if tag in pending:
                        s, d, dv = pending.pop(tag)
                        if d == sub_dir and dir_tok == com_dir:
                            start = s
                    nbytes = 0
                    m = re.search(r"len=(\d+)", ln)
                    if m:
                        nbytes = int(m.group(1))
                    else:
                        parts = ln.strip().split()
                        dir_idx = None
                        for i, tok in enumerate(parts):
                            if re.match(r'^[CB][io]:\d+:', tok):
                                dir_idx = i
                                break
                        if dir_idx is not None and dir_idx + 2 < len(parts):
                            try:
                                nbytes = int(parts[dir_idx + 2])
                            except Exception:
                                nbytes = 0
                        if nbytes == 0:
                            m2 = re.search(r"#\s*(\d+)", ln)
                            if m2:
                                nbytes = int(m2.group(1))
                    if start is not None:
                        finished.append((start, ts, nbytes, dir_tok, dev_num))
    except FileNotFoundError:
        pass
    return finished


def calculate_active_spans_urbs(usbmon_file: str,
                                invoke_windows: List[Dict],
                                time_map: Dict,
                                expand_s: float = 0.010,
                                dev_filter: Optional[int]=None) -> List[Dict]:
    """基于 URB 配对（S/C 配对）的方式计算活跃时间（更稳健），可按设备号过滤。"""
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        return []

    # 预解析 IN/OUT URBs
    out_urbs = parse_urbs_from_file(usbmon_file, ('Bo', 'Co'), dev_filter)
    in_urbs = parse_urbs_from_file(usbmon_file, ('Bi', 'Ci'), dev_filter)

    def filter_full_inside_and_sane(b: float, e: float, urbs: List[Tuple[float, float, int, str, Optional[int]]]):
        # 仅保留完全落在窗口内的 URB，且过滤“很小长度但很长持续”的异常 URB
        filtered = []
        for (s, t, nb, dir_tok, devn) in urbs:
            if s >= b and t <= e:
                if not (nb < 64 and (t - s) > 0.002):  # 小于64B且持续>2ms 的忽略
                    filtered.append((s, t, nb, dir_tok, devn))
        return filtered

    def window_intervals(b: float, e: float, urbs: List[Tuple[float, float, int, str, Optional[int]]]):
        sane = filter_full_inside_and_sane(b, e, urbs)
        win = [(s, t) for (s, t, _, _, _) in sane]
        return win

    def union_length(intervals: List[Tuple[float, float]]) -> float:
        if not intervals:
            return 0.0
        intervals.sort(key=lambda x: x[0])
        cs, ce = intervals[0]
        total = 0.0
        for s, t in intervals[1:]:
            if s <= ce:
                if t > ce:
                    ce = t
            else:
                total += (ce - cs)
                cs, ce = s, t
        total += (ce - cs)
        return total

    results: List[Dict[str, Any]] = []
    for i, window in enumerate(invoke_windows):
        b0 = window['begin'] - bt_ref + usb_ref - (expand_s or 0.0)
        e0 = window['end'] - bt_ref + usb_ref + (expand_s or 0.0)

        # 仅保留完全落入窗口内的 URB，并过滤小包长时异常
        in_iv = filter_full_inside_and_sane(b0, e0, in_urbs)
        out_iv = filter_full_inside_and_sane(b0, e0, out_urbs)

        in_intervals = window_intervals(b0, e0, in_iv)
        out_intervals = window_intervals(b0, e0, out_iv)
        all_intervals = in_intervals + out_intervals

        in_active = union_length(in_intervals)
        out_active = union_length(out_intervals)
        union_active = union_length(all_intervals)

        total_in_bytes = sum(nb for (_, _, nb, _, _) in in_iv)
        total_out_bytes = sum(nb for (_, _, nb, _, _) in out_iv)

        # 相对窗口的区间（毫秒）
        in_intervals_ms = [((s - b0) * 1000.0, (t - b0) * 1000.0) for (s, t) in in_intervals]
        out_intervals_ms = [((s - b0) * 1000.0, (t - b0) * 1000.0) for (s, t) in out_intervals]

        results.append({
            'invoke_index': i,
            'invoke_span_s': window['end'] - window['begin'],
            'bytes_in': total_in_bytes,
            'bytes_out': total_out_bytes,
            'in_active_span_s': in_active,
            'out_active_span_s': out_active,
            'union_active_span_s': union_active,
            'in_events_count': len(in_iv),
            'out_events_count': len(out_iv),
            'in_intervals_ms': in_intervals_ms,
            'out_intervals_ms': out_intervals_ms,
        })

    return results


def main():
    if len(sys.argv) != 4:
        print("用法: python analyze_usbmon_active.py <usbmon.txt> <invokes.json> <time_map.json>")
        sys.exit(1)
    
    usbmon_file = sys.argv[1]
    invokes_file = sys.argv[2]
    time_map_file = sys.argv[3]
    
    # 读取数据
    try:
        with open(invokes_file) as f:
            invokes_data = json.load(f)
        
        with open(time_map_file) as f:
            time_map = json.load(f)
        
        # 解析usbmon数据
        usbmon_records = []
        with open(usbmon_file, 'r', errors='ignore') as f:
            for line in f:
                ts, direction, bytes_count, dev_num = parse_usbmon_line(line)
                if ts is not None and direction is not None:
                    usbmon_records.append((ts, direction, bytes_count, dev_num))
        
        # 计算活跃时间（支持通过环境变量 ACTIVE_EXPAND_MS 设置扩展毫秒，默认 10ms）
        invoke_windows = invokes_data.get('spans', [])
        try:
            expand_ms_env = os.environ.get('ACTIVE_EXPAND_MS')
            expand_ms = float(expand_ms_env) if expand_ms_env else 10.0
        except Exception:
            expand_ms = 10.0
        # 优先使用更稳健的 URB 并集时长计算；若无效或事件计数为0，则回退到行级事件 min-max 估计
        try:
            # 可选设备过滤：USBMON_DEV 设置时启用
            dev_filter = None
            try:
                dev_env = os.environ.get('USBMON_DEV')
                dev_filter = int(dev_env) if dev_env else None
            except Exception:
                dev_filter = None
            active_spans = calculate_active_spans_urbs(usbmon_file, invoke_windows, time_map, expand_s=expand_ms/1000.0, dev_filter=dev_filter)
            def spans_have_meaningful_bytes(spans):
                warm = spans[1:] if len(spans) > 1 else spans
                total_bytes = sum((s.get('bytes_in', 0) or 0) + (s.get('bytes_out', 0) or 0) for s in warm)
                total_events = sum((s.get('in_events_count', 0) or 0) + (s.get('out_events_count', 0) or 0) for s in warm)
                # 要么有字节，要么至少有事件但字节不全为 0（兼容不同内核格式）
                return (total_bytes > 0) or (total_events > 0 and any(((s.get('bytes_in',0) or 0) > 0 or (s.get('bytes_out',0) or 0) > 0) for s in warm))
            # 若 URB 解析无有效字节（或无事件），回退到按行统计，与 io_split 行为对齐
            if not active_spans or not spans_have_meaningful_bytes(active_spans):
                if dev_filter is not None:
                    usbmon_records = [r for r in usbmon_records if r[3] == dev_filter]
                active_spans = calculate_active_spans(usbmon_records, invoke_windows, time_map, expand_s=expand_ms/1000.0)
        except Exception:
            active_spans = calculate_active_spans(usbmon_records, invoke_windows, time_map, expand_s=expand_ms/1000.0)
        
        # 输出结果
        result = {
            'model_name': invokes_data.get('name', 'unknown'),
            'total_invokes': len(active_spans),
            'per_invoke': active_spans
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
