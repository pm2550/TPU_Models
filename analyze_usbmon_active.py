#!/usr/bin/env python3
"""
分析USB监控数据，计算每次 invoke 的 IO 活跃时间，并以“IN ∪ OUT 并集”唯一口径扣除得到纯计算时间。

窗口定义与对齐：
- 默认使用“invoke 窗口”进行统计：仅基于 spans 中的 begin/end（忽略 set/get）。
- 可选时间轴平移：通过 SHIFT_POLICY 将 [begin,end] 整体平移以对齐 usbmon（例如 in_tail_or_out_head）。
- 窗口扩展：默认允许在 usbmon 轴上对窗口做前后扩展（ACTIVE_EXPAND_MS，默认 10ms）。
    若要求严格使用 invoke 窗口，请设置 STRICT_INVOKE_WINDOW=1（扩展为 0ms）。

时间度量：
- OUT（主机->设备）：使用 URB S→C 的并集时长（在窗口内裁剪）；字节数按重叠比例分摊。
- IN（设备->主机）：使用完成事件 C 的小间隙聚类并集时长（聚类间隙由 CLUSTER_GAP_MS 控制）。

纯计算时间（唯一口径）：
- 纯计算毫秒 pure_compute_ms = invoke_ms - io_union_both_ms，
    其中 io_union_both_ms 为“IN 聚簇区间 ∪ OUT URB 区间”的并集时长（去重）。
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
    """计算每次invoke的IO活跃时间（回退口径，仅用完成C行，避免S/C双计数）。

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
        
        # 收集窗口内的IO事件（仅统计完成C行：Ci/Co）
        in_events = []  # (timestamp, bytes) — 仅 Ci
        out_events = [] # 仅 Co

        for rec in usbmon_records:
            ts, direction, bytes_count = rec[0], rec[1], rec[2]
            if win_start <= ts <= win_end:
                if direction == 'Ci':  # 输入完成
                    in_events.append((ts, bytes_count))
                elif direction == 'Co':  # 输出完成
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
    """基于 URB 配对（S/C 配对）的方式计算活跃时间与字节数。

    关键改进：不再仅保留“完全落入窗口”的URB，否则会导致大量漏算，尤其是IN/OUT跨窗时。
    改为：对与窗口有重叠的URB进行裁剪，并按重叠时长占原URB时长的比例分摊字节数。
    这样既能稳定计算并集活跃时长（用裁剪后的区间做并集），也能更合理地估算窗口内的字节量，避免回退到逐行统计造成的双计数。
    """
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        return []

    # 预解析 IN/OUT URBs
    out_urbs = parse_urbs_from_file(usbmon_file, ('Bo', 'Co'), dev_filter)
    in_urbs = parse_urbs_from_file(usbmon_file, ('Bi', 'Ci'), dev_filter)

    def overlap_clip_intervals(
        b: float,
        e: float,
        urbs: List[Tuple[float, float, int, str, Optional[int]]]
    ) -> Tuple[List[Tuple[float, float]], float]:
        """返回：
        - 裁剪到窗口后的时间区间列表 [(cs, ce), ...]
        - 按重叠时长比例分摊后的字节数之和（float）

        同时过滤明显异常：极小字节但超长持续的URB（疑似控制/错误重试），避免污染统计。
        """
        intervals: List[Tuple[float, float]] = []
        bytes_sum: float = 0.0
        for (s, t, nb, dir_tok, devn) in urbs:
            # 无重叠
            if t <= b or s >= e:
                continue
            # 过滤异常URB：小于64B但持续>2ms
            if nb < 64 and (t - s) > 0.002:
                continue
            cs = b if s < b else s
            ce = e if t > e else t
            if ce <= cs:
                continue
            # 计算按时长占比的字节分摊
            dur = t - s
            if dur <= 0:
                # 无法按比例分摊，保守跳过
                continue
            frac = (ce - cs) / dur
            intervals.append((cs, ce))
            bytes_sum += float(nb) * max(0.0, min(1.0, frac))
        return intervals, bytes_sum

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

        # 对与窗口有重叠的URB进行裁剪，并按重叠比例估算窗口内字节
        in_intervals, in_bytes_win = overlap_clip_intervals(b0, e0, in_urbs)
        out_intervals, out_bytes_win = overlap_clip_intervals(b0, e0, out_urbs)
        all_intervals = in_intervals + out_intervals

        in_active = union_length(in_intervals)
        out_active = union_length(out_intervals)
        union_active = union_length(all_intervals)

        # 使用窗口裁剪后的按比例字节数作为窗口内字节量估计
        total_in_bytes = int(round(in_bytes_win))
        total_out_bytes = int(round(out_bytes_win))

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
            'in_events_count': len(in_intervals),
            'out_events_count': len(out_intervals),
            'in_intervals_ms': in_intervals_ms,
            'out_intervals_ms': out_intervals_ms,
        })

    return results


def build_c_event_times(usbmon_file: str) -> Tuple[List[float], List[float]]:
    """Parse usbmon.txt and return completion (C) timestamps for IN and OUT.
    Accept both Bi/Ci as IN, Bo/Co as OUT on C lines.
    """
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    cin: List[float] = []
    cout: List[float] = []
    try:
        with open(usbmon_file, 'r', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                sc = parts[2]
                if sc != 'C':
                    continue
                try:
                    ts = float(parts[1])
                    ts = ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
                m = re_dir.search(ln)
                if not m:
                    continue
                tok = m.group(1)
                if tok.startswith('Bi') or tok.startswith('Ci'):
                    cin.append(ts)
                elif tok.startswith('Bo') or tok.startswith('Co'):
                    cout.append(ts)
    except FileNotFoundError:
        pass
    cin.sort(); cout.sort()
    return cin, cout


def compute_shift_deltas_per_invoke(
    invokes: List[Dict],
    time_map: Dict,
    cin_times: List[float],
    cout_times: List[float],
    policy: str,
    tail_ms: float,
    head_ms: float,
    max_shift_ms: float,
) -> List[float]:
    """Compute per-invoke shift delta (seconds) per policy.

    Policies:
    - 'none': all zeros
    - 'in_tail': align so t1->last IN within (t1, t1+tail]
    - 'in_tail_or_out_head': prefer in_tail; if absent or too large, use out_head
    - 'out_head': align so t0->first OUT within [t0, t0+head]
    """
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        return [0.0] * len(invokes)

    tail_s = max(0.0, (tail_ms or 0.0) / 1000.0)
    head_s = max(0.0, (head_ms or 0.0) / 1000.0)
    max_s  = max(0.0, (max_shift_ms or 0.0) / 1000.0)

    deltas: List[float] = []
    for i, w in enumerate(invokes):
        t0 = usb_ref + (w['begin'] - bt_ref)
        t1 = usb_ref + (w['end'] - bt_ref)
        di = 0.0
        reason = 'none'

        def find_last_in_after_t1() -> Optional[float]:
            # binary search windows (simple linear scan is fine for small lists)
            lo = t1
            hi = t1 + tail_s
            # find last cin in (t1, hi]
            # use reversed traversal with early break
            for ts in reversed(cin_times):
                if ts <= lo:
                    break
                if ts <= hi:
                    return ts
            return None

        def find_first_out_after_t0() -> Optional[float]:
            lo = t0
            hi = t0 + head_s
            for ts in cout_times:
                if ts < lo:
                    continue
                if ts <= hi:
                    return ts
                break
            return None

        if policy == 'in_tail' or policy == 'in_tail_or_out_head':
            ts_in = find_last_in_after_t1()
            if ts_in is not None:
                di = ts_in - t1
                reason = 'in_tail'
        if (policy == 'out_head') or (policy == 'in_tail_or_out_head' and reason != 'in_tail'):
            ts_out = find_first_out_after_t0()
            if ts_out is not None:
                di = ts_out - t0
                reason = 'out_head'

        # clamp unreasonable shifts
        if di < 0.0 or di > max_s:
            di = 0.0
            reason = 'none'

        deltas.append(di)
    return deltas


def cluster_union_within_window(times: List[float], w_begin: float, w_end: float, gap_s: float, min_span_s: float) -> Tuple[float, int, List[Tuple[float, float]]]:
    """Cluster event times within [w_begin, w_end] and return (union_len_s, clusters_count, clusters).
    Each cluster span is max(last-first, min_span_s). Events must already be absolute timestamps.
    """
    ev = [ts for ts in times if w_begin <= ts <= w_end]
    if not ev:
        return 0.0, 0, []
    clusters: List[Tuple[float, float]] = []
    cs = ev[0]
    ce = ev[0]
    for ts in ev[1:]:
        if ts - ce <= gap_s:
            ce = ts
        else:
            clusters.append((cs, ce))
            cs = ce = ts
    clusters.append((cs, ce))
    total = 0.0
    for s, t in clusters:
        total += max(t - s, min_span_s)
    return total, len(clusters), clusters


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
        # 读取对齐策略参数
        shift_policy = os.environ.get('SHIFT_POLICY', 'none').strip()
        try:
            search_tail_ms = float(os.environ.get('SEARCH_TAIL_MS', '20'))
        except Exception:
            search_tail_ms = 20.0
        try:
            search_head_ms = float(os.environ.get('SEARCH_HEAD_MS', '10'))
        except Exception:
            search_head_ms = 10.0
        try:
            max_shift_ms = float(os.environ.get('MAX_SHIFT_MS', '30'))
        except Exception:
            max_shift_ms = 30.0
        try:
            cluster_gap_ms = float(os.environ.get('CLUSTER_GAP_MS', '0.1'))
        except Exception:
            cluster_gap_ms = 0.1
        try:
            min_cluster_us = float(os.environ.get('MIN_CLUSTER_US', '1'))
        except Exception:
            min_cluster_us = 1.0
        try:
            expand_ms_env = os.environ.get('ACTIVE_EXPAND_MS')
            expand_ms = float(expand_ms_env) if expand_ms_env else 10.0
        except Exception:
            expand_ms = 10.0
        # 若要求严格使用 invoke 窗口（忽略任何扩展），则强制将扩展设为 0
        strict_invoke_window = os.environ.get('STRICT_INVOKE_WINDOW', '0').strip().lower() in ('1','true','yes','on')
        if strict_invoke_window:
            expand_ms = 0.0
        # 构建 C 完成事件时间表用于对齐与聚类
        cin_times, cout_times = build_c_event_times(usbmon_file)

        # 计算每次 invoke 的平移量（秒）
        if shift_policy not in ('none', 'in_tail', 'out_head', 'in_tail_or_out_head'):
            shift_policy = 'none'
        shift_deltas = compute_shift_deltas_per_invoke(
            invoke_windows, time_map, cin_times, cout_times,
            shift_policy, search_tail_ms, search_head_ms, max_shift_ms
        )

        # 生成平移后的窗口
        invoke_windows_shifted = []
        for i, w in enumerate(invoke_windows):
            d = shift_deltas[i] if i < len(shift_deltas) else 0.0
            invoke_windows_shifted.append({'begin': w['begin'] + d, 'end': w['end'] + d})

    # 优先使用更稳健的 URB 并集时长计算；若无效或事件计数为0，则回退到行级事件 min-max 估计
        try:
            # 可选设备过滤：USBMON_DEV 设置时启用
            dev_filter = None
            try:
                dev_env = os.environ.get('USBMON_DEV')
                dev_filter = int(dev_env) if dev_env else None
            except Exception:
                dev_filter = None
            active_spans = calculate_active_spans_urbs(usbmon_file, invoke_windows_shifted, time_map, expand_s=expand_ms/1000.0, dev_filter=dev_filter)
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
                active_spans = calculate_active_spans(usbmon_records, invoke_windows_shifted, time_map, expand_s=expand_ms/1000.0)
        except Exception:
            active_spans = calculate_active_spans(usbmon_records, invoke_windows_shifted, time_map, expand_s=expand_ms/1000.0)

        # 追加基于 C 完成事件聚类的 IN/OUT 并集与纯推理（只扣 IN）以及平移/窗口信息
        usb_ref = time_map.get('usbmon_ref')
        bt_ref = time_map.get('boottime_ref')
        gap_s = max(0.0, cluster_gap_ms/1000.0)
        min_span_s = max(0.0, min_cluster_us/1e6)
        enriched = []
        
        def union_length_ms(intervals_ms):
            """计算以毫秒为单位的区间并集长度。intervals_ms: List[(start_ms,end_ms)]."""
            if not intervals_ms:
                return 0.0
            iv = sorted(intervals_ms)
            cs, ce = iv[0]
            total = 0.0
            for s, t in iv[1:]:
                if s <= ce:
                    if t > ce:
                        ce = t
                else:
                    total += (ce - cs)
                    cs, ce = s, t
            total += (ce - cs)
            return total

        for i, rec in enumerate(active_spans):
            w = invoke_windows[i]
            d = shift_deltas[i] if i < len(shift_deltas) else 0.0
            # shifted absolute window in usbmon time
            b = (w['begin'] + d) - bt_ref + usb_ref - (expand_ms/1000.0 if expand_ms else 0.0)
            e = (w['end']   + d) - bt_ref + usb_ref + (expand_ms/1000.0 if expand_ms else 0.0)
            in_u_s, in_k, in_clusters = cluster_union_within_window(cin_times, b, e, gap_s, min_span_s)
            out_u_s, out_k, out_clusters = cluster_union_within_window(cout_times, b, e, gap_s, min_span_s)
            inv_ms = (w['end'] - w['begin']) * 1000.0
            # OUT 的 URB 并集时长（毫秒）来自上游 URB 解析结果
            out_urb_ms = (rec.get('out_active_span_s') or 0.0) * 1000.0
            # IN聚簇区间（ms，相对窗口）
            in_cluster_intervals_ms = [((s - b) * 1000.0, (t - b) * 1000.0) for (s, t) in in_clusters]
            # OUT 使用 URB 区间（已相对窗口，毫秒）
            out_urb_intervals_ms = rec.get('out_intervals_ms') or []
            # 并集扣除口径（去重）
            io_union_both_ms = union_length_ms(in_cluster_intervals_ms + out_urb_intervals_ms)
            # 唯一口径：纯计算时间
            pure_compute_ms = inv_ms - io_union_both_ms
            rec['shift_policy'] = shift_policy
            rec['shift_ms'] = d * 1000.0
            rec['cluster_gap_ms'] = cluster_gap_ms
            rec['window_source'] = 'invoke'  # 明确标注：仅使用 invoke 窗口（忽略 set/get）
            rec['window_expand_ms'] = expand_ms
            rec['in_union_cluster_ms'] = in_u_s * 1000.0
            rec['out_union_urb_ms'] = out_urb_ms
            rec['io_union_both_ms'] = io_union_both_ms
            rec['pure_compute_ms'] = pure_compute_ms
            rec['in_clusters'] = in_k
            rec['out_clusters'] = out_k
            enriched.append(rec)
        active_spans = enriched
        
        # 输出结果（附带窗口定义元信息）
        result = {
            'model_name': invokes_data.get('name', 'unknown'),
            'total_invokes': len(active_spans),
            'per_invoke': active_spans,
            'window_meta': {
                'source': 'invoke',
                'strict_invoke_window': strict_invoke_window,
                'active_expand_ms': expand_ms,
                'shift_policy': shift_policy,
                'cluster_gap_ms': cluster_gap_ms
            }
        }
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
