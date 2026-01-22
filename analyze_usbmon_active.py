#!/usr/bin/env python3
"""
分析USB监控数据，计算每次 invoke 的 IO 活跃时间，并以“IN ∪ OUT 并集”口径扣除得到纯计算时间。

窗口定义与对齐：
- 默认严格使用“invoke 窗口”：仅基于 spans 的 begin/end（忽略 set/get），不做扩展；
    如需放宽，请显式设置 ACTIVE_EXPAND_MS>0。
- 可选时间轴平移：通过 SHIFT_POLICY 将 [begin,end] 做常量平移以对齐 usbmon（例如 in_tail_or_out_head）。

时间度量：
- OUT（主机->设备）：使用 URB S→C 的并集时长（在窗口内裁剪）；字节数按重叠比例分摊。
- IN（设备->主机）：混合口径——若该 URB 的提交 S 在窗口内，则用 URB S→C 区间（裁剪后）计入；
    对于窗口内的 C 而其对应 S 不在窗口内的，则用 C 完成时间按小间隙聚类（由 CLUSTER_GAP_MS 控制）。

纯计算时间：
- pure_invoke_ms =  Last C Bo和下一个 任意 Bi之间的距离
- pure_compute_ms = invoke_ms - io_span_sum_ms 

可选 off-chip 校正（实验特性）：
- 若设置了 OFFCHIP_OUT_THEORY_MIBPS（或兼容变量 OFFCHIP_OUT_MIBPS），
        则计算 Δt = bytes_out/当前OUT速率 − bytes_out/理论OUT速率，并把 max(0, Δt) 以毫秒加回 pure_compute_ms，得到 pure_compute_offchip_adj_ms。
        若未显式提供且开启 OFFCHIP_ENABLE，则默认理论速率为 320 MiB/s。
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
                        # Relaxed: if submit direction matches sub_dir, accept any corresponding C token
                        # Some usbmon variants keep Bi/Bo on C lines instead of Ci/Co
                        if d == sub_dir:
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
                                dev_filter: Optional[int]=None,
                                cin_times_for_cluster: Optional[List[float]] = None,
                                cluster_gap_s: float = 0.0001,
                                min_cluster_span_s: float = 0.000001) -> List[Dict]:
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

    # 事件时间序列（不绑定配对），用于 span 的“窗口内第一个 S 与最后一个 C”逻辑
    # 注：这里不做小包过滤，严格按照事件时间取首 S/尾 C。
    try:
        from bisect import bisect_left, bisect_right
    except Exception:
        bisect_left = None
        bisect_right = None
    sin_times, sout_times = build_submit_event_times(usbmon_file)
    cin_all, cout_all = build_c_event_times(usbmon_file)

    # small packet exclusion threshold (bytes). Default 64.
    try:
        min_urb_bytes = int(float(os.environ.get('MIN_URB_BYTES', '64')))
    except Exception:
        min_urb_bytes = 64

    # 为 span 端点选择准备：按字节阈值过滤后的“逐行事件”时间（不做URB配对）
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    sin_big: List[float] = []
    sout_big: List[float] = []
    cin_big: List[float] = []
    cout_big: List[float] = []
    try:
        with open(usbmon_file, 'r', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                sc = parts[2]
                try:
                    ts = float(parts[1])
                    ts = ts / 1e6 if ts > 1e6 else ts
                except Exception:
                    continue
                m = re_dir.search(ln)
                if not m:
                    continue
                tok = m.group(1)
                # 字节数解析（兼容 len= / # bytes）
                nbytes = 0
                dir_idx = None
                for i_tok, tok_str in enumerate(parts):
                    if re.match(r'^[CB][io]:\d+:', tok_str):
                        dir_idx = i_tok
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
                if (nbytes or 0) < min_urb_bytes:
                    continue
                if sc == 'S':
                    if tok.startswith('Bi'):
                        sin_big.append(ts)
                    elif tok.startswith('Bo'):
                        sout_big.append(ts)
                elif sc == 'C':
                    if tok.startswith('Bi') or tok.startswith('Ci'):
                        cin_big.append(ts)
                    elif tok.startswith('Bo') or tok.startswith('Co'):
                        cout_big.append(ts)
    except FileNotFoundError:
        pass
    sin_big.sort(); sout_big.sort(); cin_big.sort(); cout_big.sort()

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
            # Exclude small packets entirely from both interval contribution and bytes
            if (nb or 0) < min_urb_bytes:
                continue
            # 无重叠
            if t <= b or s >= e:
                continue
            # 注：即使字节未知（nb==0），也保留区间用于并集时长；仅在字节分摊时按0处理。
            # 若明确为超小字节且超长，抑制其字节贡献，但仍保留区间用于时长统计。
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
            # 字节统计：
            # - nb==0 视为未知：不增加字节，但保留区间
            # - nb<64 且 dur>2ms：可能为控制/异常，抑制字节但保留区间
            # - 其他：按比例分摊
            if nb > 0 and not (nb < 64 and dur > 0.002):
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

    # Completion times for IN clustering when S is not within window
    if cin_times_for_cluster is None:
        cin_times_for_cluster, _ = build_c_event_times(usbmon_file)

    results: List[Dict[str, Any]] = []
    for i, window in enumerate(invoke_windows):
        he_ms = float(window.get('head_expand_ms', 0.0) or 0.0)
        te_ms = float(window.get('tail_expand_ms', 0.0) or 0.0)
        he_s = he_ms / 1000.0
        te_s = te_ms / 1000.0
        b0 = window['begin'] - bt_ref + usb_ref - (expand_s or 0.0) - he_s
        e0 = window['end'] - bt_ref + usb_ref + (expand_s or 0.0) + te_s

        # OUT：始终使用 URB S->C 裁剪
        out_intervals, out_bytes_win = overlap_clip_intervals(b0, e0, out_urbs)

        # IN：混合口径
        # - 若该 URB 的 S 在窗口内，则使用 URB S->C 裁剪；
        # - 否则（S 不在窗口内），改用 C 聚簇（避免用跨窗的 URB 影响时长形态）。
        in_intervals: List[Tuple[float, float]] = []
        in_bytes_win_total: float = 0.0
        # 1) URB路径：仅保留 S 在窗口内的 URB，并裁剪
        urbs_s_in_window = [
            (s, t, nb, d, dv)
            for (s, t, nb, d, dv) in in_urbs
            if (s >= b0 and s <= e0 and t > b0 and s < e0)
        ]
        urbs_s_in_set_end = set(t for (s, t, nb, d, dv) in urbs_s_in_window)
        urbs_intervals, urbs_bytes = overlap_clip_intervals(b0, e0, urbs_s_in_window)
        in_intervals.extend(urbs_intervals)
        in_bytes_win_total += urbs_bytes

        # 2) C聚簇路径：从 C 完成时间中剔除已由 URB路径覆盖的那些（以 C 时间匹配 URB 的 t），再进行聚簇
        eps = 1e-9
        c_candidates: List[float] = []
        # 小包完成时间集合（用于聚簇时剔除）
        small_in_c_times = set(t for (s, t, nb, d, dv) in in_urbs if (nb or 0) < min_urb_bytes)
        for ts in cin_times_for_cluster:
            if ts < b0 or ts > e0:
                continue
            # Skip small-packet completions entirely
            if ts in small_in_c_times:
                continue
            # 如果该 C 对应的 URB 其 S 在窗口内，则跳过，由上面的 URB 区间承载
            skip = False
            # 以 end 时间匹配，考虑浮点误差
            if ts in urbs_s_in_set_end:
                skip = True
            else:
                for te in urbs_s_in_set_end:
                    if abs(te - ts) <= eps:
                        skip = True
                        break
            if not skip:
                c_candidates.append(ts)

        c_union_s, c_clusters_n, c_clusters = cluster_union_within_window(
            c_candidates, b0, e0, cluster_gap_s, min_cluster_span_s
        )
        # 将 C 聚簇转换为时间区间
        c_intervals = list(c_clusters)
        in_intervals.extend(c_intervals)

        # 按之前整体 URB 裁剪估计 IN 字节（保留此前口径）
        # 注：字节量估计与时长形态可能不完全一致，但“并集时长”符合本诉求
        _, in_bytes_full = overlap_clip_intervals(b0, e0, in_urbs)
        if in_bytes_full > in_bytes_win_total:
            in_bytes_win_total = in_bytes_full

        all_intervals = in_intervals + out_intervals

        in_active = union_length(in_intervals)
        out_active = union_length(out_intervals)
        union_active = union_length(all_intervals)

        # 使用窗口裁剪后的按比例字节数作为窗口内字节量估计
        total_in_bytes = int(round(in_bytes_win_total))
        total_out_bytes = int(round(out_bytes_win))

        # 相对窗口的区间（毫秒）
        in_intervals_ms = [((s - b0) * 1000.0, (t - b0) * 1000.0) for (s, t) in in_intervals]
        out_intervals_ms = [((s - b0) * 1000.0, (t - b0) * 1000.0) for (s, t) in out_intervals]
        # 确保即便字节为0也保留并输出 URB 区间，供并集使用

    # 方向内 span（首=窗口内第一个 Submit(S)，尾=窗口内最后一个 Complete(C)）
        def first_s_in_range(times: List[float], b: float, e: float) -> Optional[float]:
            if not times:
                return None
            if 'bisect_left' in globals() and bisect_left is not None:
                i = bisect_left(times, b)
                if i < len(times) and times[i] <= e:
                    return times[i]
                return None
            # 退化线性查找
            first = None
            for ts in times:
                if ts < b:
                    continue
                if ts > e:
                    break
                first = ts
                break
            return first

        def last_c_in_range(times: List[float], b: float, e: float) -> Optional[float]:
            if not times:
                return None
            if 'bisect_right' in globals() and bisect_right is not None:
                j = bisect_right(times, e) - 1
                if j >= 0 and times[j] >= b:
                    return times[j]
                return None
            # 退化线性查找（反向）
            last = None
            for ts in reversed(times):
                if ts > e:
                    continue
                if ts < b:
                    break
                last = ts
                break
            return last

        # 可选严格口径：要求“第一个 S 的 C 也在窗口内，最后一个 C 的 S 也在窗口内”（以 URB 配对判断），并应用小包过滤
        strict_pair = os.environ.get('SPAN_STRICT_PAIR', '0').strip().lower() in ('1','true','yes','on')

        if strict_pair:
            def first_s_with_paired_c_in(b: float, e: float, urbs_list):
                s_min = None
                for (s, t, nb, d, dv) in urbs_list:
                    if s is None or t is None:
                        continue
                    if (nb or 0) < min_urb_bytes:
                        continue
                    if b <= s <= e and b <= t <= e:
                        if s_min is None or s < s_min:
                            s_min = s
                return s_min
            def last_c_with_paired_s_in(b: float, e: float, urbs_list):
                t_max = None
                for (s, t, nb, d, dv) in urbs_list:
                    if s is None or t is None:
                        continue
                    if (nb or 0) < min_urb_bytes:
                        continue
                    if b <= s <= e and b <= t <= e:
                        if t_max is None or t > t_max:
                            t_max = t
                return t_max

            in_first_s = first_s_with_paired_c_in(b0, e0, in_urbs)
            in_last_c  = last_c_with_paired_s_in(b0, e0, in_urbs)
            out_first_s = first_s_with_paired_c_in(b0, e0, out_urbs)
            out_last_c  = last_c_with_paired_s_in(b0, e0, out_urbs)
        else:
            # IN/OUT 分别取：窗口内第一个大包 S 与最后一个大包 C（逐行事件，无配对约束）
            in_first_s = first_s_in_range(sin_big, b0, e0)
            in_last_c  = last_c_in_range(cin_big, b0, e0)
            out_first_s = first_s_in_range(sout_big, b0, e0)
            out_last_c  = last_c_in_range(cout_big, b0, e0)

        in_span_sc_ms = ((in_last_c - in_first_s) * 1000.0) if (in_first_s is not None and in_last_c is not None and (in_last_c - in_first_s) >= 0.0) else 0.0
        out_span_sc_ms = ((out_last_c - out_first_s) * 1000.0) if (out_first_s is not None and out_last_c is not None and (out_last_c - out_first_s) >= 0.0) else 0.0

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
                'in_span_sc_ms': in_span_sc_ms,
                'out_span_sc_ms': out_span_sc_ms,
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


def build_submit_event_times(usbmon_file: str) -> Tuple[List[float], List[float]]:
    """Parse usbmon.txt and return submit (S) timestamps for IN and OUT.
    Accept both Bi/Bo on S lines.
    """
    re_dir = re.compile(r"([CB][io]):(\d+):(\d+):(\d+)")
    sin: List[float] = []
    sout: List[float] = []
    try:
        with open(usbmon_file, 'r', errors='ignore') as f:
            for ln in f:
                parts = ln.split()
                if len(parts) < 3:
                    continue
                sc = parts[2]
                if sc != 'S':
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
                if tok.startswith('Bi'):
                    sin.append(ts)
                elif tok.startswith('Bo'):
                    sout.append(ts)
    except FileNotFoundError:
        pass
    sin.sort(); sout.sort()
    return sin, sout


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


def compute_alignment_tail_last_in_guard_bos(
    invokes: List[Dict],
    time_map: Dict,
    cin_times: List[float],
    bos_times: List[float],
    tail_ms: float,
    head_ms: float,
    max_shift_ms: float,
    extra_head_expand_ms: float,
) -> List[Dict[str, Any]]:
    """Compute per-invoke alignment plan:
    1) Align window tail (t1) to the last IN completion within [t1 - head_ms, t1 + tail_ms].
       - If the last IN lies before t1 (already inside window), do not move (no negative shift).
    2) Guard to include the first Bo submit (S) near head:
       - If BoS lies before shifted start by <= extra_head_expand_ms, expand head by that amount (increase window length).
       - Else reduce shift (shrink tail alignment) just enough to include BoS; head expand remains capped at extra.

    Returns a list of dict per invoke: {
      'shift_s': float, 'head_expand_s': float, 'tail_shrink_s': float,
      'last_in_ts': float|None, 'bos_ts': float|None,
      'reason': str
    }
    """
    usb_ref = time_map.get('usbmon_ref')
    bt_ref = time_map.get('boottime_ref')
    if usb_ref is None or bt_ref is None:
        return [{'shift_s': 0.0, 'head_expand_s': 0.0, 'tail_shrink_s': 0.0, 'last_in_ts': None, 'bos_ts': None, 'reason': 'no_time_map'} for _ in invokes]

    tail_s = max(0.0, (tail_ms or 0.0) / 1000.0)
    head_s = max(0.0, (head_ms or 0.0) / 1000.0)
    max_s  = max(0.0, (max_shift_ms or 0.0) / 1000.0)
    extra_s = max(0.0, (extra_head_expand_ms or 0.0) / 1000.0)

    plans: List[Dict[str, Any]] = []
    for i, w in enumerate(invokes):
        t0 = usb_ref + (w['begin'] - bt_ref)
        t1 = usb_ref + (w['end'] - bt_ref)

        # 1) Tail align to last IN within [t0 - head_s, t1 + tail_s]
        last_in = last_within(cin_times, t0 - head_s, t1 + tail_s)
        # Align tail to that IN: allow negative or positive shift
        shift = (last_in - t1) if (last_in is not None) else 0.0
        # clamp to allowed range [-max_s, max_s]
        if shift < -max_s or shift > max_s:
            # out of bound: do not align to this IN
            shift = 0.0
            last_in = None

        reason = 'tail_last_in' if last_in is not None else 'no_tail_in'
        head_expand = 0.0
        tail_shrink = 0.0

        # 2) Find first BoS near head: prefer first after t0, else last before t0 within head_s
        bos = None
        # First BoS after t0
        for ts in bos_times:
            if ts >= t0:
                if ts <= t0 + head_s:
                    bos = ts
                break
        if bos is None:
            # last before t0 within head window
            for ts in reversed(bos_times):
                if ts < t0 - head_s:
                    break
                if ts < t0:
                    bos = ts
                    break

        if bos is not None:
            start_shifted = t0 + shift
            if bos < start_shifted:
                need = start_shifted - bos  # how much earlier the head needs to be
                if need <= extra_s:
                    head_expand = need
                    reason += '+head_expand'
                else:
                    # reduce shift (move start earlier) to include BoS; cap head_expand to extra_s
                    reduce = need - extra_s
                    shift = shift - reduce
                    # clamp again to [-max_s, max_s]
                    if shift < -max_s:
                        shift = -max_s
                    if shift >  max_s:
                        shift =  max_s
                    # tail_shrink is how much we backed off from tail alignment
                    if last_in is not None:
                        ideal = (last_in - t1)
                        tail_shrink = max(0.0, ideal - shift)
                    reason += '+guard_bos_reduce_shift'

        plans.append({
            'shift_s': shift,
            'head_expand_s': head_expand,
            'tail_shrink_s': tail_shrink,
            'last_in_ts': last_in,
            'bos_ts': bos,
            'reason': reason,
        })
    return plans


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

def first_after(times: List[float], t: float) -> Optional[float]:
    """Return first event time >= t from a sorted list, else None."""
    for ts in times:
        if ts >= t:
            return ts
    return None

def last_within(times: List[float], t0: float, t1: float) -> Optional[float]:
    """Return last event time in (t0, t1], else None. times must be sorted."""
    last = None
    for ts in times:
        if ts <= t0:
            continue
        if ts <= t1:
            last = ts
        else:
            break
    return last


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
        
    # 计算活跃时间（支持通过环境变量 ACTIVE_EXPAND_MS 设置扩展毫秒，默认严格=0ms）
        invoke_windows = invokes_data.get('spans', [])
        
        # 按 USB_DEVICE 过滤 invoke_windows；不同测试映射不同，需由外部传入 usb:0/usb:1
        usb_device_filter = os.environ.get('USB_DEVICE')  # 例："usb:0" 或 "usb:1"
        if usb_device_filter:
            invoke_windows = [w for w in invoke_windows if w.get('device') == usb_device_filter]
            print(f"[INFO] Filtered invoke_windows by USB_DEVICE={usb_device_filter}: {len(invoke_windows)} spans", file=sys.stderr)
            # 针对该设备重新计算时间基准（该设备的首个大包 Submit）并平移窗口
            try:
                first_big_submit = None
                with open(usbmon_file, 'r', errors='ignore') as f:
                    for line in f:
                        # 形如 "S Bo:2:003:... bytes"，筛选到具体设备
                        if usb_device_filter == 'usb:0':
                            dev_num = '003'
                        elif usb_device_filter == 'usb:1':
                            dev_num = '004'
                        else:
                            dev_num = None
                        if dev_num is None:
                            break
                        if f"Bo:2:{dev_num}:" in line:
                            parts = line.strip().split()
                            if len(parts) >= 6 and parts[1] == 'S':
                                try:
                                    bytes_count = int(parts[5])
                                    if bytes_count > 10000:  # 只取较大的数据提交
                                        first_big_submit = float(parts[0]) / 1_000_000.0
                                        break
                                except Exception:
                                    pass
                if first_big_submit is not None:
                    # 原 time_map 基准
                    base_usb_ref = time_map.get('usbmon_ref')
                    delta = first_big_submit - base_usb_ref
                    # 平移窗口以对齐该设备的首个大包
                    for w in invoke_windows:
                        if 'begin' in w:
                            w['begin'] += delta
                        if 'end' in w:
                            w['end'] += delta
                    print(f"[INFO] Shifted invoke windows for {usb_device_filter} by {delta*1000:.2f} ms", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] Failed to rebase time_map for {usb_device_filter}: {e}", file=sys.stderr)
        # 检测single口径：若span包含 set_begin/get_end 字段，则视为single-like，
        # 在后续采用“last Co(Bo) -> next Bi(S/C)”的纯invoke定义
        try:
            single_like = any((('set_begin' in (w or {})) or ('get_end' in (w or {}))) for w in invoke_windows)
        except Exception:
            single_like = False
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
        # 默认严格窗口：不扩展；用户可显式设置 ACTIVE_EXPAND_MS>0 放宽
        try:
            expand_ms_env = os.environ.get('ACTIVE_EXPAND_MS')
            expand_ms = float(expand_ms_env) if expand_ms_env is not None else 0.0
        except Exception:
            expand_ms = 0.0
        # 严格标志默认开启；允许通过 STRICT_INVOKE_WINDOW=0 关闭
        strict_invoke_window = os.environ.get('STRICT_INVOKE_WINDOW', '1').strip().lower() in ('1','true','yes','on')
        if strict_invoke_window:
            expand_ms = 0.0
    # 构建 S/C 事件时间表用于对齐与聚类
        cin_times, cout_times = build_c_event_times(usbmon_file)
        sin_times, sout_times = build_submit_event_times(usbmon_file)

        # 计算每次 invoke 的平移量（秒）
        extra_head_expand_ms = 0.0
        try:
            extra_head_expand_ms = float(os.environ.get('EXTRA_HEAD_EXPAND_MS', '1'))
        except Exception:
            extra_head_expand_ms = 1.0

        if shift_policy not in ('none', 'in_tail', 'out_head', 'in_tail_or_out_head', 'tail_last_BiC_guard_BoS'):
            shift_policy = 'none'

        if shift_policy == 'tail_last_BiC_guard_BoS':
            # 按字节阈值筛选用于对齐的事件：
            # - 头部 Bo S：仅考虑 Bo/Co URB 中字节数 > 1KiB 的提交时间 S
            # - 尾部 Bi C：仅考虑 Bi/Ci URB 中字节数 > 512B 的完成时间 C
            try:
                head_bos_min_bytes = int(float(os.environ.get('HEAD_BOS_MIN_BYTES', '1024')))
            except Exception:
                head_bos_min_bytes = 1024
            try:
                tail_bi_min_bytes = int(float(os.environ.get('TAIL_BI_MIN_BYTES', '512')))
            except Exception:
                tail_bi_min_bytes = 512

            # 可选设备过滤（若对齐也需要限定到某个设备）
            dev_filter_align = None
            try:
                dev_env = os.environ.get('USBMON_DEV')
                dev_filter_align = int(dev_env) if dev_env else None
            except Exception:
                dev_filter_align = None

            out_urbs_align = parse_urbs_from_file(usbmon_file, ('Bo', 'Co'), dev_filter_align)
            in_urbs_align = parse_urbs_from_file(usbmon_file, ('Bi', 'Ci'), dev_filter_align)
            # 过滤并提取时间轴
            bos_times_filt = sorted([s for (s, t, nb, d, dv) in out_urbs_align if (nb or 0) > head_bos_min_bytes])
            cin_times_filt = sorted([t for (s, t, nb, d, dv) in in_urbs_align if (nb or 0) > tail_bi_min_bytes])

            plans = compute_alignment_tail_last_in_guard_bos(
                invoke_windows, time_map, cin_times_filt, bos_times_filt,
                search_tail_ms, search_head_ms, max_shift_ms, extra_head_expand_ms
            )
            shift_deltas = [p['shift_s'] for p in plans]
            head_expands = [p['head_expand_s'] for p in plans]
            tail_shrinks = [p['tail_shrink_s'] for p in plans]
        else:
            plans = None
            shift_deltas = compute_shift_deltas_per_invoke(
                invoke_windows, time_map, cin_times, cout_times,
                shift_policy, search_tail_ms, search_head_ms, max_shift_ms
            )
            head_expands = [0.0] * len(shift_deltas)
            tail_shrinks = [0.0] * len(shift_deltas)

        # 生成平移后的窗口
        invoke_windows_shifted = []
        for i, w in enumerate(invoke_windows):
            d = shift_deltas[i] if i < len(shift_deltas) else 0.0
            he = head_expands[i] if i < len(head_expands) else 0.0
            # Tail expansion is not used in this policy; we record tail_shrink diagnostically
            invoke_windows_shifted.append({
                'begin': w['begin'] + d,
                'end': w['end'] + d,
                'head_expand_ms': he * 1000.0,  # store ms for downstream
                'tail_expand_ms': 0.0,
            })

    # 优先使用更稳健的 URB 并集时长计算；若无效或事件计数为0，则回退到行级事件 min-max 估计
        try:
            # 可选设备过滤：USBMON_DEV 设置时启用
            dev_filter = None
            try:
                dev_env = os.environ.get('USBMON_DEV')
                dev_filter = int(dev_env) if dev_env else None
            except Exception:
                dev_filter = None
            active_spans = calculate_active_spans_urbs(
                usbmon_file, invoke_windows_shifted, time_map,
                expand_s=expand_ms/1000.0, dev_filter=dev_filter,
                cin_times_for_cluster=cin_times, cluster_gap_s=gap_s if 'gap_s' in locals() else 0.0001,
                min_cluster_span_s=min_span_s if 'min_span_s' in locals() else 0.000001
            )
            def spans_have_meaningful_urb(spans):
                if not spans:
                    return False
                # 若任何窗口存在URB并集区间或活跃时长>0，则认为URB解析有效
                for s in spans:
                    if (s.get('in_intervals_ms') or s.get('out_intervals_ms')):
                        return True
                    if (s.get('in_active_span_s') or 0.0) > 0.0:
                        return True
                    if (s.get('out_active_span_s') or 0.0) > 0.0:
                        return True
                    if (s.get('union_active_span_s') or 0.0) > 0.0:
                        return True
                # 最后再看字节作为弱信号
                warm = spans[1:] if len(spans) > 1 else spans
                total_bytes = sum((s.get('bytes_in', 0) or 0) + (s.get('bytes_out', 0) or 0) for s in warm)
                return total_bytes > 0
            # 若 URB 解析无有效字节（或无事件），回退到按行统计，与 io_split 行为对齐
            if not active_spans or not spans_have_meaningful_urb(active_spans):
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

        def merge_intervals_with_gap_ms(intervals_ms, bridge_ms: float):
            """在毫秒时间轴上合并区间：若相邻区间间隙 <= bridge_ms，则视为连续合并。"""
            if not intervals_ms:
                return []
            iv = sorted(intervals_ms)
            merged = []
            cs, ce = iv[0]
            for s, t in iv[1:]:
                if s - ce <= bridge_ms:
                    if t > ce:
                        ce = t
                else:
                    merged.append((cs, ce))
                    cs, ce = s, t
            merged.append((cs, ce))
            return merged

    # 可选 off-chip 调整：通过环境变量注入理论 IN 字节与均速
        def _env_float(name, default=None):
            v = os.environ.get(name)
            if v is None:
                return default
            try:
                return float(v)
            except Exception:
                return default

        # 理论 OUT 速率（MiB/s）：优先 OFFCHIP_OUT_THEORY_MIBPS，兼容 OFFCHIP_OUT_MIBPS
        theory_out_mibps = _env_float('OFFCHIP_OUT_THEORY_MIBPS', None)
        if theory_out_mibps is None:
            theory_out_mibps = _env_float('OFFCHIP_OUT_MIBPS', None)
        # 可选开关：仅当 OFFCHIP_ENABLE 为真时启用默认值（避免对所有模型强制校正）
        offchip_enable = (os.environ.get('OFFCHIP_ENABLE', '0').strip().lower() in ('1','true','yes','on'))
        if offchip_enable and theory_out_mibps is None:
            # 默认 320 MiB/s；若显式提供 MiB/ms 再换算
            per_ms = _env_float('OFFCHIP_OUT_THEORY_MIB_PER_MS', None)
            theory_out_mibps = 320.0 if per_ms is None else (per_ms * 1000.0)

        # 可选：在并集口径上桥接小间隙（毫秒）。例如 0.1 表示 <=0.1ms 的空隙视为连续
        try:
            union_gap_bridge_ms = float(os.environ.get('UNION_GAP_BRIDGE_MS', '0'))
        except Exception:
            union_gap_bridge_ms = 0.0

        for i, rec in enumerate(active_spans):
            raw_w = invoke_windows[i]
            aligned_w = invoke_windows_shifted[i] if i < len(invoke_windows_shifted) else {'begin': raw_w['begin'], 'end': raw_w['end'], 'head_expand_ms': 0.0, 'tail_expand_ms': 0.0}
            d = shift_deltas[i] if i < len(shift_deltas) else 0.0
            he_ms = aligned_w.get('head_expand_ms', 0.0) or 0.0
            te_ms = aligned_w.get('tail_expand_ms', 0.0) or 0.0
            # shifted absolute window in usbmon time with per-invoke asymmetric expand
            b = (raw_w['begin'] + d) - bt_ref + usb_ref - ((expand_ms + he_ms)/1000.0 if (expand_ms or he_ms) else 0.0)
            e = (raw_w['end']   + d) - bt_ref + usb_ref + ((expand_ms + te_ms)/1000.0 if (expand_ms or te_ms) else 0.0)
            in_u_s, in_k, in_clusters = cluster_union_within_window(cin_times, b, e, gap_s, min_span_s)
            _, out_k, _ = cluster_union_within_window(cout_times, b, e, gap_s, min_span_s)
            inv_ms = (raw_w['end'] - raw_w['begin']) * 1000.0
            # OUT 的 URB 并集时长（毫秒）来自 URB 解析
            out_urb_ms = (rec.get('out_active_span_s') or 0.0) * 1000.0
            # IN聚簇区间（ms，相对窗口）
            in_cluster_intervals_ms = [((s - b) * 1000.0, (t - b) * 1000.0) for (s, t) in in_clusters]
            # IN混合区间（ms，相对窗口）：URB(S∈window)裁剪 + C聚簇（calculate_active_spans_urbs 已输出）
            in_hybrid_intervals_ms = rec.get('in_intervals_ms') or []
            # OUT 使用 URB 区间（已相对窗口，毫秒）
            out_urb_intervals_ms = rec.get('out_intervals_ms') or []
            # 并集扣除口径（去重）：IN(混合) ∪ OUT(URB)
            io_union_both_ms = union_length_ms(in_hybrid_intervals_ms + out_urb_intervals_ms)
            # 可选：桥接小间隙后的并集
            if union_gap_bridge_ms and union_gap_bridge_ms > 0:
                in_br_ms_intervals = merge_intervals_with_gap_ms(in_hybrid_intervals_ms, union_gap_bridge_ms)
                out_br_ms_intervals = merge_intervals_with_gap_ms(out_urb_intervals_ms, union_gap_bridge_ms)
                union_bridged_ms = union_length_ms(in_br_ms_intervals + out_br_ms_intervals)
                in_union_bridged_ms = union_length_ms(in_br_ms_intervals)
                out_union_bridged_ms = union_length_ms(out_br_ms_intervals)
            else:
                in_union_bridged_ms = None
                out_union_bridged_ms = None
                union_bridged_ms = None
            # 诊断与标准化输出：IN/OUT/Union 并集时长
            in_union_hybrid_ms = union_length_ms(in_hybrid_intervals_ms)
            out_union_ms = union_length_ms(out_urb_intervals_ms)
            # 纯计算（唯一口径）
            pure_compute_ms = inv_ms - io_union_both_ms

            # 诊断：绝对窗口、IN 最后 C
            try:
                in_last_c = max((ts for ts in cin_times if b <= ts <= e), default=None)
                in_last_c_ms = (in_last_c * 1000.0) if in_last_c is not None else None
            except Exception:
                in_last_c_ms = None

            rec['shift_policy'] = shift_policy
            rec['shift_ms'] = d * 1000.0
            rec['cluster_gap_ms'] = cluster_gap_ms
            rec['window_source'] = 'invoke'
            rec['window_expand_ms'] = expand_ms
            rec['window_head_expand_ms'] = he_ms
            rec['window_tail_expand_ms'] = te_ms
            rec['invoke_window_begin_ms'] = b * 1000.0
            rec['invoke_window_end_ms'] = e * 1000.0
            rec['in_last_c_ms'] = in_last_c_ms

            # expose tail guard diagnostics when using tail_last_BiC_guard_BoS
            if plans is not None and i < len(plans):
                rec['tail_shrink_ms'] = (plans[i].get('tail_shrink_s') or 0.0) * 1000.0
                rec['align_reason'] = plans[i].get('reason')

            rec['in_union_cluster_ms'] = in_u_s * 1000.0  # 参考
            rec['in_union_hybrid_ms'] = in_union_hybrid_ms  # 诊断
            rec['out_union_urb_ms'] = out_urb_ms  # 诊断
            # 标准化字段
            rec['in_union_ms'] = in_union_hybrid_ms
            rec['out_union_ms'] = out_union_ms
            rec['union_ms'] = io_union_both_ms
            rec['pure_ms'] = pure_compute_ms
            # 兼容保留（主口径）
            rec['io_union_both_ms'] = io_union_both_ms
            rec['pure_compute_ms'] = pure_compute_ms
            rec['in_clusters'] = in_k
            rec['out_clusters'] = out_k

            # 若开启桥接，输出桥接后的口径与基于桥接的速率与pure
            if union_gap_bridge_ms and union_gap_bridge_ms > 0:
                rec['in_union_bridged_ms'] = in_union_bridged_ms
                rec['out_union_bridged_ms'] = out_union_bridged_ms
                rec['union_bridged_ms'] = union_bridged_ms
                rec['pure_bridged_ms'] = (inv_ms - union_bridged_ms) if (union_bridged_ms is not None) else None
                try:
                    in_s_b = (in_union_bridged_ms or 0.0) / 1000.0
                    out_s_b = (out_union_bridged_ms or 0.0) / 1000.0
                    bin_bytes = float(rec.get('bytes_in', 0) or 0)
                    bout_bytes = float(rec.get('bytes_out', 0) or 0)
                    rec['in_speed_bridged_mibps'] = (bin_bytes / (1024.0 * 1024.0)) / in_s_b if in_s_b > 0 else None
                    rec['out_speed_bridged_mibps'] = (bout_bytes / (1024.0 * 1024.0)) / out_s_b if out_s_b > 0 else None
                except Exception:
                    rec['in_speed_bridged_mibps'] = None
                    rec['out_speed_bridged_mibps'] = None

            # 计算当前 IN/OUT 速率（MiB/s），基于各自并集时长与字节数
            try:
                in_s = (rec.get('in_union_ms') or 0.0) / 1000.0
                out_s = (rec.get('out_union_ms') or 0.0) / 1000.0
                bin_bytes = float(rec.get('bytes_in', 0) or 0)
                bout_bytes = float(rec.get('bytes_out', 0) or 0)
                rec['in_speed_mibps'] = (bin_bytes / (1024.0 * 1024.0)) / in_s if in_s > 0 else None
                rec['out_speed_mibps'] = (bout_bytes / (1024.0 * 1024.0)) / out_s if out_s > 0 else None
            except Exception:
                rec['in_speed_mibps'] = None
                rec['out_speed_mibps'] = None

            # 方向内 span 速率（MiB/s）：以各自 S→C 跨度作为时长；若跨度为0则为 None
            try:
                MiB = 1024.0 * 1024.0
                in_span_ms = rec.get('in_span_sc_ms')
                out_span_ms = rec.get('out_span_sc_ms')
                bin_bytes = float(rec.get('bytes_in', 0) or 0)
                bout_bytes = float(rec.get('bytes_out', 0) or 0)
                rec['in_speed_span_mibps'] = (bin_bytes / MiB) / (in_span_ms / 1000.0) if (in_span_ms and in_span_ms > 0) else None
                rec['out_speed_span_mibps'] = (bout_bytes / MiB) / (out_span_ms / 1000.0) if (out_span_ms and out_span_ms > 0) else None
            except Exception:
                rec['in_speed_span_mibps'] = None
                rec['out_speed_span_mibps'] = None

            # On-chip 假设下的 span 口径：把 OUT 与 IN 的方向内 S→C 跨度相加视为 IO（通常两者不重叠）
            try:
                in_span_ms = float(rec.get('in_span_sc_ms') or 0.0)
                out_span_ms = float(rec.get('out_span_sc_ms') or 0.0)
                io_span_sum_ms = max(0.0, in_span_ms) + max(0.0, out_span_ms)
                # 纯推理时间（span 口径）
                pure_span_sum_ms = inv_ms - io_span_sum_ms
                if pure_span_sum_ms < 0:
                    pure_span_sum_ms = 0.0
                rec['io_span_sum_ms'] = io_span_sum_ms
                rec['pure_span_sum_ms'] = pure_span_sum_ms
            except Exception:
                rec['io_span_sum_ms'] = None
                rec['pure_span_sum_ms'] = None

            # 主口径选择：默认使用 span（简洁有效）；可选 'union' | 'bridged' | 'span'/'span_sum'
            primary_mode = os.environ.get('PRIMARY_IO_MODE', 'span').strip().lower()
            rec['union_primary_mode'] = primary_mode
            try:
                if primary_mode in ('span', 'span_sum') and rec.get('io_span_sum_ms') is not None:
                    # 覆盖主输出
                    rec['union_ms'] = rec['io_span_sum_ms']
                    rec['pure_ms'] = rec['pure_span_sum_ms']
                elif primary_mode == 'bridged' and rec.get('union_bridged_ms') is not None:
                    rec['union_ms'] = rec['union_bridged_ms']
                    rec['pure_ms'] = rec.get('pure_bridged_ms')
            except Exception:
                pass

            # single 口径：改为“pure = last Co(Bo) -> next Bi(S 或 C) 的间隙（毫秒）”。
            # 仅在能找到两端点时覆盖 pure 与 pure_compute。
            try:
                # 查找窗口内最后一个 OUT 完成（Co/Bo on C line）
                # 为避免窗口尾部略晚的 Co 漏选，允许在窗口尾部增加一个尾部容差：SEARCH_TAIL_MS（默认20ms）
                try:
                    tail_ms_for_pure = float(os.environ.get('SEARCH_TAIL_MS', '20'))
                except Exception:
                    tail_ms_for_pure = 20.0
                last_co = last_within(cout_times, b, e + (tail_ms_for_pure/1000.0))
                next_bi_c = first_after(cin_times, last_co) if last_co is not None else None
                next_bi_s = first_after(sin_times, last_co) if last_co is not None else None
                # 选择更早的下一个 Bi 事件
                cand = []
                if next_bi_c is not None:
                    cand.append(next_bi_c)
                if next_bi_s is not None:
                    cand.append(next_bi_s)
                next_bi = min(cand) if cand else None
                gap_ms = ((next_bi - last_co) * 1000.0) if (last_co is not None and next_bi is not None and next_bi >= last_co) else None
                # 诊断字段（绝对ms时间轴，便于核查）
                rec['last_Co_within_ms'] = (last_co * 1000.0) if last_co is not None else None
                rec['next_Bi_after_ms'] = (next_bi * 1000.0) if next_bi is not None else None
                rec['pure_gap_lastCo_to_nextBi_ms'] = gap_ms
                # 若为single-like且gap可用，则覆盖纯推理时间定义
                if single_like and gap_ms is not None and gap_ms >= 0.0:
                    # 覆盖主输出 pure_ms，并同步 pure_compute_ms 作为后续 off-chip 调整的基线
                    rec['pure_ms'] = gap_ms
                    pure_compute_ms = gap_ms
                    rec['pure_compute_ms'] = gap_ms
            except Exception:
                # 保留原有口径
                pass

            # Cross-segment diagnostics: IN carry-in/outside and delays
            # 1) Delay from t1 to first IN completion in next window region
            #    Use SEARCH_TAIL_MS tail to look right after e (shifted t1)
            try:
                first_c_after_t1 = first_after(cin_times, e)
                rec['first_BiC_after_t1_ms'] = ((first_c_after_t1 - e) * 1000.0) if first_c_after_t1 is not None else None
            except Exception:
                rec['first_BiC_after_t1_ms'] = None
            # 2) Last IN completion within +SEARCH_TAIL_MS after t1
            try:
                tail_ms = float(os.environ.get('SEARCH_TAIL_MS', '20'))
            except Exception:
                tail_ms = 20.0
            last_c_in_tail = last_within(cin_times, e, e + tail_ms/1000.0)
            rec['last_BiC_within_+tail_ms'] = ((last_c_in_tail - e) * 1000.0) if last_c_in_tail is not None else None

            # 3) Quantify carry-in IO that starts before window but completes inside
            #    Using URB intervals already clipped: if an IN interval starts < b and ends > b, it's a carry-in slice.
            try:
                in_intervals_ms = rec.get('in_intervals_ms') or []
                carry_in_ms = 0.0
                for s_ms, t_ms in in_intervals_ms:
                    # relative to window begin
                    # carry-in if original URB crossed b; our intervals_ms are already clipped, so we approximate:
                    # treat any interval that begins near 0 as potential carry-in; use a small epsilon to avoid float noise
                    if s_ms <= 0.01 and t_ms > 0.01:
                        carry_in_ms += (t_ms - max(s_ms, 0.0))
                rec['in_carry_from_prev_ms'] = carry_in_ms
            except Exception:
                rec['in_carry_from_prev_ms'] = None

            # off-chip 调整（可选）：给定理论 OUT 速率，按 bytes_out/当前速率 − bytes_out/理论速率 的非负差值加回
            # 当前速率优先采用 OUT 的 span 口径（MiB/s），统一 MiB 单位
            if theory_out_mibps is not None and theory_out_mibps > 0:
                try:
                    cur_mibps = rec.get('out_speed_span_mibps') or rec.get('out_speed_mibps')
                    if cur_mibps and cur_mibps > 0:
                        miB_out = (float(rec.get('bytes_out', 0) or 0)) / (1024.0 * 1024.0)
                        t_cur = miB_out / cur_mibps
                        t_th  = miB_out / theory_out_mibps
                        delta_ms = max(0.0, (t_cur - t_th) * 1000.0)
                        rec['pure_compute_offchip_adj_ms'] = pure_compute_ms + delta_ms
                    else:
                        rec['pure_compute_offchip_adj_ms'] = pure_compute_ms
                except Exception:
                    rec['pure_compute_offchip_adj_ms'] = pure_compute_ms
            enriched.append(rec)
        active_spans = enriched
        active_spans = enriched
        
        # === 验证检查：检测 time_map 对齐问题 ===
        total_bytes_out = sum(r.get('bytes_out', 0) or 0 for r in active_spans)
        non_zero_invokes = sum(1 for r in active_spans if (r.get('bytes_out', 0) or 0) > 0)
        if non_zero_invokes > 0:
            avg_bytes_out = total_bytes_out / non_zero_invokes
            avg_invoke_ms = sum(r.get('invoke_span_s', 0) * 1000 for r in active_spans) / len(active_spans)
            # 启发式检查：如果 invoke > 8ms 但平均 OUT < 1.5MB，可能有对齐问题
            # （典型 MN7/DeepLab 模型 8-50ms 应有 1.5-3MB OUT）
            expected_min_mb = avg_invoke_ms * 0.15  # 约 150KB/ms 的下限
            actual_mb = avg_bytes_out / (1024 * 1024)
            if avg_invoke_ms > 8 and actual_mb < expected_min_mb:
                print(f"[WARNING] 平均 bytes_out={actual_mb:.2f}MB 可能偏低（invoke={avg_invoke_ms:.1f}ms，"
                      f"预期至少 {expected_min_mb:.2f}MB）", file=sys.stderr)
                print(f"[WARNING] 可能是 time_map 对齐问题！多设备测试请为每个设备单独对齐", file=sys.stderr)
        
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
