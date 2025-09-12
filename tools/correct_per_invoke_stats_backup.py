#!/usr/bin/env python3
"""
基于原始 run_usbmon_capture_offline.sh 的正确逻辑重写的逐次统计
"""
import json
import sys
import re
import os
import argparse
from typing import Optional


def parse_records(cap_file):
    """解析 usbmon.txt，返回每行记录字典，用于后续 URB S/C 配对。
    字段：urb_id, ts, ev(S/C), dir(Bi/Bo/Ci/Co), len(bytes, 若无len=则None), dev, ep
    """
    recs = []
    with open(cap_file, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                # 标准 usbmon 格式："TIMESTAMP URB_ID SEQ_NUM EVENT ..."
                ts = float(parts[0])
                urb_id = parts[1]
                ev = parts[3] if len(parts) > 3 else 'S'
            except Exception:
                # 兼容其他格式："URB_ID TIMESTAMP ..."
                try:
                    urb_id = parts[0]
                    ts = float(parts[1])
                    ts = (ts/1e6) if ts > 1e6 else ts
                    ev = parts[2] if len(parts) > 2 else 'S'
                except Exception:
                    continue
            
            # 过滤无效 urb_id（必须以十六进制/指针样式开头）
            if not re.match(r'^[0-9a-fA-Fx]+', urb_id):
                continue

            # 查找事件类型及总线/设备/端点 (Bi:BUS:DEV:EP) - 通常在第5列
            tok = None
            dev = None
            ep = None
            idx = None
            # 方向字段通常在第4或第5列（索引3或4），向后扫描以兼容不同内核格式
            # 先从索引3开始（可匹配形如：URB_ID TIMESTAMP S Ci:...）
            start_idx = 3 if len(parts) > 3 else 4
            for i, t in enumerate(parts[start_idx:], start=start_idx):
                if re.match(r'^[BC][io]:\d+:\d+:\d+', t):
                    tok = t[:2]
                    try:
                        m = re.match(r'^[BC][io]:(\d+):(\d+):(\d+)', t)
                        if m:
                            # bus = int(m.group(1))  # 未使用
                            dev = int(m.group(2))
                            ep = int(m.group(3))
                    except Exception:
                        pass
                    idx = i
                    break
            if not tok:
                continue

            # 查找字节数：优先 len=，否则从固定位置解析
            nb = None
            mlen = re.search(r'len=(\d+)', ln)
            if mlen:
                try:
                    nb = int(mlen.group(1))
                except Exception:
                    nb = None
            elif idx is not None and len(parts) > idx + 2:
                # 没有 len= 时，尝试从方向字段后2位解析字节数
                try:
                    nb = int(parts[idx + 2])
                except Exception:
                    nb = None

            recs.append({
                'urb_id': urb_id,
                'ts': ts,
                'ev': ev,
                'dir': tok,
                'len': nb,
                'dev': dev,
                'ep': ep,
            })

    return recs

def build_urb_pairs(records):
    """将 S/C 事件按 urb_id 配对，返回已配对的 URB 列表。
    每个条目包含：dir(Bi/Bo/Ci/Co)、dev、ep、len(bytes from S)、ts_submit、ts_complete、duration
    若 S 无 len=，则 len 视为 0。
    """
    submit_map = {}
    pairs = []
    for r in records:
        urb_id = r['urb_id']
        ev = r['ev']
        if ev == 'S':
            submit_map[urb_id] = r
        elif ev == 'C':
            s = submit_map.pop(urb_id, None)
            if not s:
                continue
            # 方向以 S 行为准（更稳定）
            direction = s['dir']
            if direction not in ('Bi', 'Bo', 'Ci', 'Co'):
                continue
            dev = s['dev'] if s['dev'] is not None else r['dev']
            ep = s['ep'] if s['ep'] is not None else r['ep']
            # 使用 C 行的 actual_length 优先；若无，则退回 S 行的 buffer 长度
            length = r['len'] if r.get('len') is not None else (s['len'] if s.get('len') is not None else 0)
            ts_submit = s['ts']
            ts_complete = r['ts']
            duration = max(0.0, ts_complete - ts_submit)
            pairs.append({
                'urb_id': urb_id,
                'dir': direction,
                'dev': dev,
                'ep': ep,
                'len': length,
                'ts_submit': ts_submit,
                'ts_complete': ts_complete,
                'duration': duration,
            })
    return pairs


def stat_window(pairs, win, usb_ref, bt_ref, epoch_ref=None, extra=0.0, mode: str = 'default', *, allow_dev=None, allow_ep_in=None, allow_ep_out=None, include: str = 'full', seen_urb_ids: Optional[set] = None):
    """统计单个窗口的传输，完全按照原始脚本逻辑"""
    # 时间对齐：如果没有usbmon_ref但有epoch_ref，用epoch_ref对齐
    if usb_ref is None and epoch_ref is not None:
        # usbmon时间戳是绝对时间，invoke窗口也是绝对时间，直接比较
        b0 = win['begin'] - extra
        e0 = win['end'] + extra
    elif usb_ref is not None and bt_ref is not None:
        # 原始逻辑：转换到usbmon时间轴
        b0 = (win['begin'] - bt_ref) + usb_ref - extra
        e0 = (win['end'] - bt_ref) + usb_ref + extra
    else:
        # 回退：视为绝对时间轴一致（与 show_overlap_positions.py 一致的宽容处理）
        b0 = win['begin'] - extra
        e0 = win['end'] + extra
    
    bi = bo = ci = co = 0
    binb = boutb = 0
    in_times = []
    out_times = []
    in_intervals = []  # (start, end) clipped to [b0,e0]
    out_intervals = []

    if mode == 'bulk_complete':
        # 严格口径：仅统计已配对的 Bulk URB，且完全落入窗口（S>=b0 且 C<=e0）
        for p in pairs:
            d = p['dir']
            if d not in ('Bi', 'Bo'):
                continue
            if allow_dev is not None and p['dev'] is not None and p['dev'] != allow_dev:
                continue
            if d == 'Bi' and allow_ep_in is not None and p['ep'] is not None and p['ep'] != allow_ep_in:
                continue
            if d == 'Bo' and allow_ep_out is not None and p['ep'] is not None and p['ep'] != allow_ep_out:
                continue
            # 修正：不要对 URB 时间再次应用 extra，extra 仅对窗口扩展
            ts_s = p['ts_submit']
            ts_c = p['ts_complete']
            # 包含规则：full=完全包含；overlap=任意交叠
            inside = (ts_s >= b0 and ts_c <= e0) if include == 'full' else (ts_c >= b0 and ts_s <= e0)
            if inside:
                # 跨窗去重：同一 URB 只计一次
                if seen_urb_ids is not None:
                    uid = p.get('urb_id')
                    if uid in seen_urb_ids:
                        continue
                    seen_urb_ids.add(uid)
                # 过滤异常小长包
                if p['len'] < 64 and p['duration'] > 0.002:
                    continue
                # 只统计≥1KB的数据传输（实际传输窗口）
                if p['len'] < 1024:
                    continue
                if d == 'Bi':
                    bi += 1
                    binb += p['len']
                    in_times.append(p['ts_complete'])
                    # 记录区间并裁剪到窗口
                    ss = max(b0, p['ts_submit'])
                    ee = min(e0, p['ts_complete'])
                    if ee > ss:
                        in_intervals.append((ss, ee))
                else:
                    bo += 1
                    boutb += p['len']
                    out_times.append(p['ts_complete'])
                    ss = max(b0, p['ts_submit'])
                    ee = min(e0, p['ts_complete'])
                    if ee > ss:
                        out_intervals.append((ss, ee))
    else:
        # 兼容旧口径（不推荐）：基于行级别统计
        for r in pairs:
            pass  # 不实现旧逻辑的细节，避免误用
    
    span = e0 - b0
    toMB = lambda x: x / (1024 * 1024.0)
    
    # 计算活跃时长（真并集）：对 URB 区间在窗口内的并集长度
    def _merge_len(intervals):
        if not intervals:
            return 0.0
        intervals = sorted(intervals, key=lambda x: x[0])
        cur_s, cur_e = intervals[0]
        total = 0.0
        for s, e in intervals[1:]:
            if s <= cur_e:
                if e > cur_e:
                    cur_e = e
            else:
                total += (cur_e - cur_s)
                cur_s, cur_e = s, e
        total += (cur_e - cur_s)
        return total
    in_active_s = _merge_len(in_intervals)
    out_active_s = _merge_len(out_intervals)
    union_active_s = _merge_len(in_intervals + out_intervals)
    overlap_active_s = max(0.0, in_active_s + out_active_s - union_active_s)
    
    return {
        'span_s': span,
        'Bi': bi, 'Bo': bo, 'Ci': ci, 'Co': co,
        'bytes_in': binb, 'bytes_out': boutb,
        'MBps_in': (toMB(binb) / span if span > 0 else 0.0),
        'MBps_out': (toMB(boutb) / span if span > 0 else 0.0),
        'in_active_s': in_active_s,
        'out_active_s': out_active_s,
        'union_active_s': union_active_s,
        'overlap_active_s': overlap_active_s,
        'in_active_MBps': (toMB(binb) / in_active_s if in_active_s > 0 else 0.0),
        'out_active_MBps': (toMB(boutb) / out_active_s if out_active_s > 0 else 0.0)
    }


def main():
    parser = argparse.ArgumentParser(description='正确的逐次 usbmon 统计')
    parser.add_argument('usbmon_txt', help='usbmon.txt 文件路径')
    parser.add_argument('invokes_json', help='invokes.json 文件路径')
    parser.add_argument('time_map_json', help='time_map.json 文件路径')
    parser.add_argument('--details', action='store_true', help='显示每次详情')
    parser.add_argument('--extra', type=float, default=0.0, help='扩展窗口秒数')
    parser.add_argument('--mode', choices=['default','bulk_complete'], default='default', help='统计口径')
    parser.add_argument('--include', choices=['full','overlap'], default='overlap', help='URB 包含规则：full=完全在窗口内，overlap=任意交叠')
    parser.add_argument('--dev', type=int, default=None, help='仅统计指定设备号(DEV)')
    parser.add_argument('--ep-in', dest='ep_in', type=int, default=None, help='仅统计指定IN端点号(EP)')
    parser.add_argument('--ep-out', dest='ep_out', type=int, default=None, help='仅统计指定OUT端点号(EP)')
    
    args = parser.parse_args()
    
    # 加载时间映射
    if os.path.exists(args.time_map_json):
        tm = json.load(open(args.time_map_json))
    else:
        tm = {'usbmon_ref': None, 'boottime_ref': None}
    
    usb_ref = tm.get('usbmon_ref')
    bt_ref = tm.get('boottime_ref')
    epoch_ref = tm.get('epoch_ref_at_usbmon')
    
    print(f"时间映射: usbmon_ref={usb_ref}, boottime_ref={bt_ref}, epoch_ref={epoch_ref}")
    
    # 加载 invoke 窗口
    iv = json.load(open(args.invokes_json))['spans']
    print(f"总 invoke 数: {len(iv)}")
    
    # 解析 usbmon 记录
    records = parse_records(args.usbmon_txt)
    print(f"总 usbmon 记录: {len(records)}")
    pairs = build_urb_pairs(records)
    print(f"已配对 URB: {len(pairs)}")

    # 自动探测将分两步：
    # 1) 粗探测（全文件），便于观察
    # 2) 若后续统计为0，则基于“与窗口有交叠的URB”重探测
    auto_dev = args.dev
    auto_ep_in = args.ep_in
    auto_ep_out = args.ep_out
    def autodetect_from_records(recs):
        nonlocal auto_dev, auto_ep_in, auto_ep_out
        if args.dev is not None and args.ep_in is not None and args.ep_out is not None:
            return
        bytes_by_dev = {}
        bytes_by_ep_in = {}
        bytes_by_ep_out = {}
        for p in recs:
            d = p['dir']
            if d not in ('Bi', 'Bo'):
                continue
            dev = p['dev']; ep = p['ep']; nb = p['len'] or 0
            if dev is not None:
                bytes_by_dev[dev] = bytes_by_dev.get(dev, 0) + nb
            if d == 'Bi' and ep is not None:
                bytes_by_ep_in[ep] = bytes_by_ep_in.get(ep, 0) + nb
            if d == 'Bo' and ep is not None:
                bytes_by_ep_out[ep] = bytes_by_ep_out.get(ep, 0) + nb
        if auto_dev is None and bytes_by_dev:
            auto_dev = max(bytes_by_dev.items(), key=lambda x: x[1])[0]
        if auto_ep_in is None and bytes_by_ep_in:
            auto_ep_in = max(bytes_by_ep_in.items(), key=lambda x: x[1])[0]
        if auto_ep_out is None and bytes_by_ep_out:
            auto_ep_out = max(bytes_by_ep_out.items(), key=lambda x: x[1])[0]
    autodetect_from_records(pairs)
    # 环境变量覆盖（若提供）
    env_dev = os.environ.get('USBMON_DEV')
    if env_dev and env_dev.isdigit():
        auto_dev = int(env_dev)
    print(f"使用过滤(初探): DEV={auto_dev}, EP_IN={auto_ep_in}, EP_OUT={auto_ep_out}")
    
    # 若使用“任意交叠”，采用“最大重叠分配”逻辑，避免跨窗重复与零计数
    results = []
    if args.include == 'overlap':
        # 预计算每个窗口的时间边界（映射到 usbmon 时间）
        def map_win_extra(w):
            if usb_ref is None and bt_ref is None:
                return (w['begin'] - args.extra, w['end'] + args.extra)
            else:
                b0 = (w['begin'] - bt_ref) + (usb_ref or 0.0) - args.extra
                e0 = (w['end'] - bt_ref) + (usb_ref or 0.0) + args.extra
                return (b0, e0)
        def map_win_orig(w):
            if usb_ref is None and bt_ref is None:
                return (w['begin'], w['end'])
            else:
                b0 = (w['begin'] - bt_ref) + (usb_ref or 0.0)
                e0 = (w['end'] - bt_ref) + (usb_ref or 0.0)
                return (b0, e0)
        win_bounds = [map_win_extra(w) for w in iv]
        win_bounds_orig = [map_win_orig(w) for w in iv]
        # 初始化结果占位
        for _ in iv:
            results.append({'span_s': 0.0,'Bi':0,'Bo':0,'Ci':0,'Co':0,'bytes_in':0,'bytes_out':0,
                            'MBps_in':0.0,'MBps_out':0.0,'in_active_s':0.0,'out_active_s':0.0,
                            'union_active_s':0.0,'overlap_active_s':0.0})
        in_times = [[] for _ in iv]
        out_times = [[] for _ in iv]
        # 为“归因后度量”保留原始 URB 区间（未裁剪，绝对时间）
        in_urbs = [[] for _ in iv]   # list[(t0,t1)]
        out_urbs = [[] for _ in iv]
        # 若仍未指定 dev/ep，则在“与任一窗口有交叠的URB集合”上再探测一次，避免被无关设备干扰
        if auto_dev is None or auto_ep_in is None or auto_ep_out is None:
            overlapped_pairs = []
            for p in pairs:
                if p['dir'] not in ('Bi','Bo'):
                    continue
                t0 = p['ts_submit']; t1 = p['ts_complete']
                for (b0,e0) in win_bounds:
                    if (t1 >= b0 and t0 <= e0):
                        overlapped_pairs.append(p)
                        break
            autodetect_from_records(overlapped_pairs)
            print(f"使用过滤(窗内重探): DEV={auto_dev}, EP_IN={auto_ep_in}, EP_OUT={auto_ep_out}")
        # 过滤目标对（DEV/EP）- 修复：允许多个输入端点
        def allow(p):
            if p['dir'] not in ('Bi','Bo'):
                return False
            if auto_dev is not None and p['dev'] is not None and p['dev'] != auto_dev:
                return False
            # 对于输入端点，允许端点1和端点2（TPU常用多端点输入）
            if p['dir']=='Bi' and auto_ep_in is not None and p['ep'] is not None:
                if p['ep'] not in [1, 2]:  # 只允许端点1和2的输入
                    return False
            if p['dir']=='Bo' and auto_ep_out is not None and p['ep'] is not None and p['ep'] != auto_ep_out:
                return False
            return True
        # 为每个 URB 选择重叠最大的窗口并分配一次
        for p in pairs:
            if not allow(p):
                continue
            t0 = p['ts_submit']; t1 = p['ts_complete']
            if p['len'] < 64 and p['duration'] > 0.002:
                continue
            best_i = -1; best_ov = 0.0
            for i,(b0,e0) in enumerate(win_bounds):
                ov = max(0.0, min(e0, t1) - max(b0, t0))
                if ov > best_ov:
                    best_ov = ov; best_i = i
            if best_i < 0 or best_ov <= 0:
                continue
            r = results[best_i]
            if p['dir']=='Bi':
                r['Bi'] += 1; r['bytes_in'] += (p['len'] or 0); in_times[best_i].append(t1)
                in_urbs[best_i].append((t0, t1))
            else:
                r['Bo'] += 1; r['bytes_out'] += (p['len'] or 0); out_times[best_i].append(t1)
                out_urbs[best_i].append((t0, t1))
        # 填充 span/速率与活跃时间
        # 估计全局时间偏移 Δ：使用各窗口最早提交与最晚完成相对原始 begin/end 的偏差的中值
        deltas = []
        for i,(b0e,e0e) in enumerate(win_bounds):
            ob0, oe0 = win_bounds_orig[i]
            s_list = [t0 for (t0,_) in in_urbs[i] + out_urbs[i]]
            c_list = [t1 for (_,t1) in in_urbs[i] + out_urbs[i]]
            if s_list and c_list:
                d1 = (min(s_list) - ob0)
                d2 = (max(c_list) - oe0)
                deltas.append(0.5 * (d1 + d2))
        # 取中位数作为全局 Δ，限制在 ±0.2s 内避免异常
        def _median(xs):
            xs = sorted(xs)
            n = len(xs)
            if n == 0:
                return 0.0
            if n % 2 == 1:
                return xs[n//2]
            return 0.5 * (xs[n//2 - 1] + xs[n//2])
        global_delta = _median(deltas)
        if global_delta > 0.2:
            global_delta = 0.2
        if global_delta < -0.2:
            global_delta = -0.2

        for i,(b0,e0) in enumerate(win_bounds):
            r = results[i]
            # 原始窗口跨度（无extra），用于所有速率与上限
            ob0, oe0 = win_bounds_orig[i]
            span_orig = max(0.0, oe0 - ob0)
            r['span_s'] = span_orig
            toMB = lambda x: x/(1024*1024.0)
            r['MBps_in'] = (toMB(r['bytes_in'])/span_orig) if span_orig>0 else 0.0
            r['MBps_out'] = (toMB(r['bytes_out'])/span_orig) if span_orig>0 else 0.0
            def _merge_len(intervals):
                if not intervals:
                    return 0.0
                intervals = sorted(intervals, key=lambda x: x[0])
                cur_s, cur_e = intervals[0]
                total = 0.0
                for s, e in intervals[1:]:
                    if s <= cur_e:
                        if e > cur_e:
                            cur_e = e
                    else:
                        total += (cur_e - cur_s)
                        cur_s, cur_e = s, e
                total += (cur_e - cur_s)
                return total
            # 使用平移后的原始窗口进行裁剪度量
            sob0 = ob0 + global_delta
            soe0 = oe0 + global_delta
            in_intervals_meas = []
            out_intervals_meas = []
            for (t0,t1) in in_urbs[i]:
                ss = max(sob0, t0); ee = min(soe0, t1)
                if ee > ss:
                    in_intervals_meas.append((ss, ee))
            for (t0,t1) in out_urbs[i]:
                ss = max(sob0, t0); ee = min(soe0, t1)
                if ee > ss:
                    out_intervals_meas.append((ss, ee))
            in_len = _merge_len(in_intervals_meas)
            out_len = _merge_len(out_intervals_meas)
            union_len = _merge_len(in_intervals_meas + out_intervals_meas)
            r['in_active_s'] = in_len
            r['out_active_s'] = out_len
            r['union_active_s'] = union_len
            r['overlap_active_s'] = max(0.0, in_len + out_len - union_len)
            r['invoke'] = i
        # 若统计仍为0，最后兜底：不使用 dev/ep 过滤（仅 Bi/Bo），避免被错误过滤清空
        if sum(r['bytes_in']+r['bytes_out'] for r in results) == 0:
            print("注意: 窗口统计为0，启用无dev/ep过滤兜底")
            results = [{'span_s': r['span_s'],'Bi':0,'Bo':0,'Ci':0,'Co':0,'bytes_in':0,'bytes_out':0,
                        'MBps_in':r['MBps_in'],'MBps_out':r['MBps_out'],'in_active_s':0.0,'out_active_s':0.0}
                       for r in results]
            in_times = [[] for _ in iv]
            out_times = [[] for _ in iv]
            for p in pairs:
                if p['dir'] not in ('Bi','Bo'):
                    continue
                t0 = p['ts_submit']; t1 = p['ts_complete']
                if p['len'] < 64 and p['duration'] > 0.002:
                    continue
                # 只统计≥1KB的数据传输（实际传输窗口）
                if p['len'] < 1024:
                    continue
                best_i = -1; best_ov = 0.0
                for i,(b0,e0) in enumerate(win_bounds):
                    ov = max(0.0, min(e0, t1) - max(b0, t0))
                    if ov > best_ov:
                        best_ov = ov; best_i = i
                if best_i < 0 or best_ov <= 0:
                    continue
                r = results[best_i]
                if p['dir']=='Bi':
                    r['Bi'] += 1; r['bytes_in'] += (p['len'] or 0); in_times[best_i].append(t1)
                    in_urbs[best_i].append((t0, t1))
                else:
                    r['Bo'] += 1; r['bytes_out'] += (p['len'] or 0); out_times[best_i].append(t1)
                    out_urbs[best_i].append((t0, t1))
            # 活跃时间
            for i in range(len(results)):
                def _merge_len(intervals):
                    if not intervals:
                        return 0.0
                    intervals = sorted(intervals, key=lambda x: x[0])
                    cur_s, cur_e = intervals[0]
                    total = 0.0
                    for s, e in intervals[1:]:
                        if s <= cur_e:
                            if e > cur_e:
                                cur_e = e
                        else:
                            total += (cur_e - cur_s)
                            cur_s, cur_e = s, e
                    total += (cur_e - cur_s)
                    return total
                ob0,oe0 = win_bounds_orig[i]
                sob0 = ob0 + 0.0
                soe0 = oe0 + 0.0
                # 兜底路径保持不移位
                in_intervals_meas = []
                out_intervals_meas = []
                for (t0,t1) in in_urbs[i]:
                    ss = max(sob0, t0); ee = min(soe0, t1)
                    if ee > ss:
                        in_intervals_meas.append((ss, ee))
                for (t0,t1) in out_urbs[i]:
                    ss = max(sob0, t0); ee = min(soe0, t1)
                    if ee > ss:
                        out_intervals_meas.append((ss, ee))
                in_len = _merge_len(in_intervals_meas)
                out_len = _merge_len(out_intervals_meas)
                union_len = _merge_len(in_intervals_meas + out_intervals_meas)
                results[i]['in_active_s'] = in_len
                results[i]['out_active_s'] = out_len
                results[i]['union_active_s'] = union_len
                results[i]['overlap_active_s'] = max(0.0, in_len + out_len - union_len)
    else:
        # 原有逐窗口统计（全包含）
        for i, win in enumerate(iv):
            seen_ids = set()
            stat = stat_window(pairs, win, usb_ref, bt_ref, epoch_ref, args.extra, args.mode,
                               allow_dev=auto_dev, allow_ep_in=auto_ep_in, allow_ep_out=auto_ep_out,
                               include=args.include, seen_urb_ids=seen_ids)
            stat['invoke'] = i
            results.append(stat)
        
        if args.details:
            print(f"invoke {i}: IN={stat['bytes_in']:,}B OUT={stat['bytes_out']:,}B "
                  f"Bi={stat['Bi']} Bo={stat['Bo']} Ci={stat['Ci']} Co={stat['Co']} "
                  f"in_active={stat['in_active_s']:.4f}s out_active={stat['out_active_s']:.4f}s")
    
    # 计算 warm 摘要 (跳过第一次)
    if len(results) > 1:
        warm_results = results[1:]
        total_in = sum(r['bytes_in'] for r in warm_results)
        total_out = sum(r['bytes_out'] for r in warm_results)
        total_in_active = sum(r['in_active_s'] for r in warm_results)
        total_out_active = sum(r['out_active_s'] for r in warm_results)
        non_zero_in = len([r for r in warm_results if r['bytes_in'] > 0])
        non_zero_out = len([r for r in warm_results if r['bytes_out'] > 0])
        
        print(f"\n=== Warm 摘要 (跳过第1次) ===")
        print(f"Warm invoke 数: {len(warm_results)}")
        avg_in = total_in/len(warm_results)
        avg_out = total_out/len(warm_results)
        avg_in_active = total_in_active/len(warm_results)
        avg_out_active = total_out_active/len(warm_results)
        print(f"IN:  总字节={total_in:,}, 平均={avg_in:.0f}, 非零次数={non_zero_in}")
        print(f"OUT: 总字节={total_out:,}, 平均={avg_out:.0f}, 非零次数={non_zero_out}")
        print(f"平均传输: IN={avg_in/1024/1024:.3f}MB, OUT={avg_out/1024/1024:.3f}MB")
        print(f"平均活跃时长: IN={avg_in_active:.4f}s, OUT={avg_out_active:.4f}s")
        print(f"平均活跃速度: IN={avg_in/1024/1024/avg_in_active if avg_in_active > 0 else 0:.1f}MB/s, OUT={avg_out/1024/1024/avg_out_active if avg_out_active > 0 else 0:.1f}MB/s")
        # 机器可解析的 JSON 摘要
        import json as _json
        json_summary = {
            'dev': auto_dev,
            'ep_in': auto_ep_in,
            'ep_out': auto_ep_out,
            'warm_invokes': len(warm_results),
            'warm_avg_in_bytes': avg_in,
            'warm_avg_out_bytes': avg_out,
            'warm_total_in_bytes': total_in,
            'warm_total_out_bytes': total_out,
            'warm_avg_in_active_s': avg_in_active,
            'warm_avg_out_active_s': avg_out_active
        }
        print('JSON_SUMMARY:', _json.dumps(json_summary, ensure_ascii=False))

    # 机器可解析的逐窗口结果（不跳过首个），便于上层做“按段聚合”
    try:
        import json as _json
        per_invoke_payload = [
            {
                'invoke': int(i),
                'bytes_in': int(r.get('bytes_in', 0)),
                'bytes_out': int(r.get('bytes_out', 0)),
                'span_s': float(r.get('span_s', 0.0)),
                'in_active_s': float(r.get('in_active_s', 0.0)),
                'out_active_s': float(r.get('out_active_s', 0.0)),
                'union_active_s': float(r.get('union_active_s', 0.0)),
                'overlap_active_s': float(r.get('overlap_active_s', 0.0)),
            }
            for i, r in enumerate(results)
        ]
        print('JSON_PER_INVOKE:', _json.dumps(per_invoke_payload, ensure_ascii=False))
    except Exception:
        pass


if __name__ == '__main__':
    main()
