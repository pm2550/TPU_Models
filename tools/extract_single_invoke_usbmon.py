#!/usr/bin/env python3
import os, sys, re, json
from typing import List, Tuple

def parse_usbmon_lines(cap_file: str) -> List[dict]:
    recs = []
    with open(cap_file, 'r', errors='ignore') as f:
        for ln in f:
            parts = ln.split()
            if len(parts) < 3:
                continue
            # 兼容两类：URB_ID TIMESTAMP ... 或 TIMESTAMP URB_ID ...
            ts = None; urb_id=None; ev='S'
            try:
                # 常见：URB_ID TIMESTAMP SEQ EVENT ...
                float(parts[1]); ts = float(parts[1]); urb_id = parts[0]
                # 事件位可能在 parts[2]
                ev = parts[2] if len(parts)>2 else 'S'
                ts = ts/1e6 if ts>1e6 else ts
            except Exception:
                try:
                    # 备选：TIMESTAMP URB_ID SEQ EVENT ...
                    float(parts[0]); ts = float(parts[0]); urb_id = parts[1]
                    ev = parts[2] if len(parts)>2 else 'S'
                except Exception:
                    continue
            # 容错：若 ev 不是 S/C，则尝试从后续字段探测
            if ev not in ('S','C'):
                for t in parts:
                    if t in ('S','C'):
                        ev = t; break
            dir_tok=None; idx=None
            for i,t in enumerate(parts[3:], start=3):
                if re.match(r'^[BC][io]:\d+:\d+:\d+', t):
                    dir_tok=t[:2]; idx=i; break
            if not dir_tok: continue
            nb=None
            m = re.search(r'len=(\d+)', ln)
            if m:
                nb = int(m.group(1))
            elif idx is not None and idx+2 < len(parts):
                try:
                    nb = int(parts[idx+2])
                except Exception:
                    nb = None
            recs.append({'ln': ln.rstrip('\n'), 'ts': ts, 'urb': urb_id, 'ev': ev, 'dir': dir_tok, 'len': nb})
    return recs

def pair_urbs(recs: List[dict]) -> List[dict]:
    sub = {}
    pairs = []
    for r in recs:
        if r['dir'] not in ('Bi','Bo'): continue
        if r['ev'] == 'S':
            sub[r['urb']] = r
        elif r['ev'] == 'C':
            s = sub.pop(r['urb'], None)
            if not s: continue
            t0 = s['ts']; t1 = r['ts']
            if t1 is None or t0 is None: continue
            pairs.append({'dir': s['dir'], 'len': (r['len'] if r['len'] is not None else (s['len'] or 0) or 0), 't0': t0, 't1': t1})
    return pairs

def merge_len(intervals: List[Tuple[float,float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals, key=lambda x: x[0])
    cs, ce = intervals[0]
    total = 0.0
    for s,e in intervals[1:]:
        if s <= ce:
            if e > ce: ce = e
        else:
            total += (ce - cs)
            cs, ce = s, e
    total += (ce - cs)
    return total

def main():
    if len(sys.argv) < 6:
        print('usage: extract_single_invoke_usbmon.py <combo_root> <seg_label> <invoke_index> <pre_ms> <post_ms> [start]')
        sys.exit(2)
    combo_root, seg_label, idx_s, pre_ms_s, post_ms_s = sys.argv[1:6]
    mode = sys.argv[6] if len(sys.argv) >= 7 else ''
    invoke_index = int(idx_s)
    pre_s = float(pre_ms_s)/1000.0
    post_s = float(post_ms_s)/1000.0
    usb = os.path.join(combo_root, 'usbmon.txt')
    merged = os.path.join(combo_root, 'merged_invokes.json')
    tm = os.path.join(combo_root, 'time_map.json')
    assert os.path.isfile(usb) and os.path.isfile(merged) and os.path.isfile(tm)
    spans = json.load(open(merged)).get('spans', [])
    tmj = json.load(open(tm))
    usb_ref = tmj.get('usbmon_ref'); bt_ref = tmj.get('boottime_ref')
    idxs = [i for i,s in enumerate(spans) if s.get('seg_label') == seg_label]
    if not idxs:
        print('no spans for', seg_label); sys.exit(1)
    if invoke_index < 0 or invoke_index >= len(idxs):
        print('invoke_index out of range, got', invoke_index, 'max', len(idxs)-1); sys.exit(1)
    si = idxs[invoke_index]
    w = spans[si]
    ob = w['begin']; oe = w['end']
    if usb_ref is None or bt_ref is None:
        print('time_map missing refs'); sys.exit(1)
    b0 = (ob - bt_ref) + usb_ref
    e0 = (oe - bt_ref) + usb_ref
    # 片段范围：默认覆盖整个窗口的前后；若指定 start 模式，仅围绕窗口起点
    if mode.strip().lower() == 'start':
        sb = b0 - pre_s
        se = b0 + post_s
    else:
        sb = b0 - pre_s
        se = e0 + post_s
    # 解析 usbmon 并输出片段
    all_recs = parse_usbmon_lines(usb)
    # 原始片段（原样行）
    lines = [r['ln'] for r in all_recs if (r['ts'] is not None and sb <= r['ts'] <= se)]
    suffix = '_start' if mode.strip().lower() == 'start' else ''
    out_snip = os.path.join(combo_root, f'{seg_label}_invoke{invoke_index:03d}_usbmon_snippet{suffix}.txt')
    with open(out_snip, 'w') as f:
        f.write('\n'.join(lines))
    # 计算度量（只算 Bi/Bo 对），并估计时间偏移 Δ
    pairs = pair_urbs(all_recs)
    # 选出与扩窗片段有交叠的 URB
    cand = [p for p in pairs if not (p['t1'] < sb or p['t0'] > se)]
    # 估计 Δ（对齐原始窗口与 URB 开始/结束的偏移中值）
    delta = 0.0
    if cand:
        import statistics as _st
        d1 = _st.median([p['t0'] - b0 for p in cand])
        d2 = _st.median([p['t1'] - e0 for p in cand])
        delta = 0.5 * (d1 + d2)
        if delta > 0.5: delta = 0.5
        if delta < -0.5: delta = -0.5
    # 用 (b0+Δ, e0+Δ) 做裁剪
    cb0 = b0 + delta
    ce0 = e0 + delta
    in_iv = []; out_iv = []
    binb = boutb = 0
    for p in cand:
        t0 = p['t0']; t1 = p['t1']
        ss = max(cb0, t0); ee = min(ce0, t1)
        if ee <= ss:
            continue
        if p['dir'] == 'Bi':
            binb += (p['len'] or 0); in_iv.append((ss, ee))
        else:
            boutb += (p['len'] or 0); out_iv.append((ss, ee))
    in_iv = [(s,e) for (s,e) in in_iv if e> s]
    out_iv = [(s,e) for (s,e) in out_iv if e> s]
    in_len = merge_len(in_iv)
    out_len = merge_len(out_iv)
    union_len = merge_len(in_iv + out_iv)
    overlap_len = max(0.0, in_len + out_len - union_len)
    invoke_ms = (oe - ob) * 1000.0
    print(json.dumps({
        'seg_label': seg_label,
        'invoke_index': invoke_index,
        'invoke_begin_bt': ob,
        'invoke_end_bt': oe,
        'invoke_ms': invoke_ms,
        'usb_window_s': [sb, se],
        'mapped_window_s': [b0, e0],
        'snippet_path': out_snip,
        'bytes_in': binb,
        'bytes_out': boutb,
        'in_active_ms': in_len*1000.0,
        'out_active_ms': out_len*1000.0,
        'union_active_ms': union_len*1000.0,
        'overlap_active_ms': overlap_len*1000.0,
        'coverage': (overlap_len / in_len) if in_len>0 else 0.0
    }, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()


