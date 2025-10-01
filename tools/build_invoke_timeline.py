#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                # best-effort: allow trailing commas or stray logs
                pass
    return rows


def hexptr(p: Any) -> str:
    if isinstance(p, str):
        return p
    # Python's json may already give strings; ensure hex-like
    try:
        return hex(int(p))
    except Exception:
        return str(p)


def summarize(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    invokes: Dict[int, Dict[str, Any]] = {}
    submits: Dict[str, Dict[str, Any]] = {}  # urbp -> submit
    reaps: List[Dict[str, Any]] = []
    mems: List[Dict[str, Any]] = []

    for ev in logs:
        typ = ev.get('ev')
        if typ in ('INV_BEGIN', 'INV_END'):
            inv = int(ev.get('invoke') or 0)
            if inv not in invokes:
                invokes[inv] = {
                    'inv_begin': None,
                    'inv_end': None,
                    'submits': [],
                    'reaps': [],
                    'mems': [],
                }
            if typ == 'INV_BEGIN':
                invokes[inv]['inv_begin'] = float(ev['ts'])
            else:
                invokes[inv]['inv_end'] = float(ev['ts'])
        elif typ == 'SUBMITURB':
            urbp = hexptr(ev.get('urbp'))
            ev['ts'] = float(ev['ts'])
            ev['dir'] = str(ev.get('dir') or '')
            submits[urbp] = ev
            inv = int(ev.get('invoke') or 0)
            invokes.setdefault(inv, {'inv_begin': None, 'inv_end': None, 'submits': [], 'reaps': [], 'mems': []})
            invokes[inv]['submits'].append(ev)
        elif typ in ('REAPURB', 'REAPURBNDELAY'):
            urbp = hexptr(ev.get('urbp'))
            ev['ts'] = float(ev['ts'])
            ev['dir'] = str(ev.get('dir') or '')
            ev['urbp'] = urbp
            reaps.append(ev)
            inv = int(ev.get('invoke') or 0)
            invokes.setdefault(inv, {'inv_begin': None, 'inv_end': None, 'submits': [], 'reaps': [], 'mems': []})
            invokes[inv]['reaps'].append(ev)
        elif typ in ('MEMCPY', 'MEMMOVE'):
            ev['ts'] = float(ev['ts'])
            ev['n'] = int(ev.get('n') or 0)
            ev['dt_ms'] = float(ev.get('dt_ms') or 0.0)
            inv = int(ev.get('invoke') or 0)
            invokes.setdefault(inv, {'inv_begin': None, 'inv_end': None, 'submits': [], 'reaps': [], 'mems': []})
            invokes[inv]['mems'].append(ev)

    # Build per-invoke
    out: Dict[str, Any] = {'invokes': []}
    for inv in sorted(invokes.keys()):
        info = invokes[inv]
        inv_begin = info['inv_begin']
        inv_end = info['inv_end']
        subs = info['submits']
        reps = info['reaps']
        mms = info['mems']
        if not subs and not reps and inv_begin is None and inv_end is None:
            continue

        # T1/T2
        t1 = min((s['ts'] for s in subs), default=None)
        # last IN REAP
        in_reaps = [r for r in reps if r.get('dir') == 'IN']
        t2 = max((r['ts'] for r in in_reaps), default=None)

        # direction sets
        out_bufs = set(hexptr(s.get('buf')) for s in subs if s.get('dir') == 'OUT')
        in_bufs = set(hexptr(r.get('buf')) for r in reps if r.get('dir') == 'IN')

        # Host-pre: mem writes to OUT buffers before T1
        pre_bytes = 0
        pre_ms = 0.0
        if inv_begin is not None and t1 is not None:
            for m in mms:
                if inv_begin <= m['ts'] < t1:
                    dst = hexptr(m.get('dst'))
                    if dst in out_bufs:
                        pre_bytes += int(m['n'])
                        pre_ms += float(m['dt_ms'])

        # Host-post: mem reads from IN buffers after T2 until INV_END
        post_bytes = 0
        post_ms = 0.0
        if t2 is not None and inv_end is not None:
            for m in mms:
                if t2 <= m['ts'] <= inv_end:
                    src = hexptr(m.get('src'))
                    if src in in_bufs:
                        post_bytes += int(m['n'])
                        post_ms += float(m['dt_ms'])

        # per-URB inside [T1,T2]
        # Map urbp -> pair(submit, reap)
        pairs: List[Tuple[float, float, str, int]] = []
        # Build quick maps
        sub_map: Dict[str, Dict[str, Any]] = {hexptr(s.get('urbp')): s for s in subs}
        for r in reps:
            urbp = r['urbp']
            s = sub_map.get(urbp)
            if not s:
                continue
            s_ts = s['ts']
            r_ts = r['ts']
            if t1 is not None and t2 is not None and (s_ts < t1 or r_ts > t2):
                # only count URBs fully within the envelope
                continue
            dir_ = r.get('dir') or s.get('dir')
            al = int(r.get('al') or 0)
            pairs.append((s_ts, r_ts, dir_, al))

        # counts/bytes
        in_cnt = sum(1 for _, _, d, _ in pairs if d == 'IN')
        out_cnt = sum(1 for _, _, d, _ in pairs if d == 'OUT')
        in_bytes = sum(al for _, _, d, al in pairs if d == 'IN')
        out_bytes = sum(al for _, _, d, al in pairs if d == 'OUT')

        # union busy time inside envelope
        intervals = sorted([(s, e) for s, e, _, _ in pairs], key=lambda x: x[0])
        union_s = 0.0
        if intervals:
            cs, ce = intervals[0]
            for s, e in intervals[1:]:
                if s <= ce:
                    if e > ce: ce = e
                else:
                    union_s += (ce - cs); cs, ce = s, e
            union_s += (ce - cs)

        out['invokes'].append({
            'invoke': inv,
            'inv_begin': inv_begin,
            'inv_end': inv_end,
            't1_first_submit': t1,
            't2_last_in_reap': t2,
            'host_pre_bytes': pre_bytes,
            'host_pre_ms': round(pre_ms, 3),
            'host_post_bytes': post_bytes,
            'host_post_ms': round(post_ms, 3),
            'urb_in_cnt': in_cnt,
            'urb_out_cnt': out_cnt,
            'urb_in_bytes': in_bytes,
            'urb_out_bytes': out_bytes,
            'urb_union_ms': round(union_s * 1000.0, 3),
        })

    return out


def main():
    ap = argparse.ArgumentParser(description='Build per-invoke IOCTL+mem timeline from LD_PRELOAD log (BOOTTIME).')
    ap.add_argument('log', help='ldprobe jsonl (LDP_LOG)')
    ap.add_argument('-o', '--out', help='output json path (default: prints)')
    args = ap.parse_args()

    logs = load_jsonl(Path(args.log))
    res = summarize(logs)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()

