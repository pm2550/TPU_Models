#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def parse_invokes(ldp_rows: List[Dict[str, Any]]) -> List[Tuple[int, float, float]]:
    begins: Dict[int, float] = {}
    ends: Dict[int, float] = {}
    for ev in ldp_rows:
        if ev.get('ev') == 'INV_BEGIN':
            begins[int(ev.get('invoke', 0))] = float(ev['ts'])
        elif ev.get('ev') == 'INV_END':
            ends[int(ev.get('invoke', 0))] = float(ev['ts'])
    out: List[Tuple[int, float, float]] = []
    for k in sorted(set(begins.keys()) | set(ends.keys())):
        b = begins.get(k); e = ends.get(k)
        if b is None or e is None:
            continue
        out.append((k, b, e))
    return out


def main():
    ap = argparse.ArgumentParser(description='Extract called symbol names inside invoke windows using LD_AUDIT logs')
    ap.add_argument('--ldp', required=True, help='ldprobe.jsonl (INV_BEGIN/INV_END)')
    ap.add_argument('--audit', required=True, help='audit.jsonl (PLT/SYMBIND events)')
    ap.add_argument('-o', '--out', help='output json')
    ap.add_argument('--libfilter', default='libedgetpu|libusb|libtflite|libtensorflowlite')
    args = ap.parse_args()

    ldp = load_jsonl(Path(args.ldp))
    au = load_jsonl(Path(args.audit))
    invs = parse_invokes(ldp)

    import re
    lib_re = re.compile(args.libfilter)

    out: Dict[str, Any] = {'invokes': []}
    for inv, b, e in invs:
        events = [x for x in au if ('ev' in x and x['ev'] in ('PLT', 'SYMBIND') and isinstance(x.get('ts'), (int, float)) and b <= float(x['ts']) <= e and lib_re.search(x.get('to') or ''))]
        # Aggregate by (sym,to)
        freq: Dict[Tuple[str, str], int] = {}
        first_ts: Dict[Tuple[str, str], float] = {}
        for ev in events:
            key = (str(ev.get('sym') or ''), str(ev.get('to') or ''))
            freq[key] = freq.get(key, 0) + 1
            ts = float(ev['ts'])
            if key not in first_ts or ts < first_ts[key]:
                first_ts[key] = ts
        items = sorted(((k[0], k[1], freq[k], first_ts[k]) for k in freq.keys()), key=lambda x: (x[3], -x[2]))
        out['invokes'].append({'invoke': inv, 'begin': b, 'end': e, 'calls': [{'sym': s, 'lib': lib, 'count': c} for s, lib, c, _ in items]})

    txt = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()

