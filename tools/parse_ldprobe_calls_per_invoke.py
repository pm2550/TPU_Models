#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open('r', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def per_invoke_calls(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Gather INV windows
    spans: List[Dict[str, Any]] = []
    current = None
    for ev in rows:
        if ev.get('ev') == 'INV_BEGIN':
            current = {'invoke': int(ev.get('invoke') or 0), 'begin': float(ev['ts']), 'end': None}
        elif ev.get('ev') == 'INV_END' and current is not None:
            current['end'] = float(ev['ts'])
            spans.append(current)
            current = None

    # Aggregate call events inside each window
    # Our call events are LIBUSB_* plus MEMCPY/MEMMOVE summary if desired.
    per: List[Dict[str, Any]] = []
    for sp in spans:
        inv = sp['invoke']
        b, e = sp['begin'], sp['end']
        # Collect LIBUSB_* events
        calls: Dict[str, int] = {}
        for ev in rows:
            t = ev.get('ts')
            if not isinstance(t, (int, float)):
                continue
            if t < b or t > e:
                continue
            name = str(ev.get('ev') or '')
            if name.startswith('LIBUSB_'):
                calls[name] = calls.get(name, 0) + 1
            # Treat USBDEVFS ioctls as function-like calls for actual I/O
            if name in ('SUBMITURB', 'REAPURB', 'REAPURBNDELAY'):
                key = f'ioctl({name})'
                calls[key] = calls.get(key, 0) + 1
        per.append({'invoke': inv, 'begin': b, 'end': e, 'calls': [{'name': k, 'count': calls[k]} for k in sorted(calls.keys())]})
    return {'invokes': per}


def main():
    ap = argparse.ArgumentParser(description='Summarize per-invoke function usage from ldprobe.jsonl (LIBUSB_* events).')
    ap.add_argument('ldp', help='ldprobe.jsonl')
    ap.add_argument('-o', '--out')
    args = ap.parse_args()
    rows = load_jsonl(Path(args.ldp))
    res = per_invoke_calls(rows)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()
