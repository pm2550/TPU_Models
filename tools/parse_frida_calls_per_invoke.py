#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def parse(frida_txt: Path) -> Dict[str, Any]:
    begin_pat = re.compile(r"\blibldprobe_gate\.so!ldprobe_begin_invoke\b")
    end_pat = re.compile(r"\blibldprobe_gate\.so!ldprobe_end_invoke\b")
    call_pat = re.compile(r"\b([\w\-\.]+\.so(?:\.[^!\s]+)?)!([\w@.]+)\b")
    # focus libs
    allow_lib = re.compile(r"libedgetpu|libusb|libtflite|libtensorflowlite")

    inv = 0
    open_window = False
    per: Dict[int, Dict[str, int]] = {}

    with frida_txt.open('r', errors='ignore') as f:
        for ln in f:
            if begin_pat.search(ln):
                inv += 1
                open_window = True
                per.setdefault(inv, {})
                continue
            if end_pat.search(ln):
                open_window = False
                continue
            if not open_window:
                continue
            m = call_pat.search(ln)
            if not m:
                continue
            lib = m.group(1)
            sym = m.group(2)
            if not allow_lib.search(lib):
                continue
            key = f"{lib}:{sym}"
            per[inv][key] = per[inv].get(key, 0) + 1

    out = {"invokes": []}
    for k in sorted(per.keys()):
        items = sorted(per[k].items(), key=lambda kv: (-kv[1], kv[0]))
        out["invokes"].append({
            "invoke": k,
            "calls": [{"lib_sym": libsym, "count": cnt} for libsym, cnt in items]
        })
    return out


def main():
    ap = argparse.ArgumentParser(description="Parse frida-trace output into per-invoke calls using ldprobe begin/end markers.")
    ap.add_argument('frida_txt')
    ap.add_argument('-o', '--out')
    args = ap.parse_args()
    res = parse(Path(args.frida_txt))
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()

