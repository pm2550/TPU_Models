#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import List, Tuple, Dict, Any


def choose_python() -> str:
    venv_py = '/home/10210/Desktop/OS/.venv/bin/python'
    if os.path.exists(venv_py) and os.access(venv_py, os.X_OK):
        return venv_py
    return sys.executable


def start_usbmon_capture(bus: str, password_file: str) -> subprocess.Popen:
    usb_node = f"/sys/kernel/debug/usb/usbmon/{bus}u"
    pw = open(password_file, 'r').read()
    # Ensure usbmon module
    subprocess.run(['sudo', '-S', '-p', '', 'modprobe', 'usbmon'], input=pw, text=True, check=False)
    # Start capture
    proc = subprocess.Popen(['sudo', '-S', '-p', '', 'sh', '-c', f"cat '{usb_node}'"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdin is not None
    proc.stdin.write(pw)
    proc.stdin.flush()
    return proc


def read_usbmon(proc: subprocess.Popen, lines_out: List[str], refs_out: Dict[str, float], stop_event: threading.Event) -> None:
    usb_ref = None
    bt_ref = None
    for ln in iter(proc.stdout.readline, ''):
        if stop_event.is_set():
            break
        lines_out.append(ln)
        if usb_ref is None:
            parts = ln.split()
            if len(parts) >= 2:
                try:
                    v = float(parts[1])
                    usb_ref = (v / 1e6) if v > 1e6 else v
                    try:
                        bt_ref = time.clock_gettime(time.CLOCK_BOOTTIME)
                    except Exception:
                        bt_ref = float(open('/proc/uptime').read().split()[0])
                    refs_out['usbmon_ref'] = usb_ref
                    refs_out['boottime_ref'] = bt_ref
                except Exception:
                    pass
    # drain remaining
    try:
        proc.stdout.close()
    except Exception:
        pass


def run_invokes(model_path: str, count: int) -> List[Dict[str, float]]:
    code = f"""
import json, time, numpy as np
m=r"{model_path}"
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    except Exception as e2:
        print('INTERPRETER_FAIL', repr(e2)); raise SystemExit(1)
it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
for _ in range(5):
    it.set_tensor(inp['index'], x); it.invoke()
spans=[]
for i in range({count}):
    it.set_tensor(inp['index'], x)
    t0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.invoke()
    t1=time.clock_gettime(time.CLOCK_BOOTTIME)
    spans.append({{'begin': t0, 'end': t1}})
print(json.dumps(spans))
"""
    pybin = choose_python()
    res = subprocess.run([pybin, '-c', code], text=True, capture_output=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise SystemExit(1)
    return json.loads(res.stdout.strip())


def parse_urbs(usb_lines: List[str], dirs: Tuple[str, str]) -> List[Tuple[float, float, int, str]]:
    sub_dir, com_dir = dirs
    pending: Dict[str, Tuple[float, str]] = {}
    finished: List[Tuple[float, float, int, str]] = []
    re_dir = re.compile(r"([CB][io]):\d+:\d+:\d+")
    for ln in usb_lines:
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
        if dir_tok not in (sub_dir, com_dir):
            continue
        if sc == 'S':
            pending[tag] = (ts, dir_tok)
        elif sc == 'C':
            start = None
            if tag in pending:
                s, d = pending.pop(tag)
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
                        dir_idx = i; break
                if dir_idx is not None and dir_idx + 2 < len(parts):
                    try:
                        nbytes = int(parts[dir_idx+2])
                    except Exception:
                        nbytes = 0
                if nbytes == 0:
                    m2 = re.search(r"#\s*(\d+)", ln)
                    if m2:
                        nbytes = int(m2.group(1))
            if start is not None:
                finished.append((start, ts, nbytes, dir_tok))
    return finished


def per_invoke_intersection(urbs: List[Tuple[float, float, int, str]], spans: List[Dict[str, float]], usbmon_ref: float, boottime_ref: float, expand_s: float) -> List[Dict[str, Any]]:
    results = []
    for w in spans:
        b_bt = w['begin']; e_bt = w['end']
        b = b_bt - boottime_ref + usbmon_ref
        e = e_bt - boottime_ref + usbmon_ref
        b -= expand_s; e += expand_s
        window_urbs = [u for u in urbs if (u[0] < e and u[1] > b)]
        intervals: List[Tuple[float, float]] = []
        bytes_sum = 0
        for s, t, nb, _ in window_urbs:
            ss = s if s > b else b
            tt = t if t < e else e
            if tt > ss:
                intervals.append((ss, tt))
            bytes_sum += nb
        active = 0.0
        if intervals:
            intervals.sort(key=lambda x: x[0])
            cs, ce = intervals[0]
            for ss, tt in intervals[1:]:
                if ss <= ce:
                    if tt > ce:
                        ce = tt
                else:
                    active += (ce - cs)
                    cs, ce = ss, tt
            active += (ce - cs)
        results.append({'bytes': bytes_sum, 'active_s': active})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--bus', default='0')
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--pre_s', type=float, default=1.0)
    ap.add_argument('--post_s', type=float, default=1.0)
    ap.add_argument('--expand_s', type=float, default=2.0)
    args = ap.parse_args()

    pw_file = '/home/10210/Desktop/OS/password.text'

    # 1) start usbmon
    proc = start_usbmon_capture(args.bus, pw_file)
    lines: List[str] = []
    refs: Dict[str, float] = {}
    stop = threading.Event()
    t = threading.Thread(target=read_usbmon, args=(proc, lines, refs, stop), daemon=True)
    t.start()

    time.sleep(max(0.0, args.pre_s))

    # 2) run invokes to get BOOTTIME windows in memory
    spans = run_invokes(args.model, args.count)

    time.sleep(max(0.0, args.post_s))

    # 3) stop capture
    stop.set()
    try:
        proc.send_signal(signal.SIGTERM)
    except Exception:
        pass
    try:
        proc.wait(timeout=1.0)
    except Exception:
        pass

    usb_ref = refs.get('usbmon_ref'); bt_ref = refs.get('boottime_ref')
    print(json.dumps({'usbmon_ref': usb_ref, 'boottime_ref': bt_ref, 'usbmon_lines': len(lines), 'invokes': len(spans)}))

    # 4) parse and compute per-invoke intersection
    out_urbs = parse_urbs(lines, ('Bo','Co'))
    in_urbs = parse_urbs(lines, ('Bi','Ci'))
    print(json.dumps({'Bo': sum(1 for _ in filter(lambda u: u[3]=='Co', out_urbs)), 'Bi': sum(1 for _ in filter(lambda u: u[3]=='Ci', in_urbs))}))

    if not lines or usb_ref is None or bt_ref is None:
        print('no_capture_or_no_refs')
        return

    out_stats = per_invoke_intersection(out_urbs, spans, usb_ref, bt_ref, args.expand_s)
    in_stats = per_invoke_intersection(in_urbs, spans, usb_ref, bt_ref, args.expand_s)
    print('idx,out_bytes,out_active_s,in_bytes,in_active_s')
    for i in range(min(len(out_stats), len(in_stats))):
        ob = out_stats[i]['bytes']; oa = out_stats[i]['active_s']
        ib = in_stats[i]['bytes']; ia = in_stats[i]['active_s']
        print(f"{i},{ob},{oa},{ib},{ia}")


if __name__ == '__main__':
    main()



