#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str], check: bool = True, capture_output: bool = False, text: bool = True, input_text: str | None = None):
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=text, input=input_text)


def start_usbmon_capture(bus: str, out_path: Path, pw_file: Path) -> subprocess.Popen:
    run(["sudo", "-S", "-p", "", "modprobe", "usbmon"], input_text=pw_file.read_text())
    # truncate file and start capture (root)
    run(["sudo", "-S", "-p", "", "sh", "-c", f": > '{out_path}'"], input_text=pw_file.read_text())
    node = f"/sys/kernel/debug/usb/usbmon/{bus}u"
    proc = subprocess.Popen(["sudo", "-S", "-p", "", "sh", "-c", f"cat '{node}' >> '{out_path}'"], stdin=subprocess.PIPE, text=True)
    # feed password once to sudo -S
    assert proc.stdin is not None
    proc.stdin.write(pw_file.read_text())
    proc.stdin.flush()
    time.sleep(0.1)
    return proc


def build_time_map(usb_txt: Path, tm_path: Path):
    deadline = time.time() + 10.0
    usb_ts = None
    while time.time() < deadline and usb_ts is None:
        try:
            with usb_txt.open('r', errors='ignore') as f:
                for ln in f:
                    parts = ln.split()
                    if len(parts) >= 2:
                        try:
                            v = float(parts[1])
                            usb_ts = (v / 1e6) if v > 1e6 else v
                            break
                        except Exception:
                            continue
        except FileNotFoundError:
            pass
        if usb_ts is None:
            time.sleep(0.02)
    if usb_ts is None:
        tm_path.write_text(json.dumps({"usbmon_ref": None, "boottime_ref": None}))
        return False
    try:
        bt_ref = time.clock_gettime(time.CLOCK_BOOTTIME)
    except Exception:
        bt_ref = float(open('/proc/uptime').read().split()[0])
    tm_path.write_text(json.dumps({"usbmon_ref": usb_ts, "boottime_ref": bt_ref}))
    return True


def run_invoke_windows(model_path: Path, count: int, iv_path: Path):
    code = f"""
import json, time, numpy as np
m=r"{model_path}"
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception as e:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    except Exception as e2:
        print('INTERPRETER_FAIL', repr(e), repr(e2))
        raise SystemExit(1)
it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
for _ in range(20):
    it.set_tensor(inp['index'], x); it.invoke()
spans=[]
for i in range({count}):
    it.set_tensor(inp['index'], x)
    t0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.invoke()
    t1=time.clock_gettime(time.CLOCK_BOOTTIME)
    spans.append({{'begin': t0, 'end': t1}})
open(r"{iv_path}", 'w').write(json.dumps({{'name':'invoke_only','spans':spans}}))
"""
    run([sys.executable, "-c", code])


def chown_rw(pw_file: Path, *paths: Path):
    p = pw_file.read_text()
    for path in paths:
        run(["sudo", "-S", "-p", "", "chown", f"{os.getuid()}:{os.getgid()}", str(path)], input_text=p)
        run(["sudo", "-S", "-p", "", "chmod", "0644", str(path)], input_text=p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bus', default='0', help='usbmon bus number, 0 means 0u (all)')
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--pre_s', type=float, default=1.0)
    ap.add_argument('--post_s', type=float, default=1.0)
    ap.add_argument('--expand_s', type=float, default=2.0)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    usb_txt = out_dir / 'usbmon.txt'
    tm_json = out_dir / 'time_map.json'
    iv_json = out_dir / 'invokes.json'
    pw_file = Path('/home/10210/Desktop/OS/password.text')

    # start capture
    cap = start_usbmon_capture(args.bus, usb_txt, pw_file)
    try:
        time.sleep(max(args.pre_s, 0.0))
        ok = build_time_map(usb_txt, tm_json)
        # run invoke-only windows
        run_invoke_windows(Path(args.model), args.count, iv_json)
        time.sleep(max(args.post_s, 0.0))
    finally:
        try:
            cap.send_signal(signal.SIGTERM)
            time.sleep(0.1)
        except Exception:
            pass

    # fix perms
    chown_rw(pw_file, usb_txt, tm_json, iv_json)

    # per-invoke intersection stats
    stats = run([sys.executable, str(Path(__file__).with_name('per_invoke_intersection_stats.py')), str(out_dir), str(args.expand_s)], capture_output=True)
    print(stats.stdout)


if __name__ == '__main__':
    main()

