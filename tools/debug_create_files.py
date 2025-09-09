#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict | None = None):
    return subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bus', required=True)
    ap.add_argument('--duration', type=int, required=True)
    ap.add_argument('--count', type=int, default=10)
    args = ap.parse_args()

    model = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print('[whoami]', run(['whoami']).stdout.strip())
    print('[pwd]', Path.cwd())
    print('[model_exists]', model.exists(), model)
    print('[outdir]', out, 'writable=', os.access(out, os.W_OK))

    env = os.environ.copy()
    env['COUNT'] = str(args.count)
    env['LEAD_S'] = '1'

    script = '/home/10210/Desktop/OS/run_usbmon_capture_offline.sh'
    cmd = [script, str(model), model.stem, str(out), str(args.bus), str(args.duration)]
    print('[exec]', cmd)
    res = run(cmd, env=env)
    print('[returncode]', res.returncode)
    print('[stdout]\n', res.stdout)
    print('[stderr]\n', res.stderr)

    if out.exists():
        print('[outdir ls]')
        for p in sorted(out.iterdir()):
            try:
                print(p.name, p.stat().st_size)
            except Exception as e:
                print(p.name, 'stat_error', e)

        # quick grep counts
        usb_txt = out / 'usbmon.txt'
        if usb_txt.exists():
            cnt_cmd = f"grep -c ' Bo:' '{usb_txt}' || true; grep -c ' Bi:' '{usb_txt}' || true; grep -c ' Co:' '{usb_txt}' || true; grep -c ' Ci:' '{usb_txt}' || true"
            proc = subprocess.run(['bash', '-lc', cnt_cmd], text=True, capture_output=True)
            print('[counts]\n' + proc.stdout)


if __name__ == '__main__':
    main()



