#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict | None = None):
    return subprocess.run(cmd, check=True, text=True, capture_output=True, env=env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bus', default='0', help='usbmon bus number; 0 means 0u (all buses)')
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--duration', type=int, default=12)
    ap.add_argument('--lead_s', type=float, default=1.0)
    ap.add_argument('--expand_s', type=float, default=2.0)
    ap.add_argument('--print_lines', type=int, default=80)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env['COUNT'] = str(args.count)
    env['LEAD_S'] = str(args.lead_s)

    # 1) 调用现有 shell 脚本采集（仅 invoke 窗口）
    cmd = [
        '/home/10210/Desktop/OS/run_usbmon_capture_offline.sh',
        args.model,
        Path(args.model).stem,
        str(out_dir),
        str(args.bus),
        str(args.duration),
    ]
    cap_res = run(cmd, env=env)
    sys.stdout.write(cap_res.stdout)
    sys.stderr.write(cap_res.stderr)

    # 2) 逐次统计（交集口径 + 扩窗）
    stat_cmd = [
        sys.executable,
        '/home/10210/Desktop/OS/tools/per_invoke_intersection_stats.py',
        str(out_dir),
        str(args.expand_s),
    ]
    stat_res = run(stat_cmd)
    lines = stat_res.stdout.splitlines()
    for ln in lines[: args.print_lines]:
        print(ln)


if __name__ == '__main__':
    main()

