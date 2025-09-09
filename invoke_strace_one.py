#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import json
import signal
import tempfile
import subprocess

import numpy as np
from pycoral.utils.edgetpu import make_interpreter


def parse_strace_summary_table(path: str):
    with open(path, 'r') as f:
        s = f.read()
    stats = {}
    in_stats = False
    total_secs = None
    for line in s.splitlines():
        if line.startswith('% time') and 'seconds' in line:
            in_stats = True
            continue
        if in_stats and line.strip().startswith('------'):
            continue
        if in_stats and line.strip().startswith('total'):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    total_secs = float(parts[1])
                except Exception:
                    pass
            break
        if in_stats and line.strip():
            parts = line.split()
            if len(parts) >= 6:
                try:
                    pct = float(parts[0])
                    secs = float(parts[1])
                    calls = int(parts[3])
                    syscall = parts[5]
                    stats[syscall] = {'pct': pct, 'secs': secs, 'calls': calls}
                except Exception:
                    continue
    io_keys = ['read', 'write', 'readv', 'writev', 'pread64', 'pwrite64']
    io_secs = sum(stats[k]['secs'] for k in stats if k in io_keys)
    ioctl_secs = stats.get('ioctl', {}).get('secs', 0.0)
    ioctl_calls = stats.get('ioctl', {}).get('calls', 0)
    return {
        'total_secs': total_secs,
        'io_secs': io_secs,
        'ioctl_secs': ioctl_secs,
        'ioctl_calls': ioctl_calls,
        'raw': s,
    }


def parse_strace_window_lines(path: str, t0_epoch: float, t1_epoch: float, margin_sec: float = 0.002):
    """按绝对时间窗口筛选 -ttt 输出的系统调用，统计窗口内 IO/IOCTL。"""
    with open(path, 'r') as f:
        lines = f.readlines()
    win_start = t0_epoch - margin_sec
    win_end = t1_epoch + margin_sec
    io_keys = {'read', 'write', 'readv', 'writev', 'pread64', 'pwrite64'}
    io_calls = 0
    io_secs = 0.0
    ioctl_calls = 0
    ioctl_secs = 0.0
    captured = []
    for line in lines:
        # 形如: 1722845596.123456 ioctl(....) = ... <0.000123>
        parts = line.strip().split()
        if not parts:
            continue
        try:
            ts = float(parts[0])
        except Exception:
            continue
        if ts < win_start or ts > win_end:
            continue
        # 取 syscall 名称
        rest = line[len(parts[0]):].lstrip()
        name = rest.split('(')[0].strip()
        # 取耗时 <> 内数值（若有 -T 开启时）
        dur = 0.0
        if '<' in line and '>' in line:
            try:
                dur = float(line.rsplit('<', 1)[1].split('>')[0])
            except Exception:
                pass
        if name in io_keys:
            io_calls += 1
            io_secs += dur
        elif name == 'ioctl':
            ioctl_calls += 1
            ioctl_secs += dur
        captured.append(line.rstrip())
    return {
        'io_calls': io_calls,
        'io_secs': io_secs,
        'ioctl_calls': ioctl_calls,
        'ioctl_secs': ioctl_secs,
        'captured_lines': captured,
        'win_start': win_start,
        'win_end': win_end,
    }


def run_invoke_strace(model_path: str, warmup: int = 20, margin_sec: float = 0.002):
    it = make_interpreter(model_path)
    it.allocate_tensors()
    inp = it.get_input_details()[0]

    if inp['dtype'].__name__ == 'uint8':
        x = np.random.randint(0, 256, inp['shape'], dtype=np.uint8)
    else:
        x = np.random.randint(-128, 128, inp['shape'], dtype=np.int8)

    # Warmup not traced
    for _ in range(warmup):
        it.set_tensor(inp['index'], x)
        it.invoke()

    # Prepare strace attach (带绝对时间戳 -ttt 与时长 -T)
    pid = os.getpid()
    tmp = tempfile.NamedTemporaryFile('w', suffix='.strace', delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Set tensor BEFORE attach, so traced window覆盖仅 invoke
    it.set_tensor(inp['index'], x)

    # Attach strace to current process
    proc = subprocess.Popen([
        'strace', '-ff', '-ttt', '-T', '-p', str(pid),
        '-e', 'trace=ioctl,read,write,readv,writev,pread64,pwrite64',
        '-o', tmp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    time.sleep(0.05)  # give strace time to attach

    t0_perf = time.perf_counter()
    t0_epoch = time.time()
    it.invoke()
    t1_perf = time.perf_counter()
    t1_epoch = time.time()

    # Detach strace
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=2)
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=1)
        except Exception:
            pass

    # 解析窗口内的事件
    window = parse_strace_window_lines(tmp_path, t0_epoch, t1_epoch, margin_sec=margin_sec)
    # 也可解析表格总览（备用）
    # table = parse_strace_summary_table(tmp_path)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

    return {
        'invoke_ms': (t1_perf - t0_perf) * 1000.0,
        'win_margin_sec': margin_sec,
        'io_secs': window['io_secs'],
        'ioctl_calls': window['ioctl_calls'],
        'ioctl_secs': window['ioctl_secs'],
        'window': {'start': window['win_start'], 'end': window['win_end']},
        # 'captured': window['captured_lines'],  # 如需详细行，可开启
    }


def main():
    if len(sys.argv) < 2:
        print('用法: python invoke_strace_one.py <model_path> [warmup] [margin_ms]')
        sys.exit(1)
    model = sys.argv[1]
    warm = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    margin_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    res = run_invoke_strace(model, warmup=warm, margin_sec=margin_ms/1000.0)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == '__main__':
    main()


