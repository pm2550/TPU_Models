#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys


def mbps(total_bytes: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (total_bytes / (1024.0 * 1024.0)) / seconds


def summarize_warm(out_path: str, in_path: str) -> dict:
    outj = json.load(open(out_path))
    inj = json.load(open(in_path))
    out_list = outj.get('per_invoke', [])
    in_list = inj.get('per_invoke', [])
    if not out_list or not in_list:
        return {}
    # 跳过首个 invoke
    out_warm = out_list[1:]
    in_warm = in_list[1:]

    out_bytes = sum(x.get('bytes_out', 0) for x in out_warm)
    out_active = sum(x.get('out_active_span_s', 0.0) for x in out_warm)
    in_bytes = sum(x.get('bytes_in', 0) for x in in_warm)
    in_active = sum(x.get('in_active_span_s', 0.0) for x in in_warm)

    # 平均每次
    n = len(out_warm)
    m = len(in_warm)
    out_avg_bytes = (out_bytes / n) if n else 0
    in_avg_bytes = (in_bytes / m) if m else 0
    out_avg_active = (out_active / n) if n else 0.0
    in_avg_active = (in_active / m) if m else 0.0

    res = {
        'invokes': n,
        'out_total_bytes': out_bytes,
        'out_total_active_s': out_active,
        'out_overall_MBps': mbps(out_bytes, out_active),
        'out_avg_bytes': out_avg_bytes,
        'out_avg_active_s': out_avg_active,
        'out_avg_MBps': mbps(out_avg_bytes, out_avg_active) if out_avg_active > 0 else 0.0,
        'in_total_bytes': in_bytes,
        'in_total_active_s': in_active,
        'in_overall_MBps': mbps(in_bytes, in_active),
        'in_avg_bytes': in_avg_bytes,
        'in_avg_active_s': in_avg_active,
        'in_avg_MBps': mbps(in_avg_bytes, in_avg_active) if in_avg_active > 0 else 0.0,
    }
    return res


def main():
    if len(sys.argv) < 3:
        print('用法: python summarize_warm_usbmon.py <out_active_union.json> <in_active_union.json>')
        sys.exit(1)
    out_path = sys.argv[1]
    in_path = sys.argv[2]
    res = summarize_warm(out_path, in_path)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


