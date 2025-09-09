#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys


def main():
    if len(sys.argv) < 3:
        print('用法: python print_per_invoke_usbmon.py <out_active_union.json> <in_active_union.json>')
        sys.exit(1)
    out_path = sys.argv[1]
    in_path = sys.argv[2]
    outj = json.load(open(out_path))
    inj = json.load(open(in_path))
    po = outj.get('per_invoke', [])
    pi = inj.get('per_invoke', [])
    n = min(len(po), len(pi))
    print('idx,out_bytes,out_active_s,in_bytes,in_active_s')
    for idx in range(n):
        ob = po[idx].get('bytes_out', 0)
        oa = po[idx].get('out_active_span_s', 0.0)
        ib = pi[idx].get('bytes_in', 0)
        ia = pi[idx].get('in_active_span_s', 0.0)
        print(f"{idx},{ob},{oa},{ib},{ia}")

    # warm 摘要（跳过首个 invoke）
    po_w = po[1:]
    pi_w = pi[1:]
    out_sum = sum(x.get('bytes_out', 0) for x in po_w)
    in_sum = sum(x.get('bytes_in', 0) for x in pi_w)
    out_nz = sum(1 for x in po_w if x.get('bytes_out', 0) > 0)
    in_nz = sum(1 for x in pi_w if x.get('bytes_in', 0) > 0)
    out_avg = (out_sum / len(po_w)) if po_w else 0.0
    in_avg = (in_sum / len(pi_w)) if pi_w else 0.0
    print('--- warm summary (skip first) ---')
    print(json.dumps({
        'invokes': len(po_w),
        'out_sum': out_sum,
        'out_avg': out_avg,
        'out_nonzero': out_nz,
        'in_sum': in_sum,
        'in_avg': in_avg,
        'in_nonzero': in_nz,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()



