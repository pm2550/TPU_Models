#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from strace_tpu_analysis import run_strace_tpu_analysis


def main():
    if len(sys.argv) < 3:
        print("用法: python run_strace_one.py <model_path> <name> [num_infer]")
        sys.exit(1)
    path = sys.argv[1]
    name = sys.argv[2]
    num = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    if not os.path.exists(path):
        print(f"❌ 模型不存在: {path}")
        sys.exit(1)
    ana = run_strace_tpu_analysis(path, name, num_inferences=num)
    if not ana:
        print("❌ 分析失败")
        sys.exit(2)
    io_ms = ana.get('io_time', 0.0) * 1000.0
    ioctl_calls = ana.get('tpu_calls', 0)
    ioctl_ms = ana.get('syscall_stats', {}).get('ioctl', {}).get('seconds', 0.0) * 1000.0
    avg_ms = ana.get('inference_time', 0.0)
    print("\n=== 单次结果 ===")
    print(f"【{name}】 avg_infer={avg_ms:.3f} ms | IO总时长={io_ms:.3f} ms | ioctl: {ioctl_calls} 次, {ioctl_ms:.3f} ms")


if __name__ == "__main__":
    main()



