#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from strace_tpu_analysis import run_strace_tpu_analysis


def main():
    models = [
        ("./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite", "单层Conv2D"),
        ("./layered models/mn/tpu/mnv2_224_layer1_tiny_int8_edgetpu.tflite", "tiny第一层"),
    ]

    results = []
    for path, name in models:
        if not os.path.exists(path):
            print(f"❌ 模型不存在: {path}")
            continue
        ana = run_strace_tpu_analysis(path, name, num_inferences=30)
        if not ana:
            print(f"❌ 分析失败: {name}")
            continue
        results.append((name, ana))

    if not results:
        return

    print("\n=== 结果汇总 ===")
    for name, ana in results:
        io_ms = ana.get('io_time', 0.0) * 1000.0
        ioctl_calls = ana.get('tpu_calls', 0)
        ioctl_ms = ana.get('syscall_stats', {}).get('ioctl', {}).get('seconds', 0.0) * 1000.0
        avg_ms = ana.get('inference_time', 0.0)
        print(f"【{name}】 avg_infer={avg_ms:.3f} ms | IO总时长={io_ms:.3f} ms | ioctl: {ioctl_calls} 次, {ioctl_ms:.3f} ms")


if __name__ == "__main__":
    main()



