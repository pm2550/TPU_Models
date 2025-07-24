#!/usr/bin/env python3
# one_interrupt_per_run_fix.py
# 每轮恰好随机 1 次 small 插队；small 一次成功，无尺寸错误。

import time, random, sys, numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

RUNS = 10
seg_paths  = [f'./segment_models/seg{i}.tflite' for i in range(4)]
small_path = './segment_models/small.tflite'

# ─── 1. 缓存 Interpreter ───
seg_itp   = [make_interpreter(p) for p in seg_paths]
small_itp = make_interpreter(small_path)
for itp in seg_itp + [small_itp]:
    itp.allocate_tensors()

def invoke(itp, arr):
    common.set_input(itp, arr)
    t0 = time.perf_counter()
    itp.invoke()
    dt = time.perf_counter() - t0
    return common.output_tensor(itp, 0).copy(), dt * 1000  # ms

def run_seg(itp_seg, x, need_small, dummy_small):
    small_ms = 0.0
    if need_small:
        while True:               # 必须成功一次
            try:
                _, t = invoke(small_itp, dummy_small)
                small_ms = t
                break
            except Exception as e:
                print("small failed → retry:", e)
                time.sleep(0.005)

    out, seg_ms = invoke(itp_seg, x)
    return out, seg_ms, small_ms

# ─── 2. 主流程 ───
hdr = f"{'Run':>3} | {'seg(ms)':>8} | {'small(ms)':>9} | {'total(ms)':>9}"
print(hdr); print('-'*len(hdr))

for r in range(RUNS):
    first_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    x = first_img
    target = random.randint(0, 3)          # 本轮插队段
    seg_sum = small_sum = 0.0

    for idx, itp in enumerate(seg_itp):
        need_small = (idx == target)
        x, seg_t, small_t = run_seg(itp, x, need_small, first_img)
        seg_sum   += seg_t
        small_sum += small_t

    print(f"{r:>3} | {seg_sum:8.1f} | {small_sum:9.1f} |"
          f"{seg_sum+small_sum:9.1f}")
