import time, random, sys, numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

RUNS = 10
large_path = './onlyCo/large.tflite'
small_path = './onlyCo/small.tflite'

# 1. 缓存 Interpreter
large_itp = make_interpreter(large_path)
small_itp = make_interpreter(small_path)
for itp in [large_itp, small_itp]:
    itp.allocate_tensors()

def invoke(itp, arr):
    common.set_input(itp, arr)
    t0 = time.perf_counter()
    itp.invoke()
    dt = time.perf_counter() - t0
    return common.output_tensor(itp, 0).copy(), dt * 1000  # ms

def run_large(itp_large, x, need_small, dummy_small):
    small_ms = 0.0
    if need_small:
        while True:
            try:
                _, t = invoke(small_itp, dummy_small)
                small_ms = t
                break
            except Exception as e:
                print("small failed → retry:", e)
                time.sleep(0.005)
    out, large_ms = invoke(itp_large, x)
    return out, large_ms, small_ms

# 2. 主流程
hdr = f"{'Run':>3} | {'large(ms)':>10} | {'small(ms)':>9} | {'total(ms)':>9}"
print(hdr); print('-'*len(hdr))

for r in range(RUNS):
    first_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
    x = first_img
    need_small = True  # 每轮恰好插队一次
    x, large_t, small_t = run_large(large_itp, x, need_small, first_img)
    print(f"{r:>3} | {large_t:10.1f} | {small_t:9.1f} | {large_t+small_t:9.1f}")
