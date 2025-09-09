#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import numpy as np
import sys
from pycoral.utils.edgetpu import make_interpreter


def measure_invoke_only(interpreter, input_tensor):
    # 仅测 invoke，用 set_tensor 在外侧
    t0 = time.perf_counter()
    interpreter.invoke()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0  # ms


def run(model_path: str, repeats: int = 10, prewarm: int = 50):
    out = {
        'model': model_path,
        'cold_ms': [],
        'warm_ms': [],
    }

    # 冷启动：每次新建解释器
    for _ in range(repeats):
        interpreter = make_interpreter(model_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        x = (np.random.randint(0, 256, size=inp['shape'], dtype=np.uint8)
             if inp['dtype'].__name__ == 'uint8' else
             np.random.randint(-128, 128, size=inp['shape'], dtype=np.int8))
        interpreter.set_tensor(inp['index'], x)
        out['cold_ms'].append(measure_invoke_only(interpreter, inp))

    # 热启动：单次创建解释器，多次 invoke
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    x = (np.random.randint(0, 256, size=inp['shape'], dtype=np.uint8)
         if inp['dtype'].__name__ == 'uint8' else
         np.random.randint(-128, 128, size=inp['shape'], dtype=np.int8))
    # 预热
    for _ in range(prewarm):
        interpreter.set_tensor(inp['index'], x)
        interpreter.invoke()
    # 记录
    for _ in range(repeats):
        interpreter.set_tensor(inp['index'], x)
        out['warm_ms'].append(measure_invoke_only(interpreter, inp))

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'model.tflite'
    run(model)




