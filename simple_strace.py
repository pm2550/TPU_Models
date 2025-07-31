#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单strace分析TPU IO开销
"""

import subprocess
import tempfile
import os

def simple_strace_test():
    print("🔍 简单strace测试")
    
    # 测试单层模型
    print("\n1. 测试单层Conv2D:")
    cmd1 = [
        'strace', '-c', '-e', 'trace=read,write,ioctl',
        'bash', '-c', '''
source .venv/bin/activate
python3 -c "
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

interpreter = make_interpreter('./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
dummy_input = np.random.randint(-128, 128, input_details[0]['shape'], dtype=np.int8)

times = []
for i in range(10):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f'Conv2D平均: {sum(times)/len(times):.3f} ms')
"
        '''
    ]
    
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        print("输出:", result1.stdout)
        print("strace:", result1.stderr)
    except Exception as e:
        print(f"错误: {e}")
    
    print("\n2. 测试完整MobileNet:")
    cmd2 = [
        'strace', '-c', '-e', 'trace=read,write,ioctl',
        'bash', '-c', '''
source .venv/bin/activate  
python3 -c "
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

interpreter = make_interpreter('./model/mobilenet.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
dummy_input = np.random.randint(0, 256, input_details[0]['shape'], dtype=np.uint8)

times = []
for i in range(10):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f'MobileNet平均: {sum(times)/len(times):.3f} ms')
"
        '''
    ]
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        print("输出:", result2.stdout)
        print("strace:", result2.stderr)
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    simple_strace_test() 