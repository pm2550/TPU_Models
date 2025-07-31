#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strace对比CPU版本: 单层vs完整模型
区分是TPU问题还是TensorFlow Lite问题
"""

import subprocess
import os

def strace_cpu_test():
    print("🔍 CPU版本strace对比测试")
    print("=" * 50)
    
    # 1. 测试单层Conv2D CPU版本
    print("\n1️⃣ 测试单层Conv2D (CPU版本):")
    cmd1 = [
        'strace', '-c', '-e', 'trace=read,write,ioctl',
        'bash', '-c', '''
source .venv/bin/activate
python3 -c "
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter('./layered models/cpu/conv2d_3x3_stride2.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
dummy_input = np.random.randint(-128, 128, input_details[0]['shape'], dtype=np.int8)

# 预热
for _ in range(3):
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

times = []
for i in range(20):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f'Conv2D CPU平均: {sum(times)/len(times):.3f} ms')
"
        '''
    ]
    
    try:
        result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        print("📊 性能结果:", result1.stdout.strip())
        print("📡 strace统计:")
        print(result1.stderr)
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 2. 测试完整MobileNet CPU版本
    print("\n2️⃣ 测试完整MobileNet (CPU版本):")
    cmd2 = [
        'strace', '-c', '-e', 'trace=read,write,ioctl',
        'bash', '-c', '''
source .venv/bin/activate  
python3 -c "
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

interpreter = Interpreter('./model/mobilenet_cpu2.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
dummy_input = np.random.randint(0, 256, input_details[0]['shape'], dtype=np.uint8)

# 预热
for _ in range(3):
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

times = []
for i in range(20):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f'MobileNet CPU平均: {sum(times)/len(times):.3f} ms')
"
        '''
    ]
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        print("📊 性能结果:", result2.stdout.strip())
        print("📡 strace统计:")
        print(result2.stderr)
    except Exception as e:
        print(f"❌ 错误: {e}")

    print("\n" + "=" * 60)
    print("🔬 对比分析总结:")
    print("=" * 60)
    print("如果CPU版本:")
    print("  - 单层 < 完整模型: 说明是TensorFlow Lite软件问题") 
    print("  - 单层 ≈ 期望值: 说明是TPU硬件特性问题")
    print("  - 相似的系统调用模式: 说明IO不是瓶颈")

if __name__ == "__main__":
    strace_cpu_test() 