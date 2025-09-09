#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNet-V2时间轴分析 - 第一层 vs 完整模型
"""

import subprocess
import tempfile
import os
import re
from datetime import datetime

def analyze_mobilenetv2_models():
    print("🔬 MobileNet-V2 时间轴详细分析")
    print("=" * 80)
    print("对比模型:")
    print("  单层: ./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite")
    print("  完整: ./model/mobilenet.tflite")
    print()
    
    # 1. 分析单层Conv2D (MobileNet-V2第一层)
    print("1️⃣ 单层Conv2D (MobileNet-V2第一层) 时间轴:")
    print("-" * 60)
    
    test_script_single = '''
source .venv/bin/activate
python3 -c "
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

print('=== MobileNet-V2 第一层测试 ===')
print('T0: 程序开始')

print('T1: 开始加载模型...')
start_load = time.perf_counter()
interpreter = make_interpreter('./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite')
end_load = time.perf_counter()
load_time = (end_load - start_load) * 1000
print(f'T2: 模型加载完成 - 耗时 {load_time:.3f}ms')

print('T3: 开始分配张量...')
start_alloc = time.perf_counter()
interpreter.allocate_tensors()
end_alloc = time.perf_counter()
alloc_time = (end_alloc - start_alloc) * 1000
print(f'T4: 张量分配完成 - 耗时 {alloc_time:.3f}ms')

input_details = interpreter.get_input_details()
print('输入形状:', input_details[0]['shape'])
print('输入类型:', input_details[0]['dtype'])

dummy_input = np.random.randint(-128, 128, input_details[0]['shape'], dtype=np.int8)

print('T5: 第一次推理 (冷启动)...')
start_1st = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
_ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end_1st = time.perf_counter()
first_time = (end_1st - start_1st) * 1000
print(f'T6: 第一次推理完成 - 耗时 {first_time:.3f}ms')

print('T7: 第二次推理 (热启动)...')
start_2nd = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
_ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end_2nd = time.perf_counter()
second_time = (end_2nd - start_2nd) * 1000
print(f'T8: 第二次推理完成 - 耗时 {second_time:.3f}ms')

print('T9: 第三次推理...')
start_3rd = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
_ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end_3rd = time.perf_counter()
third_time = (end_3rd - start_3rd) * 1000
print(f'T10: 第三次推理完成 - 耗时 {third_time:.3f}ms')

total_time = (end_3rd - start_load) * 1000
overhead_time = total_time - first_time - second_time - third_time
print(f'\\n📊 单层模型分析:')
print(f'   模型加载: {load_time:.3f}ms')
print(f'   张量分配: {alloc_time:.3f}ms')
print(f'   第1次推理: {first_time:.3f}ms (冷启动)')
print(f'   第2次推理: {second_time:.3f}ms (热启动)')
print(f'   第3次推理: {third_time:.3f}ms (稳定状态)')
print(f'   其他开销: {overhead_time:.3f}ms')
print(f'   总耗时: {total_time:.3f}ms')
"
    '''
    
    try:
        result1 = subprocess.run(['bash', '-c', test_script_single], 
                                capture_output=True, text=True, timeout=30)
        print("📊 单层模型执行结果:")
        for line in result1.stdout.strip().split('\n'):
            print(f"   {line}")
        
        if result1.stderr:
            print(f"   错误: {result1.stderr.strip()}")
    except Exception as e:
        print(f"❌ 单层模型测试失败: {e}")
    
    print("\n" + "="*80)
    
    # 2. 分析完整MobileNet
    print("2️⃣ 完整MobileNet-V2时间轴:")
    print("-" * 60)
    
    test_script_full = '''
source .venv/bin/activate
python3 -c "
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

print('=== 完整MobileNet-V2测试 ===')
print('T0: 程序开始')

print('T1: 开始加载完整模型...')
start_load = time.perf_counter()
interpreter = make_interpreter('./model/mobilenet.tflite')
end_load = time.perf_counter()
load_time = (end_load - start_load) * 1000
print(f'T2: 完整模型加载完成 - 耗时 {load_time:.3f}ms')

print('T3: 开始分配张量...')
start_alloc = time.perf_counter()
interpreter.allocate_tensors()
end_alloc = time.perf_counter()
alloc_time = (end_alloc - start_alloc) * 1000
print(f'T4: 张量分配完成 - 耗时 {alloc_time:.3f}ms')

input_details = interpreter.get_input_details()
print('输入形状:', input_details[0]['shape'])
print('输入类型:', input_details[0]['dtype'])

dummy_input = np.random.randint(0, 256, input_details[0]['shape'], dtype=np.uint8)

print('T5: 第一次推理 (冷启动)...')
start_1st = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
_ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end_1st = time.perf_counter()
first_time = (end_1st - start_1st) * 1000
print(f'T6: 第一次推理完成 - 耗时 {first_time:.3f}ms')

print('T7: 第二次推理 (热启动)...')
start_2nd = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
_ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end_2nd = time.perf_counter()
second_time = (end_2nd - start_2nd) * 1000
print(f'T8: 第二次推理完成 - 耗时 {second_time:.3f}ms')

print('T9: 第三次推理...')
start_3rd = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
_ = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
end_3rd = time.perf_counter()
third_time = (end_3rd - start_3rd) * 1000
print(f'T10: 第三次推理完成 - 耗时 {third_time:.3f}ms')

total_time = (end_3rd - start_load) * 1000
overhead_time = total_time - first_time - second_time - third_time
print(f'\\n📊 完整模型分析:')
print(f'   模型加载: {load_time:.3f}ms')
print(f'   张量分配: {alloc_time:.3f}ms')
print(f'   第1次推理: {first_time:.3f}ms (冷启动)')
print(f'   第2次推理: {second_time:.3f}ms (热启动)')
print(f'   第3次推理: {third_time:.3f}ms (稳定状态)')
print(f'   其他开销: {overhead_time:.3f}ms')
print(f'   总耗时: {total_time:.3f}ms')
"
    '''
    
    try:
        result2 = subprocess.run(['bash', '-c', test_script_full], 
                                capture_output=True, text=True, timeout=30)
        print("📊 完整模型执行结果:")
        for line in result2.stdout.strip().split('\n'):
            print(f"   {line}")
        
        if result2.stderr:
            print(f"   错误: {result2.stderr.strip()}")
    except Exception as e:
        print(f"❌ 完整模型测试失败: {e}")
    
    # 3. 对比分析
    print("\n" + "="*80)
    print("3️⃣ 对比分析总结:")
    print("-" * 60)
    
    # 分析结果的关键数据提取
    single_output = result1.stdout if 'result1' in locals() else ""
    full_output = result2.stdout if 'result2' in locals() else ""
    
    # 提取关键数据
    def extract_times(output):
        times = {}
        for line in output.split('\n'):
            if '模型加载完成' in line:
                times['load'] = float(re.search(r'([\d.]+)ms', line).group(1))
            elif '张量分配完成' in line:
                times['alloc'] = float(re.search(r'([\d.]+)ms', line).group(1))
            elif '第1次推理完成' in line:
                times['first'] = float(re.search(r'([\d.]+)ms', line).group(1))
            elif '第2次推理完成' in line:
                times['second'] = float(re.search(r'([\d.]+)ms', line).group(1))
            elif '第3次推理完成' in line:
                times['third'] = float(re.search(r'([\d.]+)ms', line).group(1))
        return times
    
    try:
        single_times = extract_times(single_output)
        full_times = extract_times(full_output)
        
        print("📈 详细对比 (单层 vs 完整):")
        print(f"{'阶段':<15} | {'单层(ms)':<10} | {'完整(ms)':<10} | {'差异':<15}")
        print("-" * 60)
        
        for stage in ['load', 'alloc', 'first', 'second', 'third']:
            if stage in single_times and stage in full_times:
                single_val = single_times[stage]
                full_val = full_times[stage]
                diff = single_val - full_val
                ratio = single_val / full_val if full_val > 0 else 0
                
                stage_names = {
                    'load': '模型加载',
                    'alloc': '张量分配',
                    'first': '第1次推理',
                    'second': '第2次推理',
                    'third': '第3次推理'
                }
                
                print(f"{stage_names[stage]:<15} | {single_val:<10.3f} | {full_val:<10.3f} | {diff:+.3f} ({ratio:.2f}x)")
        
        print(f"\n💡 关键发现:")
        if 'second' in single_times and 'second' in full_times:
            single_stable = single_times['second']
            full_stable = full_times['second']
            ratio = single_stable / full_stable
            
            print(f"   • 稳定状态推理时间: 单层 {single_stable:.3f}ms vs 完整 {full_stable:.3f}ms")
            print(f"   • 性能比率: {ratio:.2f}x (单层/完整)")
            
            if ratio > 2:
                print(f"   🔴 单层模型明显慢于完整模型!")
                print(f"   🔍 主要瓶颈可能是:")
                print(f"      - TPU固定初始化开销分摊不足")
                print(f"      - 小计算量无法充分利用TPU并行能力")
                print(f"      - EdgeTPU对大模型优化更好")
    
    except Exception as e:
        print(f"❌ 对比分析失败: {e}")

def main():
    analyze_mobilenetv2_models()

if __name__ == "__main__":
    main() 