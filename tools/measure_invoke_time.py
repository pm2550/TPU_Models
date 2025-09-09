#!/usr/bin/env python3
"""
测试模型 invoke 时间，不使用 usbmon
"""
import time
import numpy as np
import sys
import os
import json
from pathlib import Path


def measure_model(model_path, warmup_count=50, test_count=100):
    """测量模型 invoke 时间"""
    try:
        # 尝试使用 pycoral
        from pycoral.utils.edgetpu import make_interpreter
        interpreter = make_interpreter(model_path)
        print(f"使用 pycoral 加载: {model_path}")
    except Exception as e1:
        try:
            # 回退到 tflite_runtime
            from tflite_runtime.interpreter import Interpreter, load_delegate
            interpreter = Interpreter(
                model_path=model_path,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
            print(f"使用 tflite_runtime 加载: {model_path}")
        except Exception as e2:
            print(f"模型加载失败: {e1}, {e2}")
            return None
    
    interpreter.allocate_tensors()
    
    # 准备输入数据
    input_details = interpreter.get_input_details()[0]
    if input_details['dtype'].__name__ == 'uint8':
        input_data = np.random.randint(0, 256, size=input_details['shape'], dtype=np.uint8)
    else:
        input_data = np.random.randint(-128, 128, size=input_details['shape'], dtype=np.int8)
    
    print(f"输入形状: {input_details['shape']}, 类型: {input_details['dtype']}")
    
    # Warm-up
    print(f"Warm-up {warmup_count} 次...")
    for _ in range(warmup_count):
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
    
    # 测试
    print(f"测试 {test_count} 次...")
    invoke_times = []
    
    for i in range(test_count):
        interpreter.set_tensor(input_details['index'], input_data)
        
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        
        invoke_time = end_time - start_time
        invoke_times.append(invoke_time)
        
        if (i + 1) % 20 == 0:
            print(f"  完成 {i + 1}/{test_count}")
    
    return invoke_times


def main():
    models = [
        # 分段模型
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg1_int8_edgetpu.tflite", "seg1"),
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg2_int8_edgetpu.tflite", "seg2"),
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg3_int8_edgetpu.tflite", "seg3"),
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg4_int8_edgetpu.tflite", "seg4"),
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg5_int8_edgetpu.tflite", "seg5"),
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg6_int8_edgetpu.tflite", "seg6"),
        ("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu/resnet101_seg7_int8_edgetpu.tflite", "seg7"),
        # 完整模型
        ("/home/10210/Desktop/OS/layered models/resnet101_full/tpu/resnet101_full_int8_edgetpu.tflite", "full"),
    ]
    
    results = {}
    
    for model_path, name in models:
        if not Path(model_path).exists():
            print(f"模型不存在: {model_path}")
            continue
            
        print(f"\n=== 测试 {name} ===")
        invoke_times = measure_model(model_path, warmup_count=50, test_count=100)
        
        if invoke_times:
            # 计算统计数据
            avg_time = sum(invoke_times) / len(invoke_times)
            min_time = min(invoke_times)
            max_time = max(invoke_times)
            
            # 转换为毫秒
            avg_ms = avg_time * 1000
            min_ms = min_time * 1000
            max_ms = max_time * 1000
            
            results[name] = {
                'avg_ms': avg_ms,
                'min_ms': min_ms,
                'max_ms': max_ms,
                'times': invoke_times
            }
            
            print(f"平均时间: {avg_ms:.2f} ms")
            print(f"最小时间: {min_ms:.2f} ms") 
            print(f"最大时间: {max_ms:.2f} ms")
        else:
            print(f"测试失败: {name}")
    
    # 输出汇总
    print(f"\n{'='*60}")
    print("=== 汇总结果 ===")
    print(f"{'模型':<10} {'平均时间(ms)':<15} {'最小时间(ms)':<15} {'最大时间(ms)':<15}")
    print("-" * 60)
    
    for name, data in results.items():
        print(f"{name:<10} {data['avg_ms']:<15.2f} {data['min_ms']:<15.2f} {data['max_ms']:<15.2f}")
    
    # 计算分段总和
    if all(f'seg{i}' in results for i in range(1, 8)):
        seg_total = sum(results[f'seg{i}']['avg_ms'] for i in range(1, 8))
        print(f"\n分段总和: {seg_total:.2f} ms")
        
        if 'full' in results:
            full_time = results['full']['avg_ms']
            ratio = seg_total / full_time
            print(f"Full 模型: {full_time:.2f} ms")
            print(f"分段/Full 比例: {ratio:.2f}x")


if __name__ == '__main__':
    main()

