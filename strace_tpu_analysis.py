#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用strace分析TPU IO开销：单层vs完整模型
"""

import os
import subprocess
import tempfile
import time
import re
import sys

def run_strace_tpu_analysis(model_path, model_name, num_inferences=20):
    """
    使用strace分析TPU模型的系统调用
    """
    print(f"\n🔍 strace分析: {model_name}")
    print(f"模型: {model_path}")
    print("=" * 60)
    
    # 创建strace输出文件
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.strace', delete=False) as f:
        strace_file = f.name
    
    # 创建测试脚本
    test_script = f'''
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

# 加载模型
interpreter = make_interpreter("{model_path}")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

if input_dtype == np.uint8:
    dummy_input = np.random.randint(0, 256, input_shape, dtype=np.uint8)
elif input_dtype == np.int8:
    dummy_input = np.random.randint(-128, 128, input_shape, dtype=np.int8)
else:
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

print(f"开始测试: {model_name}")
print(f"输入形状: {{input_shape}}")

# 预热
for _ in range(3):
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()

# 正式测试  
times = []
for i in range({num_inferences}):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]["index"])
    end = time.perf_counter()
    times.append((end - start) * 1000)

avg_time = sum(times) / len(times)
print(f"平均推理时间: {{avg_time:.3f}} ms")
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_file = f.name
    
    try:
        # 运行strace命令
        cmd = [
            'strace',
            '-c',  # 统计模式
            '-f',  # 跟踪子进程
            '-e', 'trace=read,write,readv,writev,pread64,pwrite64,ioctl,openat,close,mmap,munmap',
            '-o', strace_file,
            'bash', '-c', f'source .venv/bin/activate && python3 {script_file}'
        ]
        
        print(f"运行strace分析 ({num_inferences} 次推理)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"❌ strace失败: {result.stderr}")
            return None
        
        # 解析结果
        program_output = result.stdout
        print("✅ 程序输出:")
        print(program_output)
        
        # 解析strace统计
        with open(strace_file, 'r') as f:
            strace_output = f.read()
        
        analysis = parse_strace_output(strace_output, program_output)
        return analysis
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        return None
    finally:
        # 清理文件
        for tmp_file in [strace_file, script_file]:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)

def parse_strace_output(strace_output, program_output):
    """
    解析strace输出
    """
    analysis = {
        'syscall_stats': {},
        'total_time': 0.0,
        'inference_time': 0.0,
        'io_calls': 0,
        'io_time': 0.0,
        'tpu_calls': 0
    }
    
    # 提取推理时间
    time_match = re.search(r'平均推理时间: ([\d.]+) ms', program_output)
    if time_match:
        analysis['inference_time'] = float(time_match.group(1))
    
    # 解析系统调用统计
    lines = strace_output.split('\n')
    in_stats = False
    
    for line in lines:
        if '% time' in line and 'seconds' in line:
            in_stats = True
            continue
        elif '------' in line:
            continue
        elif 'total' in line and in_stats:
            # 总时间行
            parts = line.split()
            if len(parts) >= 2:
                try:
                    analysis['total_time'] = float(parts[1])
                except:
                    pass
            break
        elif in_stats and line.strip():
            # 统计行
            parts = line.split()
            if len(parts) >= 6:
                try:
                    percent_time = float(parts[0])
                    seconds = float(parts[1])
                    calls = int(parts[3])
                    syscall = parts[5]
                    
                    analysis['syscall_stats'][syscall] = {
                        'percent_time': percent_time,
                        'seconds': seconds,
                        'calls': calls
                    }
                    
                    # 分类系统调用
                    if syscall in ['read', 'write', 'readv', 'writev', 'pread64', 'pwrite64']:
                        analysis['io_calls'] += calls
                        analysis['io_time'] += seconds
                    elif syscall in ['ioctl']:
                        analysis['tpu_calls'] += calls
                        
                except (ValueError, IndexError):
                    continue
    
    return analysis

def compare_tpu_analyses(single_analysis, full_analysis):
    """
    对比单层和完整模型的分析结果
    """
    print(f"\n" + "=" * 80)
    print(f"🔬 TPU IO 开销对比分析")
    print(f"=" * 80)
    
    if not single_analysis or not full_analysis:
        print("❌ 分析数据不完整")
        return
    
    # 基本性能对比
    single_time = single_analysis['inference_time']
    full_time = full_analysis['inference_time']
    perf_ratio = single_time / full_time
    
    print(f"📊 基本性能:")
    print(f"   单层Conv2D:  {single_time:.3f} ms")
    print(f"   完整MobileNet: {full_time:.3f} ms")
    print(f"   单层/完整比率: {perf_ratio:.2f}x")
    
    # IO开销对比
    single_io_time = single_analysis['io_time'] * 1000  # 转换为ms
    full_io_time = full_analysis['io_time'] * 1000
    single_io_calls = single_analysis['io_calls']
    full_io_calls = full_analysis['io_calls']
    
    print(f"\n📡 IO开销对比:")
    print(f"   单层IO时间:   {single_io_time:.3f} ms ({single_io_calls} 次调用)")
    print(f"   完整IO时间:   {full_io_time:.3f} ms ({full_io_calls} 次调用)")
    print(f"   IO时间比率:   {single_io_time/full_io_time if full_io_time > 0 else 0:.2f}x")
    
    # IO开销占推理时间的比例
    single_io_ratio = (single_io_time / single_time * 100) if single_time > 0 else 0
    full_io_ratio = (full_io_time / full_time * 100) if full_time > 0 else 0
    
    print(f"\n🔍 IO开销占比:")
    print(f"   单层模型: {single_io_ratio:.1f}% IO开销")
    print(f"   完整模型: {full_io_ratio:.1f}% IO开销")
    
    # TPU控制开销
    single_tpu_calls = single_analysis['tpu_calls']
    full_tpu_calls = full_analysis['tpu_calls']
    
    print(f"\n⚡ TPU控制调用:")
    print(f"   单层模型: {single_tpu_calls} 次ioctl")
    print(f"   完整模型: {full_tpu_calls} 次ioctl")
    
    # 结论分析
    print(f"\n💡 分析结论:")
    if single_io_ratio > 50:
        print(f"🔴 单层模型被IO拖慢了!")
        print(f"   - IO开销占 {single_io_ratio:.1f}% 的推理时间")
        print(f"   - 证明小模型在TPU上主要瓶颈是数据传输")
    elif single_io_ratio > 20:
        print(f"🟡 单层模型有明显IO开销")
        print(f"   - IO开销占 {single_io_ratio:.1f}% 的推理时间")
        print(f"   - IO是性能瓶颈之一，但不是主要原因")
    else:
        print(f"🟢 单层模型IO开销较小")
        print(f"   - IO开销仅占 {single_io_ratio:.1f}% 的推理时间")
        print(f"   - 性能差主要由其他因素(TPU初始化、调度等)导致")

def main():
    """
    主函数
    """
    print("🍓 Raspberry Pi TPU IO开销strace分析")
    print("=" * 60)
    print("对比分析: 单层Conv2D vs 完整MobileNet的IO开销")
    print()
    
    # 检查strace
    try:
        subprocess.run(['strace', '--version'], capture_output=True, check=True)
    except:
        print("❌ 需要安装strace: sudo apt-get install strace")
        sys.exit(1)
    
    # 分析两个模型
    models = [
        ("./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite", "单层Conv2D"),
        ("./model/mobilenet.tflite", "完整MobileNet")
    ]
    
    analyses = []
    for model_path, model_name in models:
        if os.path.exists(model_path):
            analysis = run_strace_tpu_analysis(model_path, model_name)
            if analysis:
                analyses.append((model_name, analysis))
        else:
            print(f"❌ 模型不存在: {model_path}")
    
    # 对比分析
    if len(analyses) >= 2:
        single_analysis = next((a[1] for a in analyses if "单层" in a[0]), None)
        full_analysis = next((a[1] for a in analyses if "完整" in a[0]), None)
        
        if single_analysis and full_analysis:
            compare_tpu_analyses(single_analysis, full_analysis)
    else:
        print("❌ 需要两个模型进行对比分析")

if __name__ == "__main__":
    main() 