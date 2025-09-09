#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细strace分析 - 解决时间精度和ioctl失败问题
"""

import subprocess
import tempfile
import os

def detailed_strace_analysis():
    print("🔬 详细strace分析")
    print("=" * 60)
    
    # 1. 更详细的strace - 显示具体的系统调用内容
    print("\n1️⃣ 详细系统调用跟踪 (单层Conv2D):")
    print("-" * 50)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.strace', delete=False) as f:
        strace_file = f.name
    
    cmd1 = [
        'strace', 
        '-tt',  # 显示微秒级时间戳
        '-T',   # 显示每个系统调用的时间
        '-v',   # 详细输出
        '-e', 'trace=openat,read,write,ioctl,mmap,munmap,close',
        '-o', strace_file,
        'bash', '-c', '''
source .venv/bin/activate
python3 -c "
import time
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

print('开始加载模型...')
interpreter = make_interpreter('./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite')
print('分配张量...')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
dummy_input = np.random.randint(-128, 128, input_details[0]['shape'], dtype=np.int8)

print('开始推理...')
start = time.perf_counter()
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
end = time.perf_counter()
print(f'推理时间: {(end-start)*1000:.3f} ms')
"
        '''
    ]
    
    try:
        result = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
        print("程序输出:")
        print(result.stdout)
        
        # 读取详细的strace日志
        if os.path.exists(strace_file):
            with open(strace_file, 'r') as f:
                strace_content = f.read()
            
            print("\n📋 详细系统调用分析:")
            
            # 分析ioctl调用
            ioctl_lines = [line for line in strace_content.split('\n') if 'ioctl' in line]
            if ioctl_lines:
                print(f"🔍 IOCTL调用分析 ({len(ioctl_lines)} 次):")
                for i, line in enumerate(ioctl_lines[:5]):  # 只显示前5次
                    print(f"  {i+1}: {line}")
            
            # 分析文件操作
            file_ops = [line for line in strace_content.split('\n') 
                       if any(op in line for op in ['openat', 'read', 'write', 'mmap'])]
            if file_ops:
                print(f"\n📁 文件操作分析 ({len(file_ops)} 次):")
                for i, line in enumerate(file_ops[:10]):  # 只显示前10次
                    print(f"  {i+1}: {line}")
            
            # 查找时间最长的操作
            timed_lines = [line for line in strace_content.split('\n') if '<' in line and '>' in line]
            if timed_lines:
                times = []
                for line in timed_lines:
                    try:
                        # 提取时间，格式如 <0.000123>
                        time_part = line.split('<')[-1].split('>')[0]
                        times.append((float(time_part), line))
                    except:
                        continue
                
                if times:
                    times.sort(reverse=True)
                    print(f"\n⏱️  最耗时的系统调用 (前5个):")
                    for i, (duration, line) in enumerate(times[:5]):
                        print(f"  {i+1}: {duration:.6f}s - {line[:80]}...")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        if os.path.exists(strace_file):
            os.unlink(strace_file)
    
    # 2. 使用不同的时间测量方法
    print(f"\n2️⃣ 时间精度验证:")
    print("-" * 50)
    
    cmd2 = [
        'strace', 
        '-c',    # 统计模式
        '-S', 'time',  # 按时间排序
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

# 运行5次推理
for i in range(5):
    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()
"
        '''
    ]
    
    try:
        result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
        print("多次推理的系统调用统计:")
        print(result2.stderr)
    except Exception as e:
        print(f"❌ 错误: {e}")

    # 3. 检查EdgeTPU设备状态
    print(f"\n3️⃣ EdgeTPU设备状态检查:")
    print("-" * 50)
    
    # 检查设备文件
    device_checks = [
        'ls -la /dev/apex_0 2>/dev/null || echo "EdgeTPU设备未找到"',
        'lsusb | grep -i coral || echo "Coral USB设备未找到"',
        'dmesg | grep -i "apex\\|coral" | tail -5 || echo "无相关内核消息"'
    ]
    
    for cmd in device_checks:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f"💡 {cmd.split('||')[0].strip()}:")
            print(f"   {result.stdout.strip() or result.stderr.strip()}")
        except:
            print(f"   检查失败")

def main():
    detailed_strace_analysis()

if __name__ == "__main__":
    main() 