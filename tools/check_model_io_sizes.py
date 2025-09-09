#!/usr/bin/env python3
"""
检查 TFLite 模型的输入输出尺寸
"""
import sys
import numpy as np
from pathlib import Path

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("需要 tflite_runtime 或 tensorflow")
        sys.exit(1)


def check_model_io(model_path):
    """检查模型的输入输出尺寸"""
    try:
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        
        # 获取输入信息
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n=== {model_path.name} ===")
        print(f"文件大小: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
        
        total_input_bytes = 0
        total_output_bytes = 0
        
        print("\n输入张量:")
        for i, inp in enumerate(input_details):
            shape = inp['shape']
            dtype = inp['dtype']
            
            # 计算字节数
            if dtype == np.uint8 or dtype == np.int8:
                bytes_per_element = 1
            elif dtype == np.float16:
                bytes_per_element = 2
            elif dtype == np.float32 or dtype == np.int32:
                bytes_per_element = 4
            else:
                bytes_per_element = 4  # 默认
            
            total_elements = np.prod(shape) if None not in shape else 0
            tensor_bytes = total_elements * bytes_per_element
            total_input_bytes += tensor_bytes
            
            print(f"  {i}: shape={shape}, dtype={dtype}, bytes={tensor_bytes:,}")
        
        print("\n输出张量:")
        for i, out in enumerate(output_details):
            shape = out['shape']
            dtype = out['dtype']
            
            # 计算字节数
            if dtype == np.uint8 or dtype == np.int8:
                bytes_per_element = 1
            elif dtype == np.float16:
                bytes_per_element = 2
            elif dtype == np.float32 or dtype == np.int32:
                bytes_per_element = 4
            else:
                bytes_per_element = 4  # 默认
            
            total_elements = np.prod(shape) if None not in shape else 0
            tensor_bytes = total_elements * bytes_per_element
            total_output_bytes += tensor_bytes
            
            print(f"  {i}: shape={shape}, dtype={dtype}, bytes={tensor_bytes:,}")
        
        print(f"\n理论数据量:")
        print(f"  总输入: {total_input_bytes:,} bytes ({total_input_bytes/1024/1024:.2f} MB)")
        print(f"  总输出: {total_output_bytes:,} bytes ({total_output_bytes/1024/1024:.2f} MB)")
        
        return total_input_bytes, total_output_bytes
        
    except Exception as e:
        print(f"分析 {model_path.name} 失败: {e}")
        return 0, 0


def main():
    tpu_dir = Path("/home/10210/Desktop/OS/layered models/resnet101_balanced/tpu")
    
    print("ResNet101 EdgeTPU 分段模型 I/O 尺寸分析")
    print("=" * 60)
    
    total_theoretical_in = 0
    total_theoretical_out = 0
    
    for seg in range(1, 8):
        model_path = tpu_dir / f"resnet101_seg{seg}_int8_edgetpu.tflite"
        if model_path.exists():
            in_bytes, out_bytes = check_model_io(model_path)
            total_theoretical_in += in_bytes
            total_theoretical_out += out_bytes
    
    print(f"\n{'='*60}")
    print(f"理论总计:")
    print(f"  总输入: {total_theoretical_in:,} bytes ({total_theoretical_in/1024/1024:.2f} MB)")
    print(f"  总输出: {total_theoretical_out:,} bytes ({total_theoretical_out/1024/1024:.2f} MB)")


if __name__ == '__main__':
    main()

