import os
import argparse
import time
import numpy as np
import platform

# 使用tflite_runtime而不是tensorflow
from tflite_runtime.interpreter import Interpreter, load_delegate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to EdgeTPU .tflite model")
    parser.add_argument("--verbose", action="store_true", help="显示详细信息")
    args = parser.parse_args()
    
    # 首先打印系统信息
    print("=== 系统信息 ===")
    print(f"平台: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"处理器: {platform.processor()}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在")
        return
        
    # 检测模型类型
    basename = os.path.basename(args.model)
    if "mobilenet" in basename.lower():
        print("检测到MobileNet模型")
        model_type = "cnn"
    elif "forecast" in basename.lower() or "weather" in basename.lower() or "lstm" in basename.lower():
        print("检测到预测/LSTM模型")
        model_type = "lstm"
    else:
        print("未识别的模型类型，将尝试自动检测")
        model_type = "auto"
    
    print("\n=== EdgeTPU检测 ===")
    try:
        # 尝试提前检测TPU并加载代理
        delegate = load_delegate('libedgetpu.so.1')
        print("✓ 成功加载EdgeTPU库")
        
        # 检查TPU硬件
        import subprocess
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if '1a6e:089a' in result.stdout:
            print("✓ 检测到Coral EdgeTPU设备 (ID: 1a6e:089a)")
        else:
            print("! 未检测到已知的EdgeTPU设备，请检查USB连接")
            for line in result.stdout.splitlines():
                if any(id in line for id in ['1a6e', '18d1']):
                    print(f"  可能的EdgeTPU设备: {line}")
    except Exception as e:
        print(f"✗ EdgeTPU库加载失败: {e}")
        print("  请确保已安装libedgetpu并且TPU设备已连接")
        return
    
    # EdgeTPU特定的改进模型加载方法
    print("\n=== 加载EdgeTPU模型 ===")
    print(f"模型路径: {args.model}")
    try:
        # 为模型解释器设置选项
        options = {}
        
        # 在Linux上，尝试使用XNNPACK或mmap加速
        print("使用EdgeTPU代理加载模型...")
        interpreter = Interpreter(
            model_path=args.model,
            experimental_delegates=[delegate]
        )
        
        # 分配张量
        print("分配输入/输出张量...")
        interpreter.allocate_tensors()
        print("✓ 模型加载成功!")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("  可能的原因:")
        print("  - 模型可能不是为EdgeTPU正确编译的")
        print("  - TPU权限可能不正确，尝试: sudo usermod -aG plugdev $USER")
        print("  - 可能需要重启设备使权限更改生效")
        return
    
    # 获取输入和输出信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n=== 模型信息 ===")
    print(f"输入数量: {len(input_details)}")
    for i, detail in enumerate(input_details):
        print(f"输入 {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype'].__name__}")
    
    print(f"输出数量: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"输出 {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype'].__name__}")
    
    # 准备输入数据
    print("\n=== 准备输入数据 ===")
    if model_type == "cnn":
        prepare_cnn_inputs(interpreter, input_details)
    elif model_type == "lstm":
        prepare_lstm_inputs(interpreter, input_details)
    else:
        # 自动检测最合适的输入准备方法
        if len(input_details) == 1:
            shape = input_details[0]['shape']
            if len(shape) == 4:  # [batch, height, width, channels]
                model_type = "cnn"
                prepare_cnn_inputs(interpreter, input_details)
            elif len(shape) == 3:  # [batch, time_steps, features]
                model_type = "lstm"
                prepare_lstm_inputs(interpreter, input_details)
            else:
                prepare_generic_inputs(interpreter, input_details)
        else:
            prepare_generic_inputs(interpreter, input_details)
    
    # 运行推理
    print("\n=== 执行EdgeTPU推理 ===")
    
    # 首先进行一次预热推理
    print("预热推理...")
    interpreter.invoke()
    
    # 然后进行性能测试
    times = []
    n_runs = 10
    print(f"进行{n_runs}次推理测量性能...")
    for i in range(n_runs):
        start = time.time()
        interpreter.invoke()
        end = time.time()
        elapsed_ms = (end - start) * 1000
        times.append(elapsed_ms)
        print(f"Run {i+1}: {elapsed_ms:.2f} ms")
    
    # 计算统计数据
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print("\n=== 性能结果 ===")
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"标准差: {std_time:.2f} ms")
    print(f"最小值: {min_time:.2f} ms")
    print(f"最大值: {max_time:.2f} ms")
    print(f"推理速度: {1000/avg_time:.1f} FPS")
    
    # 显示输出数据
    print("\n=== 输出数据 ===")
    for i, detail in enumerate(output_details):
        output_data = interpreter.get_tensor(detail['index'])
        flat_output = output_data.flatten()
        
        if len(flat_output) > 0:
            print(f"输出 {i} ({detail['name']}): shape={output_data.shape}")
            if len(flat_output) <= 10:
                print(f"  所有值: {flat_output}")
            else:
                print(f"  前5个值: {flat_output[:5]}")
                print(f"  最大值: {np.max(flat_output)}, 最小值: {np.min(flat_output)}")

def prepare_cnn_inputs(interpreter, input_details):
    """为CNN图像模型准备输入"""
    for detail in input_details:
        shape = detail['shape']
        if len(shape) == 4:  # [batch, height, width, channels]
            # 生成随机图像数据
            if detail['dtype'] == np.uint8:
                # 量化模型用0-255的值
                data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
                print(f"为CNN图像输入准备随机uint8数据，形状: {shape}")
            else:
                # 非量化模型用0-1的浮点值
                data = np.random.random(shape).astype(np.float32)
                print(f"为CNN图像输入准备随机float32数据，形状: {shape}")
                
            interpreter.set_tensor(detail['index'], data)
        else:
            print(f"警告: CNN模型有非标准输入形状: {shape}")

def prepare_lstm_inputs(interpreter, input_details):
    """为LSTM/预测模型准备输入"""
    for detail in input_details:
        shape = detail['shape']
        if len(shape) == 3:  # [batch, time_steps, features]
            # 生成随机序列数据
            if detail['dtype'] == np.float32:
                # 大多数LSTM使用float32
                data = np.random.randn(*shape).astype(np.float32)
            else:
                # 处理其他数据类型
                data = np.random.random(shape).astype(detail['dtype'])
                
            print(f"为LSTM序列输入准备随机数据，形状: {shape}")
            interpreter.set_tensor(detail['index'], data)
        else:
            print(f"警告: LSTM模型有非标准输入形状: {shape}")

def prepare_generic_inputs(interpreter, input_details):
    """为一般模型准备输入"""
    for detail in input_details:
        shape = detail['shape']
        dtype = detail['dtype']
        
        # 根据数据类型生成合适的随机数据
        if dtype == np.float32:
            data = np.random.randn(*shape).astype(np.float32)
        elif dtype == np.uint8:
            data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        elif dtype == np.int32 or dtype == np.int64:
            data = np.random.randint(-100, 100, size=shape, dtype=dtype)
        else:
            # 默认使用零填充
            data = np.zeros(shape, dtype=dtype)
            
        print(f"为输入 '{detail['name']}' 准备随机数据，形状: {shape}")
        interpreter.set_tensor(detail['index'], data)

if __name__ == "__main__":
    main()