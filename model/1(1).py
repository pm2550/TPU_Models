import os
import re
import argparse
import time
import sys
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter, load_delegate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--debug", action="store_true", help="启用详细调试输出")
    args = parser.parse_args()
    
    print(f"加载模型: {args.model}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件 {args.model} 不存在!")
        return
    
    # 读取模型文件并检查是否为EdgeTPU模型
    try:
        with open(args.model, 'rb') as f:
            model_data = f.read(1024*1024)  # 读取前1MB进行检查
            if b'edgetpu-custom-op' in model_data:
                print("警告: 检测到EdgeTPU特定模型，需要TPU硬件支持")
    except Exception as e:
        print(f"读取模型文件时出错: {e}")
        
    # 检查模型类型
    basename = os.path.basename(args.model)
    if "mobilenet" in basename.lower():
        print("检测到MobileNet CNN模型")
        model_type = "cnn"
    elif "forecast" in basename.lower() or "lstm" in basename.lower():
        print("检测到LSTM/预测模型")
        model_type = "lstm"
    else:
        print("未能从文件名判断模型类型，将自动检测")
        model_type = "auto"

    # 检查Edge TPU是否可用
    print("\n===== 检查Edge TPU可用性 =====")
    tpu_available = False
    
    # 检查EdgeTPU库是否存在
    import subprocess
    try:
        result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
        if 'libedgetpu.so.1' in result.stdout:
            print("√ 找到libedgetpu.so.1库")
        else:
            print("× 未找到libedgetpu.so.1库")
    except Exception as e:
        print(f"检查库文件时出错: {e}")
    
    # 检查TPU硬件
    try:
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        tpu_ids = ['1a6e:089a', '18d1:9302', '18d1:9301']
        for id in tpu_ids:
            if id in result.stdout:
                print(f"√ 找到Edge TPU硬件设备 ({id})")
                tpu_available = True
                break
        if not tpu_available:
            print("× 未找到Edge TPU硬件设备")
    except Exception as e:
        print(f"检查硬件时出错: {e}")
    
    # 尝试加载Edge TPU代理
    try:
        delegate = load_delegate('libedgetpu.so.1')
        print("√ 成功加载Edge TPU代理")
        tpu_available = True
    except Exception as e:
        print(f"× 加载Edge TPU代理失败: {e}")
        tpu_available = False
    
    print(f"TPU可用性结论: {'可用' if tpu_available else '不可用'}")
    
    # 尝试加载模型
    print("\n===== 加载模型 =====")
    try:
        # 先尝试使用TPU加载模型
        if tpu_available:
            print("尝试使用Edge TPU加载...")
            try:
                interpreter = Interpreter(
                    model_path=args.model,
                    experimental_delegates=[load_delegate('libedgetpu.so.1')]
                )
                print("使用TPU加载成功!")
            except Exception as e:
                print(f"使用TPU加载失败: {e}")
                print("回退到CPU模式...")
                interpreter = Interpreter(model_path=args.model)
                print("使用CPU加载成功!")
        else:
            print("使用CPU加载...")
            interpreter = Interpreter(model_path=args.model)
            print("使用CPU加载成功!")
        
        interpreter.allocate_tensors()
        print("张量分配成功!")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 获取输入和输出详情
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n===== 模型输入输出信息 =====")
    print(f"输入数量: {len(input_details)}")
    for i, detail in enumerate(input_details):
        print(f"输入 {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype'].__name__}")
    
    print(f"输出数量: {len(output_details)}")
    for i, detail in enumerate(output_details):
        print(f"输出 {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype'].__name__}")
    
    # 根据模型类型自动检测
    if model_type == "auto":
        # 从输入形状自动判断
        if len(input_details) == 1:
            shape = input_details[0]['shape']
            if len(shape) == 4 and shape[3] in [1, 3, 4]:
                model_type = "cnn"
                print(f"自动检测为CNN模型，输入形状: {shape}")
            elif len(shape) == 3:
                model_type = "lstm"
                print(f"自动检测为LSTM模型，输入形状: {shape}")
            else:
                model_type = "unknown"
                print(f"无法识别的输入形状: {shape}")
        else:
            for detail in input_details:
                if any(x in detail['name'].lower() for x in ["node", "graph", "neighbor"]):
                    model_type = "graph"
                    print("自动检测为图神经网络模型")
                    break
            if model_type == "auto":
                model_type = "unknown"
                print("无法自动检测模型类型")
    
    # 为模型准备输入数据
    print("\n===== 准备输入数据 =====")
    try:
        if model_type == "cnn":
            prepare_cnn_inputs(interpreter, input_details)
        elif model_type == "lstm":
            prepare_lstm_inputs(interpreter, input_details)
        elif model_type == "graph":
            prepare_graph_inputs(interpreter, input_details)
        else:
            print("使用通用方法准备输入...")
            prepare_generic_inputs(interpreter, input_details)
    except Exception as e:
        print(f"准备输入数据时出错: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return
    
    # 执行推理
    print("\n===== 执行推理 =====")
    times = []
    for i in range(10):
        try:
            start = time.time()
            interpreter.invoke()
            end = time.time()
            elapsed_ms = (end - start) * 1000
            times.append(elapsed_ms)
            print(f"运行 {i+1}: {elapsed_ms:.2f} ms")
        except Exception as e:
            print(f"推理失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            break
    
    # 显示结果
    if times:
        print(f"\n推理成功完成! 平均时间: {np.mean(times):.2f} ms, 方差: {np.var(times):.2f} ms^2")
        
        print("\n===== 输出结果示例 =====")
        for i, detail in enumerate(output_details):
            output_data = interpreter.get_tensor(detail['index'])
            flat_output = output_data.flatten()
            if len(flat_output) > 0:
                print(f"输出 {i} ({detail['name']}): shape={output_data.shape}")
                max_display = min(5, len(flat_output))
                print(f"  前{max_display}个值: {flat_output[:max_display]}")
                print(f"  最大值: {np.max(flat_output)}, 最小值: {np.min(flat_output)}")
                if np.isnan(flat_output).any():
                    print("  警告: 输出包含NaN值!")
    else:
        print("\n没有成功完成推理")

def prepare_cnn_inputs(interpreter, input_details):
    """为CNN模型准备输入"""
    for detail in input_details:
        shape = detail['shape']
        if len(shape) == 4:  # [batch, height, width, channels]
            # 创建随机图像数据
            if detail['dtype'] == np.uint8:
                # 量化模型使用0-255范围
                data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
            else:
                # 浮点模型使用0-1范围
                data = np.random.random(shape).astype(detail['dtype'])
            
            print(f"为CNN输入 '{detail['name']}' 创建形状为 {shape} 的随机图像数据")
            interpreter.set_tensor(detail['index'], data)
        else:
            raise ValueError(f"CNN模型应有4D输入，但找到形状: {shape}")

def prepare_lstm_inputs(interpreter, input_details):
    """为LSTM/预测模型准备输入"""
    for detail in input_details:
        shape = detail['shape']
        if len(shape) == 3:  # [batch, time_steps, features]
            # 创建随机序列数据
            data = np.random.random(shape).astype(detail['dtype'])
            print(f"为LSTM输入 '{detail['name']}' 创建形状为 {shape} 的随机序列数据")
            interpreter.set_tensor(detail['index'], data)
        else:
            raise ValueError(f"LSTM模型应有3D输入，但找到形状: {shape}")

def prepare_graph_inputs(interpreter, input_details):
    """为图神经网络模型准备输入"""
    # 假设一些默认值
    num_nodes = 1000
    input_dim = 32
    max_neighbors = 32
    
    # 遍历所有输入，根据名称准备相应数据
    for detail in input_details:
        name = detail['name'].lower()
        shape = detail['shape']
        dtype = detail['dtype']
        
        try:
            if "node_features" in name or "features" in name:
                # 如果能够从形状中获取节点数和特征维度，使用实际值
                if len(shape) == 2:
                    num_nodes, input_dim = shape
                data = np.random.randn(num_nodes, input_dim).astype(dtype)
                print(f"为节点特征 '{detail['name']}' 创建形状为 {shape} 的随机特征")
                
            elif "neighbor" in name and "indices" in name:
                # 邻居索引
                if len(shape) >= 2:
                    max_neighbors = shape[1]
                data = np.zeros((1, max_neighbors), dtype=dtype)
                print(f"为邻居索引 '{detail['name']}' 创建形状为 {shape} 的零数组")
                
            elif "head" in name or "tail" in name:
                # 头尾节点索引
                data = np.zeros(shape, dtype=dtype)
                print(f"为节点索引 '{detail['name']}' 创建形状为 {shape} 的零数组")
                
            elif "relation" in name:
                # 关系索引
                data = np.zeros(shape, dtype=dtype)
                print(f"为关系索引 '{detail['name']}' 创建形状为 {shape} 的零数组")
                
            elif "mask" in name:
                # 掩码
                data = np.zeros(shape, dtype=dtype)
                print(f"为掩码 '{detail['name']}' 创建形状为 {shape} 的零数组")
                
            else:
                # 其他未知输入
                data = np.zeros(shape, dtype=dtype)
                print(f"为未知输入 '{detail['name']}' 创建形状为 {shape} 的零数组")
            
            interpreter.set_tensor(detail['index'], data)
        except Exception as e:
            print(f"为输入 '{detail['name']}' 准备数据时出错: {e}")
            raise

def prepare_generic_inputs(interpreter, input_details):
    """通用输入准备方法"""
    for detail in input_details:
        shape = detail['shape']
        dtype = detail['dtype']
        
        # 根据数据类型创建合适的随机数据
        if dtype == np.float32 or dtype == np.float64:
            # 浮点数据，使用正态分布
            data = np.random.randn(*shape).astype(dtype)
        elif dtype == np.uint8:
            # 无符号整数，使用0-255范围
            data = np.random.randint(0, 256, size=shape, dtype=dtype)
        elif dtype == np.int32 or dtype == np.int64:
            # 有符号整数，使用小范围
            data = np.random.randint(-10, 10, size=shape, dtype=dtype)
        else:
            # 其他类型，使用零数组
            data = np.zeros(shape, dtype=dtype)
        
        print(f"为输入 '{detail['name']}' 创建形状为 {shape} 的随机数据")
        interpreter.set_tensor(detail['index'], data)

if __name__ == "__main__":
    main()