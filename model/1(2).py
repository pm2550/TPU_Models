import os
import re
import argparse
import time
import numpy as np
# 修改导入语句为tflite_runtime
from tflite_runtime.interpreter import Interpreter, load_delegate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    args = parser.parse_args()
    
    # 检查模型类型
    basename = os.path.basename(args.model)
    if "mobilenet" in basename.lower():
        print("检测到MobileNet模型")
        model_type = "cnn"
    elif "forecast" in basename.lower() or "lstm" in basename.lower():
        print("检测到LSTM/预测模型")
        model_type = "lstm"
    else:
        print("假设为图模型，将从输入形状判断")
        model_type = "auto"

    # 检查Edge TPU是否可用
    print("检查Edge TPU可用性...")
    tpu_available = False
    try:
        # 尝试加载TPU代理
        load_delegate('libedgetpu.so.1')
        print("Edge TPU库加载成功！")
        tpu_available = True
    except Exception as e:
        print(f"Edge TPU库加载失败: {e}")
        print("将使用CPU模式...")
    
    # 检查模型是否是EdgeTPU特定模型
    is_edgetpu_model = False
    try:
        with open(args.model, 'rb') as f:
            model_data = f.read(1024*1024)  # 读取前1MB
            is_edgetpu_model = b'edgetpu-custom-op' in model_data
            
        if is_edgetpu_model and not tpu_available:
            print("警告：检测到EdgeTPU特定模型，但EdgeTPU不可用")
            print("这类模型无法在CPU上运行，请使用非EdgeTPU编译版本的模型")
    except Exception as e:
        print(f"检查模型文件时出错: {e}")
    
    print(f"加载模型 {args.model} ...")
    try:
        if tpu_available:
            print("使用Edge TPU加载...")
            interpreter = Interpreter(
                model_path=args.model,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
        else:
            print("使用CPU加载...")
            interpreter = Interpreter(model_path=args.model)
        interpreter.allocate_tensors()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # 打印详细的输入信息
    print("Input details:")
    for i, detail in enumerate(input_details):
        print(f"Input {i}: name={detail['name']}, shape={detail['shape']}, type={detail['dtype'].__name__}")
    
    # 根据模型类型准备输入数据
    if model_type == "cnn":
        prepare_cnn_model_inputs(interpreter, input_details)
    elif model_type == "lstm":
        prepare_lstm_model_inputs(interpreter, input_details)
    elif model_type == "graph" or model_type == "auto":
        # 自动检测类型
        model_type = detect_model_type(input_details)
        print(f"自动检测模型类型: {model_type}")
        
        if model_type == "cnn":
            prepare_cnn_model_inputs(interpreter, input_details)
        elif model_type == "lstm":
            prepare_lstm_model_inputs(interpreter, input_details)
        else:
            # 使用通用准备方法
            prepare_generic_inputs(interpreter, input_details)
    
    # 计时(跑10次)
    times = []
    for i in range(10):
        try:
            start = time.time()
            interpreter.invoke()
            end = time.time()
            elapsed_ms = (end - start)*1000
            times.append(elapsed_ms)
            print(f"Run {i+1}: {elapsed_ms:.2f} ms")
        except Exception as e:
            print(f"推理失败: {e}")
            break
    
    if times:
        print(f"Avg: {np.mean(times):.2f} ms, Var: {np.var(times):.2f} ms^2")
        
        # 打印输出信息
        print("\n输出信息:")
        for i, detail in enumerate(output_details):
            output_data = interpreter.get_tensor(detail['index'])
            print(f"输出 {i}: shape={output_data.shape}, 前几个值={output_data.flatten()[:5]}")
    else:
        print("没有成功完成推理")

def detect_model_type(input_details):
    """
    检测模型类型: 图模型、CNN图像模型或LSTM序列模型
    """
    # 如果有graph相关的输入名称，就是图模型
    for detail in input_details:
        name = detail['name'].lower()
        if any(x in name for x in ["node", "neighbor", "graph", "relation"]):
            return "graph"
    
    # 检查输入张量的维度和形状
    for detail in input_details:
        shape = detail['shape']
        
        # CNN模型通常有4个维度 [batch, height, width, channels]
        if len(shape) == 4 and shape[3] in [1, 3, 4]:
            return "cnn"
            
        # LSTM模型通常有3个维度 [batch, time_steps, features]
        elif len(shape) == 3:
            return "lstm"
    
    # 默认为通用模型
    return "generic"

def prepare_cnn_model_inputs(interpreter, input_details):
    """
    为CNN模型(如MobileNet)准备输入
    """
    for detail in input_details:
        try:
            # 为CNN模型创建随机输入数据
            shape = detail['shape']
            if len(shape) == 4:  # [batch, height, width, channels]
                # 创建标准的随机图像输入 (范围0-1或0-255取决于量化类型)
                if detail['dtype'] == np.uint8:
                    # 量化模型: 0-255范围
                    data = np.random.randint(0, 256, size=shape, dtype=np.uint8)
                else:
                    # 浮点模型: 0-1范围
                    data = np.random.random(shape).astype(np.float32)
                
                print(f"为CNN输入 '{detail['name']}' 创建形状为 {shape} 的随机图像数据")
                interpreter.set_tensor(detail['index'], data)
            else:
                print(f"警告: CNN模型中的非标准输入 {detail['name']} 形状为 {shape}，跳过")
        except Exception as e:
            print(f"设置CNN模型输入 {detail['name']} 时出错: {e}")

def prepare_lstm_model_inputs(interpreter, input_details):
    """
    为LSTM序列模型准备输入
    """
    for detail in input_details:
        try:
            # 为LSTM模型创建随机输入数据
            shape = detail['shape']
            if len(shape) == 3:  # [batch, time_steps, features]
                # 创建随机序列数据
                data = np.random.random(shape).astype(detail['dtype'])
                print(f"为LSTM输入 '{detail['name']}' 创建形状为 {shape} 的随机序列数据")
                interpreter.set_tensor(detail['index'], data)
            else:
                print(f"警告: LSTM模型中的非标准输入 {detail['name']} 形状为 {shape}，跳过")
        except Exception as e:
            print(f"设置LSTM模型输入 {detail['name']} 时出错: {e}")

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