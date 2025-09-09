import os
import re
import argparse
import time
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter, load_delegate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    args = parser.parse_args()
    
    # 解析文件名，假设 "./model/3000.tflite" => num_nodes=3000
    basename = os.path.basename(args.model)
    m = re.match(r"(\d+)\.tflite", basename)
    if not m:
        print("无法从模型名中解析节点数, 默认使用 1000")
        num_nodes = 1000
    else:
        num_nodes = int(m.group(1))
    
    # 其他超参数(须与编译时一致)
    input_dim = 32  # 这也可以动态获取，但通常特征维度是固定的
    
    print(f"Loading model {args.model} ...")
    
    # 详细检查Edge TPU是否可用
    print("检查Edge TPU可用性...")
    
    def check_edgetpu_availability():
        """详细检查Edge TPU硬件和库是否可用"""
        # 1. 检查库是否可以加载
        try:
            from tensorflow.lite.python.interpreter import load_delegate
        except ImportError:
            print("无法导入TensorFlow Lite库")
            return False
            
        # 2. 检查libedgetpu.so.1是否存在
        import subprocess
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if 'libedgetpu.so.1' not in result.stdout:
                # 也可以尝试直接检查文件
                import os
                possible_locations = [
                    '/usr/lib/libedgetpu.so.1',
                    '/usr/local/lib/libedgetpu.so.1',
                    '/lib/libedgetpu.so.1'
                ]
                if not any(os.path.exists(loc) for loc in possible_locations):
                    print("未找到libedgetpu.so.1库文件")
                    print("请确保已安装Edge TPU运行时")
                    return False
        except Exception as e:
            print(f"检查libedgetpu.so.1时出错: {e}")
            
        # 3. 尝试检测硬件
        try:
            import subprocess
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            # 检查常见的EdgeTPU USB设备标识
            edgetpu_ids = ['18d1:9302', '18d1:9301', '1a6e:089a']  # Google Coral设备的VID:PID
            if not any(id in result.stdout for id in edgetpu_ids):
                print("未检测到Edge TPU USB设备")
                print("请确保已连接Edge TPU设备（如Google Coral）")
                return False
        except Exception as e:
            print(f"检查Edge TPU硬件时出错: {e}")
    
        # 4. 尝试创建一个简单的代理
        try:
            load_delegate('libedgetpu.so.1')
            print("Edge TPU库加载成功！")
            return True
        except Exception as e:
            print(f"Edge TPU库加载失败: {e}")
            print("虽然找到了库文件，但加载失败，可能是兼容性问题")
            return False
    
    # 检查TPU
    tpu_available = check_edgetpu_availability()
    
    # 根据TPU可用性决定如何加载模型
    try:
        if tpu_available:
            print("使用Edge TPU加载模型...")
            interpreter = Interpreter(
                model_path=args.model,
                experimental_delegates=[load_delegate('libedgetpu.so.1')]
            )
        else:
            print("使用CPU加载模型...")
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
    
    # 检测模型类型
    model_type = detect_model_type(input_details)
    print(f"检测到模型类型: {model_type}")
    
    # 根据模型类型准备输入数据
    if model_type == "graph":
        prepare_graph_model_inputs(interpreter, input_details, num_nodes, input_dim)
    elif model_type == "cnn":
        prepare_cnn_model_inputs(interpreter, input_details)
    elif model_type == "lstm":
        prepare_lstm_model_inputs(interpreter, input_details)
    else:
        print(f"警告: 未知模型类型 {model_type}，将使用默认图模型输入")
        prepare_graph_model_inputs(interpreter, input_details, num_nodes, input_dim)
    
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
    
    # 默认为图模型
    return "graph"

def prepare_graph_model_inputs(interpreter, input_details, num_nodes, input_dim):
    """
    为图神经网络模型准备输入
    """
    # 动态确定邻居数量 - 从neighbor_indices输入形状获取
    max_neighbors = None
    for detail in input_details:
        if "neighbor_indices" in detail['name']:
            # 形状是[1, max_neighbors]
            max_neighbors = detail['shape'][1]
            print(f"动态获取到邻居数量: {max_neighbors}")
            break
    
    # 如果没有找到，使用默认值
    if max_neighbors is None:
        max_neighbors = 32  # 默认值
        print(f"无法获取邻居数量，使用默认值: {max_neighbors}")
    
    # 准备填充张量 - 根据模型实际输入进行修改
    # 节点特征
    node_features_data = np.random.randn(num_nodes, input_dim).astype(np.float32)
    
    # 尾节点索引（用于链接预测）
    tail_index_data = np.zeros(1, dtype=np.int32)
    
    # 邻居索引 - 使用动态获取的max_neighbors
    neighbor_indices_data = np.zeros((1, max_neighbors), dtype=np.int32)
    
    # 头节点索引（用于链接预测）
    head_index_data = np.zeros(1, dtype=np.int32)
    
    # 关系索引 - 使用动态获取的max_neighbors
    relation_indices_data = np.zeros((1, max_neighbors), dtype=np.int32)
    
    # 邻居掩码 - 使用动态获取的max_neighbors
    neighbor_mask_data = np.zeros((1, max_neighbors), dtype=np.float32)
    
    # 设置输入张量 - 根据输入详情中的名称匹配正确的输入张量
    for detail in input_details:
        try:
            if "node_features" in detail['name']:
                interpreter.set_tensor(detail['index'], node_features_data)
            elif "tail_index" in detail['name']:
                interpreter.set_tensor(detail['index'], tail_index_data)
            elif "neighbor_indices" in detail['name']:
                interpreter.set_tensor(detail['index'], neighbor_indices_data)
            elif "head_index" in detail['name']:
                interpreter.set_tensor(detail['index'], head_index_data)
            elif "relation_indices" in detail['name']:
                interpreter.set_tensor(detail['index'], relation_indices_data)
            elif "neighbor_mask" in detail['name']:
                interpreter.set_tensor(detail['index'], neighbor_mask_data)
            else:
                print(f"警告: 未知图模型输入 {detail['name']}，跳过")
        except Exception as e:
            print(f"设置图模型输入 {detail['name']} 时出错: {e}")

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

if __name__=="__main__":
    main()