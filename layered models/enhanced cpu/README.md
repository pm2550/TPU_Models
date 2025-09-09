# Enhanced Layered Models - CPU版本

## 📁 目录内容
本目录包含8个增强分层模型的CPU版本（INT8量化TFLite格式）

## 🔧 模型架构

### Conv-Stack-n 系列
- **conv_stack_1_int8.tflite** (3.4KB) - 1层Conv2D
- **conv_stack_3_int8.tflite** (24.5KB) - 3层Conv2D  
- **conv_stack_5_int8.tflite** (45.6KB) - 5层Conv2D
- **conv_stack_7_int8.tflite** (66.7KB) - 7层Conv2D

**规格**:
- 输入: 224×224×3 (RGB图像)
- 输出: 224×224×32
- 操作: Conv2D 3×3, 32 filters, stride 1, ReLU6

### DW-Stack-n 系列  
- **dw_stack_1_int8.tflite** (2.9KB) - 1层DepthwiseConv2D
- **dw_stack_3_int8.tflite** (6.3KB) - 3层DepthwiseConv2D
- **dw_stack_5_int8.tflite** (9.7KB) - 5层DepthwiseConv2D
- **dw_stack_7_int8.tflite** (13KB) - 7层DepthwiseConv2D

**规格**:
- 输入: 224×224×32
- 输出: 224×224×32  
- 操作: DepthwiseConv2D 3×3, depth=1, stride 1, ReLU6

## 🚀 使用方法

### Python TensorFlow Lite
```python
import tensorflow as tf
import numpy as np

# 加载模型
interpreter = tf.lite.Interpreter(model_path="conv_stack_1_int8.tflite")
interpreter.allocate_tensors()

# 获取输入输出详情
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 准备输入数据 (UINT8, 0-255)
input_data = np.random.randint(0, 256, size=(1, 224, 224, 3), dtype=np.uint8)

# 推理
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
```

### C++ TensorFlow Lite
```cpp
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

// 加载模型
auto model = tflite::FlatBufferModel::BuildFromFile("conv_stack_1_int8.tflite");
tflite::InterpreterBuilder builder(*model, resolver);
std::unique_ptr<tflite::Interpreter> interpreter;
builder(&interpreter);

// 分配张量并运行推理
interpreter->AllocateTensors();
interpreter->Invoke();
```

## ⚡ 性能特点
- **INT8量化**: 模型大小减少75%，推理速度提升
- **CPU优化**: 适合在各种CPU平台运行
- **低内存占用**: 适合移动设备和嵌入式系统
- **即插即用**: 无需特殊硬件加速器

## 🎯 适用场景
- 移动应用开发
- 嵌入式系统部署
- 服务器端批量处理
- 开发和测试环境 