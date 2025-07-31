#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

def load_interpreter(model_path, use_tpu=True):
    if use_tpu:
        from pycoral.utils.edgetpu import make_interpreter
        return make_interpreter(model_path)
    else:
        return Interpreter(model_path)          # 纯 CPU

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="TFLite 模型文件路径 (.tflite)")
    parser.add_argument("--warmup", type=int, default=5, help="热身推理次数，不计入统计")
    parser.add_argument("--runs", type=int, default=50, help="统计时的推理次数")
    parser.add_argument("--cpu", action="store_true",
                    help="Force CPU, do not use Edge‑TPU")
    args = parser.parse_args()

    # 1. 加载 TFLite 模型
    # interpreter =make_interpreter(args.model)
    interpreter = load_interpreter(args.model, use_tpu=not args.cpu)

    model_path=args.model
    interpreter.allocate_tensors()
    
    # 获取输入和输出张量信息
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    

    input_idx = input_details[0]['index']
    input_shape = input_details[0]['shape']
   

    # 2. 根据 input_shape 生成随机数据
    dtype = input_details[0]['dtype']
    dummy_input = np.random.random_sample(input_shape).astype(dtype)

    # 3. 热身推理 
    for _ in range(args.warmup):
        interpreter.set_tensor(input_idx, dummy_input)
        interpreter.invoke()


    seg_pre, seg_infer, seg_post = [], [], []      
    total_times = []                              
    for _ in range(args.runs):
        t_a = time.perf_counter()                  

        interpreter.set_tensor(input_idx, dummy_input)
        t_b = time.perf_counter()                 

        interpreter.invoke()
        t_c = time.perf_counter()                

        _ = interpreter.get_tensor(output_details[0]['index'])
        t_d = time.perf_counter()                

        seg_pre.append((t_b - t_a) * 1000)         
        seg_infer.append((t_c - t_b) * 1000)
        seg_post.append((t_d - t_c) * 1000)
        total_times.append((t_d - t_a) * 1000)

    import json, statistics as st
    print(json.dumps({
        "pre_ms":   seg_pre,
        "infer_ms": seg_infer,
        "post_ms":  seg_post,
        "avg_ms":   st.mean(total_times)
    }))
    
if __name__ == "__main__":
    main()
