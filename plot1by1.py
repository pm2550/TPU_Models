#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import subprocess
import psutil
import time
import threading
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ---------------------
# 1. temperature reading function
# ---------------------
def read_cpu_temp():
    """
    Attempts to read CPU temperature (in °C).
    If unavailable, returns None.
    """
    try:
        # Method 1: read from /sys/class/thermal (common on RPi or similar)
        # with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
        #     temp_str = f.read().strip()
        #     return float(temp_str) / 1000.0

        # Method 2: use psutil.sensors_temperatures() (depends on platform support)
        temps = psutil.sensors_temperatures()
        for name, entries in temps.items():
            for entry in entries:
                if entry.current is not None:
                    return entry.current
        return None
    except Exception:
        return None

# ---------------------
# 2. system monitoring functions
# ---------------------
def monitor_system_info(interval=1.0, stop_event=None, output_list=None):
    """
    Continuously monitors CPU/memory/IO/temperature and appends data to output_list.
    Sampling every 'interval' seconds, stops if stop_event is set.
    """
    if output_list is None:
        output_list = []
    while True:
        if stop_event and stop_event.is_set():
            break
        cpu_percent = psutil.cpu_percent(interval=None)
        mem_info = psutil.virtual_memory()
        swap_info = psutil.swap_memory()
        disk_io = psutil.disk_io_counters()
        temp = read_cpu_temp()

        sample_time = time.time()
        data_point = {
            "time": sample_time,
            "cpu_percent": cpu_percent,
            "mem_percent": mem_info.percent,
            "swap_percent": swap_info.percent,
            "disk_read_MB": disk_io.read_bytes / 1024 / 1024,
            "disk_write_MB": disk_io.write_bytes / 1024 / 1024,
            "temp_celsius": temp
        }
        output_list.append(data_point)

        time.sleep(interval)

# ---------------------
# 3. inference functions
# ---------------------
from pycoral.utils.edgetpu import make_interpreter
from tflite_runtime.interpreter import Interpreter

def build_interpreter(model_path, use_tpu=False):
    if use_tpu:
        return make_interpreter(model_path)
    return Interpreter(model_path)


def prepare_interpreter(model_path, warmup=5,useTpu=False):
    # itp = build_interpreter(model_path, use_tpu=useTpu)
    itp=build_interpreter(model_path, use_tpu=useTpu)
    itp.allocate_tensors()

   
    inp_idx = itp.get_input_details()[0]['index']
    out_idx = itp.get_output_details()[0]['index']
    dummy   = np.random.random_sample(
                 itp.get_input_details()[0]['shape']
             ).astype(itp.get_input_details()[0]['dtype'])
    for _ in range(warmup):
        itp.set_tensor(inp_idx, dummy)
        itp.invoke()
        _ = itp.get_tensor(out_idx)
    return itp, inp_idx, out_idx, dummy

def run_once(interpreter, inp_idx, out_idx, dummy):
        t_a = time.perf_counter_ns()
        interpreter.set_tensor(inp_idx, dummy)
        t_b = time.perf_counter_ns()

        interpreter.invoke()
        t_c = time.perf_counter_ns()

        _ = interpreter.get_tensor(out_idx)
        t_d = time.perf_counter_ns()

        # 统一返回毫秒
        return ((t_b - t_a) / 1e6,
            (t_c - t_b) / 1e6,
            (t_d - t_c) / 1e6,
            (t_d - t_a) / 1e6)


# ---------------------
# 4. plotting functions
# ---------------------
def plot_monitor_data(data_list, output_dir="./results", output_prefix="monitor"):
    """
    Plots the time-series data of CPU/memory/IO/temperature.
    All outputs are saved in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(data_list)
    if df.empty:
        print("No system monitor data to plot.")
        return
    
    # Convert absolute timestamps to relative times
    t0 = df["time"].iloc[0]
    df["relative_time"] = df["time"] - t0

    output_prefix = os.path.join(output_dir, output_prefix)

    # 4.1 CPU Usage
    plt.figure(figsize=(8, 4))
    plt.plot(df["relative_time"], df["cpu_percent"], marker='o', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_cpu_usage.png")
    plt.close()

    # 4.2 Memory Usage
    plt.figure(figsize=(8, 4))
    plt.plot(df["relative_time"], df["mem_percent"], marker='o', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mem_usage.png")
    plt.close()
    
    # 4.3 Swap Usage
    plt.figure(figsize=(8, 4))
    plt.plot(df["relative_time"], df["swap_percent"], marker='o', linestyle='-')
    plt.xlabel("Time (s)")
    plt.ylabel("Swap Usage (%)")
    plt.title("Swap Usage Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_swap_usage.png")
    plt.close()

    # 4.4 Disk IO
    plt.figure(figsize=(8, 4))
    plt.plot(df["relative_time"], df["disk_read_MB"], marker='o', linestyle='-', label='Read (MB)')
    plt.plot(df["relative_time"], df["disk_write_MB"], marker='x', linestyle='-', label='Write (MB)')
    plt.xlabel("Time (s)")
    plt.ylabel("Disk IO (MB)")
    plt.title("Disk IO Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_disk_io.png")
    plt.close()

    # 4.5 Temperature (if available)
    if df["temp_celsius"].notnull().any():
        plt.figure(figsize=(8, 4))
        plt.plot(df["relative_time"], df["temp_celsius"], marker='o', linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("CPU Temperature Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_temperature.png")
        plt.close()

    # Save the entire monitor data to CSV
    df.to_csv(f"{output_prefix}_stats.csv", index=False)
    print(f"System monitor data and charts have been saved to {output_prefix}_*.png / CSV.")

def plot_inference_hist(times_list, output_dir="./results", output_prefix="mobilenet"):
    """
    Plots a histogram of the inference times. All outputs are saved in output_dir.
    """
    if not times_list:
        print("No inference times to plot histogram.")
        return

    os.makedirs(output_dir, exist_ok=True)
    output_prefix = os.path.join(output_dir, output_prefix)

    plt.figure(figsize=(8, 4))
    plt.hist(times_list, bins='auto', alpha=0.7, edgecolor='black')
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Number of Inferences")
    plt.title("Mobilenet Inference Time Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_inference_time_hist.png")
    plt.close()

    # Also save times to a CSV for reference
    df_times = pd.DataFrame({"inference_time_ms": times_list})
    df_times.to_csv(f"{output_prefix}_inference_times.csv", index=False)
    print(f"Inference time histogram and CSV saved: {output_prefix}_inference_time_hist.png / CSV")


def plot_segments(pre, infer_, post, output_dir="./results", prefix="mobilenet"):
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, prefix)
    df = pd.DataFrame({
        "pre_ms": pre, "infer_ms": infer_, "post_ms": post
    })
    df.to_csv(f"{prefix}_segments.csv", index=False)

    # 折线
    plt.figure(figsize=(8,4))
    plt.plot(df.index, df["pre_ms"],  label="A→B  (load)")
    plt.plot(df.index, df["infer_ms"],label="B→C  (infer)")
    plt.plot(df.index, df["post_ms"], label="C→D  (save)")
    plt.xlabel("Inference #"); plt.ylabel("Time (ms)")
    plt.title("Per‑Inference Segment Times")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{prefix}_segments_line.png"); plt.close()


# ---------------------
# 5. Main Entry Point
# ---------------------
def main():
    model_path1 = "./model/mobilenet.tflite"
    model_path2 = "./model/mobilenet2.tflite"
    # model_path = "./model/mobilenet_cpu2.tflite"
    # model_path1 = "./model/test for cache/7m.tflite"
    # model_path2 = "./model/test for cache/mn7.tflite"
    # model_path2="./edgetpu/test_data/inception_v1_224_quant_edgetpu.tflite"
    num_runs   = 1000

    # 5-1 Start system monitoring
    monitor_data = []
    stop_event   = threading.Event()
    monitor_thread=threading.Thread(
        target=monitor_system_info,
        args=(0.01, stop_event, monitor_data),
        daemon=True
    )
    monitor_thread.start()

    # 5-2 Inference loop
    pre1_list, infer1_list, post1_list, total1_list = [], [], [], []
    pre2_list, infer2_list, post2_list, total2_list = [], [], [], []

    # 获取模型名用于文件夹
    model1_name = os.path.splitext(os.path.basename(model_path1))[0]
    model2_name = os.path.splitext(os.path.basename(model_path2))[0]
    outdir1 = os.path.join("./results", model1_name)
    outdir2 = os.path.join("./results", model2_name)
    os.makedirs(outdir1, exist_ok=True)
    os.makedirs(outdir2, exist_ok=True)

    print(f"Running {num_runs} inferences on {model_path1} and {model_path2} ...")
    itp1, inp1, out1, dummy1 = prepare_interpreter(model_path1, warmup=5, useTpu=True)
    itp2, inp2, out2, dummy2 = prepare_interpreter(model_path2, warmup=5, useTpu=True)
    for i in range(num_runs):
        if i % 2 == 0:
            t_pre, t_inf, t_post, t_tot = run_once(itp1, inp1, out1, dummy1)
            pre1_list.append(t_pre)
            infer1_list.append(t_inf)
            post1_list.append(t_post)
            total1_list.append(t_tot)
        else:
            t_pre, t_inf, t_post, t_tot = run_once(itp2, inp2, out2, dummy2)
            pre2_list.append(t_pre)
            infer2_list.append(t_inf)
            post2_list.append(t_post)
            total2_list.append(t_tot)

    print("Inferences finished.")

    # 5-3 Plot results and stop monitoring
    stop_event.set()
    monitor_thread.join()

    # 打印平均推理时长（ms）
    if total1_list:
        avg_total1 = float(np.mean(total1_list))
    else:
        avg_total1 = float('nan')
    if total2_list:
        avg_total2 = float(np.mean(total2_list))
    else:
        avg_total2 = float('nan')
    print(f"Average per-inference latency (ms): {model1_name}: {avg_total1:.3f}, {model2_name}: {avg_total2:.3f}")

    plot_monitor_data(monitor_data, outdir1, f"{model1_name}_monitor")
    plot_inference_hist(total1_list, outdir1, model1_name)
    plot_segments(pre1_list, infer1_list, post1_list, outdir1, model1_name)
    plot_monitor_data(monitor_data, outdir2, f"{model2_name}_monitor")
    plot_inference_hist(total2_list, outdir2, model2_name)
    plot_segments(pre2_list, infer2_list, post2_list, outdir2, model2_name)


if __name__ == "__main__":
    main()
