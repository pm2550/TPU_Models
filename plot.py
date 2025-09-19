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


def prepare_interpreter(model_path, warmup=5, useTpu=False):
    itp = build_interpreter(model_path, use_tpu=useTpu)
    itp.allocate_tensors()
    
    # Get input and output tensor information
    input_details = itp.get_input_details()[0]
    output_details = itp.get_output_details()[0]
    inp_idx = input_details['index']
    out_idx = output_details['index']
    
    # Create appropriate dummy data based on data type
    input_shape = input_details['shape']
    input_dtype = input_details['dtype']
    
    if input_dtype == np.uint8:
        dummy = np.random.randint(0, 256, input_shape, dtype=np.uint8)
    else:
        dummy = np.random.random_sample(input_shape).astype(input_dtype)
    
    # Warmup runs - record first invoke time (cold start)
    first_invoke_time = None
    for i in range(warmup):
        if i == 0:
            # Record first invoke time (cold start)
            start_time = time.perf_counter_ns()
            itp.set_tensor(inp_idx, dummy)
            itp.invoke()
            end_time = time.perf_counter_ns()
            first_invoke_time = (end_time - start_time) / 1e6  # Convert to milliseconds
            _ = itp.get_tensor(out_idx)
            print(f"  First warmup: {first_invoke_time:.2f} ms (cold start)")
        else:
            itp.set_tensor(inp_idx, dummy)
            if i == warmup - 1:
                # Record last warmup time
                start_time = time.perf_counter_ns()
                itp.invoke()
                end_time = time.perf_counter_ns()
                last_warmup_time = (end_time - start_time) / 1e6
                _ = itp.get_tensor(out_idx)
                print(f"  Last warmup: {last_warmup_time:.2f} ms (warmed up)")
            else:
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

        # Convert nanoseconds to milliseconds
        return ((t_b - t_a) / 1e6,
            (t_c - t_b) / 1e6,
            (t_d - t_c) / 1e6,
            (t_d - t_a) / 1e6)


# ---------------------
# 4. plotting functions
# ---------------------
def plot_monitor_data(data_list, output_prefix="monitor"):
    """
    Plots the time-series data of CPU/memory/IO/temperature.
    """
    df = pd.DataFrame(data_list)
    if df.empty:
        print("No system monitor data to plot.")
        return
    
    output_prefix = f"./results/{output_prefix}"
    
    # Convert absolute timestamps to relative times
    t0 = df["time"].iloc[0]
    df["relative_time"] = df["time"] - t0

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

def plot_inference_hist(times_list, output_prefix="mobilenet"):
    """
    Plots a histogram of the inference times.
    X-axis: Inference time (seconds)
    Y-axis: Number of inferences
    """
    if not times_list:
        print("No inference times to plot histogram.")
        return

    output_prefix = f"./results/{output_prefix}"

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


def plot_segments(pre, infer_, post, prefix="mobilenet"):
    prefix = f"./results/{prefix}"
    df = pd.DataFrame({
        "pre_ms": pre, "infer_ms": infer_, "post_ms": post
    })
    df.to_csv(f"{prefix}_segments.csv", index=False)

    # Plot the segments
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
    model_path = "./model/mobilenet.tflite"
    # model_path = "./models_local/public/inceptionv3_8seg_uniform_local/combos_K2_run1/tpu/seg1_int8_edgetpu.tflite"
    # model_path = "./model/mobilenet+.tflite"
    # model_path = "./layered models/mn/tpu/mnv2_224_layer1_int8_edgetpu.tflite"
    # model_path = "./layered models/mn/tpu/mnv2_224_layer1_tiny_int8_edgetpu.tflite"
    # model_path = "./layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite"
    # model_path = "./edgetpu/test_data/inception_v1_224_quant_edgetpu.tflite"
    # model_path="./inception_v1_224_quant_cpu.tflite"
    # model_path= "./tpu/conv2d_tpu.tflite"
    # model_path="./tpu/relu_tpu.tflite"
    # num_runs   = 1000
    num_runs   = 1000

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    # 5-1 Start system monitoring
    monitor_data = []
    stop_event   = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_system_info,
        args=(0.01, stop_event, monitor_data),
        daemon=True
    )
    monitor_thread.start()

    # 5-2 Prepare interpreter once (not in the loop!)
    print(f"Preparing TPU interpreter for {model_path}...")
    try:
        itp, inp_idx, out_idx, dummy = prepare_interpreter(model_path, warmup=5, useTpu=True)
        print("TPU interpreter ready.")
    except Exception as e:
        print(f"Error preparing TPU interpreter: {e}")
        stop_event.set()
        monitor_thread.join()
        return

    # 5-3 Inference loop
    pre_list, infer_list, post_list, total_list = [], [], [], []
    invoke_list = []  # Track pure invoke time (inference only)
    print(f"Running {num_runs} inferences...")
    
    # Start timing for throughput calculation
    inference_start_time = time.time()
    
    for i in range(num_runs):
        if i % 100 == 0:
            print(f"Progress: {i}/{num_runs}")
        
        try:
            t_pre, t_inf, t_post, t_tot = run_once(itp, inp_idx, out_idx, dummy)
            pre_list.append(t_pre)
            infer_list.append(t_inf)
            post_list.append(t_post)
            total_list.append(t_tot)
            invoke_list.append(t_inf)  # Store pure inference time as invoke time
        except Exception as e:
            print(f"Error during inference {i}: {e}")
            continue

    print("Inferences finished.")
    
    # Calculate comprehensive statistics
    inference_duration = time.time() - inference_start_time
    
    # Invoke time statistics (pure inference time)
    avg_invoke_time = np.mean(invoke_list)
    std_invoke_time = np.std(invoke_list)
    min_invoke_time = np.min(invoke_list)
    max_invoke_time = np.max(invoke_list)
    
    # Total time statistics (including pre/post processing)
    avg_total_time = np.mean(total_list)
    std_total_time = np.std(total_list)
    min_total_time = np.min(total_list)
    max_total_time = np.max(total_list)
    
    throughput = len(total_list) / inference_duration if total_list else 0
    
    print(f"  Total runs: {len(total_list)}")
    print(f"  Total duration: {inference_duration:.2f} seconds")
    print(f"  Average invoke time: {avg_invoke_time:.2f} ± {std_invoke_time:.2f} ms")
    print(f"  Average total time: {avg_total_time:.2f} ± {std_total_time:.2f} ms")
    print(f"  Min invoke time: {min_invoke_time:.2f} ms")
    print(f"  Max invoke time: {max_invoke_time:.2f} ms")
    print(f"  Throughput: {throughput:.2f} inferences/second")
    
    print(f"\nCOMPARISON:")
    print(f"  Average invoke time difference: 0.00%")
    print(f"  TPU 0 variability (CV): {std_invoke_time/avg_invoke_time*100:.2f}%")

    # 5-4 Plot results and stop monitoring
    stop_event.set()
    monitor_thread.join()

    # Create results directory if it doesn't exist
    os.makedirs("./results", exist_ok=True)

    plot_monitor_data(monitor_data, "mobilenet_monitor")
    plot_inference_hist(total_list, "mobilenet")
    plot_segments(pre_list, infer_list, post_list, "mobilenet")


if __name__ == "__main__":
    main()
