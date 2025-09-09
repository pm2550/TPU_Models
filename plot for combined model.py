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
# import json
# def run_inference_once(model_path):
#     cmd = ["python3", "run.py", "--model", model_path,
#            "--warmup", "0", "--runs", "1"]

#     # cmd = ["python3", "run.py", "--model", model_path,
#     #    "--cpu",                     
#     #    "--warmup", "0", "--runs", "1"]

#     res = subprocess.run(cmd, capture_output=True, text=True)
#     if res.returncode != 0:
#         raise RuntimeError(f"Inference failed:\n{res.stderr}")  
#     jtxt = res.stdout.strip()
#     if not jtxt:
#         raise RuntimeError(f"run.py returned empty stdout:\n{res.stderr}")
#     try:
#             j = json.loads(jtxt)
#     except json.JSONDecodeError as e:
#             raise RuntimeError(f"run.py stdout is not JSON:\n{jtxt}\n\nstderr:\n{res.stderr}") from e

#     j = json.loads(res.stdout.strip())
#     return (j["pre_ms"][0], j["infer_ms"][0],
#             j["post_ms"][0], j["avg_ms"])      

from pycoral.utils.edgetpu import make_interpreter
from tflite_runtime.interpreter import Interpreter

def build_interpreter(model_path, use_tpu=True):
    if use_tpu:
        return make_interpreter(model_path)
    return Interpreter(model_path)

def prepare_interpreter(model_path, warmup=5, use_tpu=True):
    itp = build_interpreter(model_path, use_tpu)
    itp.allocate_tensors()

    # itp = Interpreter(model_path)
    # itp.allocate_tensors()
    inp_details  = itp.get_input_details()
    out_details  = itp.get_output_details()
    inp_indices  = [d['index'] for d in inp_details]     # e.g. [0,1]
    out_indices  = [d['index'] for d in out_details]     # e.g. [24,25]

  
    h, w, _ = inp_details[0]['shape'][1:4]
    dummy_u8 = np.zeros((1, h, w, 3), dtype=np.uint8)

   # warmup inference
    for _ in range(warmup):
        for idx in inp_indices:
            itp.set_tensor(idx, dummy_u8)
        itp.invoke()

    return itp, inp_indices, out_indices, dummy_u8

def set_two_inputs(itp, inp_indices, img_u8):
   
    for idx in inp_indices:
        # scale, zp = itp.get_input_details()[inp_indices.index(idx)]['quantization']
        # q_img = (img_u8 / scale + zp).astype(np.uint8)
        itp.set_tensor(idx, img_u8)
        
def run_once(itp, inp_indices, out_indices, dummy_u8):
    t0 = time.perf_counter_ns()

   
    for idx in inp_indices:
        itp.set_tensor(idx, dummy_u8)
    t1 = time.perf_counter_ns()


    itp.invoke()
    t2 = time.perf_counter_ns()


    for idx in out_indices:
        _ = itp.get_tensor(idx)       
    t3 = time.perf_counter_ns()


    return ((t1-t0)/1e6,
            (t2-t1)/1e6,
            (t3-t2)/1e6,
            (t3-t0)/1e6)

# ---------------------
# 4. plotting functions
# ---------------------
def plot_monitor_data(data_list, output_prefix="monitor"):
    """
    Plots the time-series data of CPU/memory/IO/temperature.
    """
    output_prefix = f"./results/{output_prefix}"
    df = pd.DataFrame(data_list)
    if df.empty:
        print("No system monitor data to plot.")
        return
    
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
    output_prefix = f"./results/{output_prefix}"
    if not times_list:
        print("No inference times to plot histogram.")
        return

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
    # model_path = "./model/mobilenet+.tflite"
    num_runs   = 500

    # 5-1 Start system monitoring
    monitor_data = []
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_system_info,
        args=(0.01, stop_event, monitor_data),
        daemon=True
    )
    monitor_thread.start()

    # 5-2 Inference loop
    pre_list, infer_list, post_list, total_list = [], [], [], []
    print(f"Running {num_runs} inferences on {model_path} ...")
    itp, inp_idx, out_idx, dummy = prepare_interpreter(model_path, warmup=5)
    for _ in range(num_runs):
        t_pre, t_inf, t_post, t_tot = run_once(itp, inp_idx, out_idx, dummy)
        pre_list.append(t_pre)
        infer_list.append(t_inf)
        post_list.append(t_post)
        total_list.append(t_tot)

    print("Inferences finished.")

    # 5-3 Plot results and stop monitoring
    stop_event.set()
    monitor_thread.join()

    plot_monitor_data(monitor_data, "mobilenet_monitor")
    plot_inference_hist(total_list, "mobilenet")
    plot_segments(pre_list, infer_list, post_list, "mobilenet")


if __name__ == "__main__":
    main()
