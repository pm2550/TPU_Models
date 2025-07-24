#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU vs TPU 推理时间对比脚本 - 简化版本（无功耗比较）
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

def load_and_merge_results(results_dir="./results"):
    """
    加载CPU和TPU的测试结果并合并
    """
    cpu_file = os.path.join(results_dir, "cpu_benchmark_results.csv")
    tpu_file = os.path.join(results_dir, "tpu_benchmark_results.csv")
    
    try:
        cpu_df = pd.read_csv(cpu_file)
        cpu_df['platform'] = 'CPU'
        print(f"Loaded CPU results: {len(cpu_df)} entries from {cpu_file}")
    except FileNotFoundError:
        print(f"CPU results not found at {cpu_file}")
        cpu_df = pd.DataFrame()
    
    try:
        tpu_df = pd.read_csv(tpu_file)
        tpu_df['platform'] = 'TPU'
        print(f"Loaded TPU results: {len(tpu_df)} entries from {tpu_file}")
    except FileNotFoundError:
        print(f"TPU results not found at {tpu_file}")
        tpu_df = pd.DataFrame()
    
    if cpu_df.empty and tpu_df.empty:
        print("No results found!")
        return pd.DataFrame()
    
    # 合并数据
    combined_df = pd.concat([cpu_df, tpu_df], ignore_index=True)
    return combined_df

def create_inference_time_comparison(df, common_layers, cpu_times, tpu_times, output_dir):
    """
    创建推理时间对比图（分开的第一张图）
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 推理时间对比
    x = np.arange(len(common_layers))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU', alpha=0.8, color='skyblue', edgecolor='blue')
    bars2 = ax.bar(x + width/2, tpu_times, width, label='TPU', alpha=0.8, color='orange', edgecolor='darkorange')
    
    ax.set_xlabel("Layer Type", fontsize=12)
    ax.set_ylabel("Inference Time (ms)", fontsize=12)
    ax.set_title("Inference Time Comparison: CPU vs TPU", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_layers, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(cpu_times) * 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(cpu_times) * 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cpu_vs_tpu_inference_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ 推理时间对比图已保存到: {output_path}")

def create_speedup_comparison(df, common_layers, cpu_times, tpu_times, output_dir):
    """
    创建加速比对比图（分开的第二张图）
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 计算加速比
    speedup = [cpu/tpu if tpu > 0 else 0 for cpu, tpu in zip(cpu_times, tpu_times)]
    
    # 根据加速比大小设置颜色
    colors = ['green' if s > 1 else 'red' if s < 1 else 'gray' for s in speedup]
    
    bars = ax.bar(common_layers, speedup, alpha=0.8, color=colors, edgecolor='black')
    
    ax.set_xlabel("Layer Type", fontsize=12)
    ax.set_ylabel("Speedup (CPU Time / TPU Time)", fontsize=12)
    ax.set_title("TPU Speedup over CPU", fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.8, linewidth=2, label='No speedup (1x)')
    ax.legend(fontsize=11)
    
    # 添加数值标签
    for bar, speedup_val in zip(bars, speedup):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(speedup) * 0.02,
                f'{speedup_val:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cpu_vs_tpu_speedup.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    print(f"✅ 加速比对比图已保存到: {output_path}")
    
    return speedup

def create_comparison_plots(df, output_dir):
    """
    创建CPU vs TPU对比图表（分开的两张图）
    """
    if df.empty:
        print("No data to plot!")
        return
    
    # 获取共同的层类型
    cpu_layers = set(df[df['platform'] == 'CPU']['layer_type'])
    tpu_layers = set(df[df['platform'] == 'TPU']['layer_type'])
    common_layers = sorted(cpu_layers.intersection(tpu_layers))
    
    if not common_layers:
        print("No common layers found for comparison!")
        print(f"CPU layers: {cpu_layers}")
        print(f"TPU layers: {tpu_layers}")
        return
    
    # 准备对比数据
    cpu_times = []
    tpu_times = []
    
    for layer in common_layers:
        cpu_data = df[(df['platform'] == 'CPU') & (df['layer_type'] == layer)]
        tpu_data = df[(df['platform'] == 'TPU') & (df['layer_type'] == layer)]
        
        if not cpu_data.empty and not tpu_data.empty:
            cpu_times.append(cpu_data['avg_time_ms'].iloc[0])
            tpu_times.append(tpu_data['avg_time_ms'].iloc[0])
    
    # 确保输出文件夹存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建两张分开的图
    print("生成推理时间对比图...")
    create_inference_time_comparison(df, common_layers, cpu_times, tpu_times, output_dir)
    
    print("生成加速比对比图...")
    speedup = create_speedup_comparison(df, common_layers, cpu_times, tpu_times, output_dir)
    
    # 打印统计摘要
    print("\n" + "="*80)
    print("COMPARISON SUMMARY (Inference Time Only)")
    print("="*80)
    
    for i, layer in enumerate(common_layers):
        print(f"\n{layer}:")
        print(f"  CPU: {cpu_times[i]:8.2f} ms")
        print(f"  TPU: {tpu_times[i]:8.2f} ms")
        print(f"  Speedup: {speedup[i]:6.2f}x")
        
        if speedup[i] > 1:
            print(f"  🚀 TPU is {speedup[i]:.2f}x faster")
        elif speedup[i] < 1:
            print(f"  ⚠️  CPU is {1/speedup[i]:.2f}x faster")
        else:
            print(f"  ⚖️  Similar performance")

def main():
    # 检查命令行参数
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
        output_dir = results_dir
    else:
        results_dir = "./results"
        output_dir = "./results"
    
    print(f"Loading benchmark results from: {results_dir}")
    df = load_and_merge_results(results_dir)
    
    if not df.empty:
        print(f"Found {len(df)} total results")
        print("Creating comparison plots...")
        create_comparison_plots(df, output_dir)
        print("Comparison completed!")
        print(f"Check '{output_dir}/cpu_vs_tpu_inference_time.png' and '{output_dir}/cpu_vs_tpu_speedup.png'")
    else:
        print("No data available for comparison")

if __name__ == "__main__":
    main()