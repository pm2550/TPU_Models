#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time
from typing import List, Tuple, Dict, Any

import numpy as np


def _make_interpreter(model_path: str, use_tpu: bool):
    if use_tpu:
        try:
            from pycoral.utils.edgetpu import make_interpreter
            return make_interpreter(model_path)
        except Exception as e:
            # Fallback to CPU if Edge TPU stack is not available in this venv
            print(f"[WARN] Edge TPU via pycoral unavailable ({e}); trying tflite_runtime delegate...", file=sys.stderr)
            try:
                from tflite_runtime.interpreter import Interpreter, load_delegate
                return Interpreter(model_path, experimental_delegates=[load_delegate('libedgetpu.so.1')])
            except Exception as e2:
                print(f"[WARN] Edge TPU via tflite_runtime failed ({e2}); falling back to CPU.", file=sys.stderr)
    from tflite_runtime.interpreter import Interpreter
    return Interpreter(model_path)


def _prepare(model_path: str, warmup: int, use_tpu: bool):
    itp = _make_interpreter(model_path, use_tpu)
    itp.allocate_tensors()
    inp = itp.get_input_details()[0]
    out = itp.get_output_details()[0]
    inp_idx = inp["index"]
    out_idx = out["index"]
    shape = inp["shape"]
    dtype = inp["dtype"]
    if np.issubdtype(dtype, np.integer):
        # uint8/int8 inputs typical for Edge TPU models
        info = itp.get_input_details()[0]
        # Prefer random full-range integers to avoid constant folding side-effects
        if dtype == np.uint8:
            dummy = np.random.randint(0, 256, size=shape, dtype=np.uint8)
        elif dtype == np.int8:
            dummy = np.random.randint(-128, 128, size=shape, dtype=np.int8)
        else:
            dummy = np.zeros(shape, dtype=dtype)
    else:
        dummy = np.random.random_sample(shape).astype(dtype)

    for _ in range(max(0, warmup)):
        itp.set_tensor(inp_idx, dummy)
        itp.invoke()
        _ = itp.get_tensor(out_idx)

    return itp, inp_idx, out_idx, dummy


def _run_invoke_only(itp, inp_idx: int, out_idx: int, dummy: np.ndarray) -> float:
    # 仅统计 invoke() 时间
    itp.set_tensor(inp_idx, dummy)
    t0 = time.perf_counter()
    itp.invoke()
    t1 = time.perf_counter()
    _ = itp.get_tensor(out_idx)
    return (t1 - t0) * 1000.0


def benchmark_single(target_model: str, reference_model: str, runs: int, warmup: int, use_tpu: bool, invoke_ref_each_iter: bool = False) -> Dict[str, Any]:
    # Keep reference resident (co-compiled). Optionally invoke each iteration.
    ref_itp, ref_inp, ref_out, ref_dummy = _prepare(reference_model, warmup=warmup, use_tpu=use_tpu)

    itp, inp_idx, out_idx, dummy = _prepare(target_model, warmup=warmup, use_tpu=use_tpu)

    inf_list = []
    for _ in range(runs):
        if invoke_ref_each_iter:
            _ = _run_invoke_only(ref_itp, ref_inp, ref_out, ref_dummy)
        inf = _run_invoke_only(itp, inp_idx, out_idx, dummy)
        inf_list.append(inf)

    return {
        "target": os.path.abspath(target_model),
        "reference": os.path.abspath(reference_model),
        "runs": runs,
        "invoke_ms": inf_list,
        "avg_invoke_ms": float(np.mean(inf_list)) if inf_list else None,
    }


def benchmark_partitioned(seg_files: List[str], reference_model: str, runs: int, warmup: int, use_tpu: bool, invoke_ref_each_iter: bool = False) -> Dict[str, Any]:
    ref_itp, ref_inp, ref_out, ref_dummy = _prepare(reference_model, warmup=warmup, use_tpu=use_tpu)

    results: Dict[str, Any] = {"reference": os.path.abspath(reference_model), "runs": runs, "segments": []}
    for seg in seg_files:
        itp, inp_idx, out_idx, dummy = _prepare(seg, warmup=warmup, use_tpu=use_tpu)
        inf_list = []
        for _ in range(runs):
            if invoke_ref_each_iter:
                _ = _run_invoke_only(ref_itp, ref_inp, ref_out, ref_dummy)
            inf = _run_invoke_only(itp, inp_idx, out_idx, dummy)
            inf_list.append(inf)
        results["segments"].append({
            "segment": os.path.abspath(seg),
            "invoke_ms": inf_list,
            "avg_invoke_ms": float(np.mean(inf_list)) if inf_list else None,
        })
    return results


def discover_segments(seg_dir: str) -> List[str]:
    # Prefer canonical naming seg0..seg3; fallback to any .tflite in order by name
    candidates = [os.path.join(seg_dir, f"seg{i}.tflite") for i in range(4)]
    segs = [p for p in candidates if os.path.exists(p)]
    if len(segs) == 4:
        return segs
    # Fallback: any tflite, sorted
    all_tflite = [os.path.join(seg_dir, f) for f in os.listdir(seg_dir) if f.endswith('.tflite')]
    all_tflite.sort()
    return all_tflite


def main():
    parser = argparse.ArgumentParser(description="Benchmark co-compiled TFLite models on Edge TPU (invoke time only)")
    parser.add_argument("--reference", required=True, help="参考模型路径 (保持常驻以模拟共编译)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--target", help="整模型路径 (Scheme 1)")
    mode.add_argument("--seg-dir", help="子图目录，包含 seg0..seg3 (Scheme 2)")
    parser.add_argument("--runs", type=int, default=200, help="计时次数")
    parser.add_argument("--warmup", type=int, default=10, help="热身次数")
    # 强制使用 Edge TPU；不提供 CPU 回退
    parser.add_argument("--invoke-ref", action="store_true", help="每轮先运行一次参考模型")
    args = parser.parse_args()

    use_tpu = True

    try:
        if args.target:
            result = benchmark_single(args.target, args.reference, args.runs, args.warmup, use_tpu, args.invoke_ref)
        else:
            segs = discover_segments(args.seg_dir)
            if not segs:
                raise RuntimeError(f"未在 {args.seg_dir} 发现任何 .tflite 子图")
            result = benchmark_partitioned(segs, args.reference, args.runs, args.warmup, use_tpu, args.invoke_ref)
        print(json.dumps(result))
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


