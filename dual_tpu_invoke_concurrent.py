#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import numpy as np
from typing import Dict, Any, List
from multiprocessing import Process, Queue
import os

from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus


MODEL_PATH = './layered models/tpu/conv2d_3x3_stride2_edgetpu.tflite'


def build_device_strings() -> List[str]:
    """枚举 EdgeTPU 设备并构造成 libedgetpu 接受的 device 字符串。
    优先返回精确绑定（usb:N / pci:N）；若不可得则返回空列表（调用方可走不绑定）。
    """
    devs = list_edge_tpus()
    usb_count = 0
    pci_count = 0
    device_strings: List[str] = []
    for d in devs:
        dtype = (d.get('type') or '').upper()
        if dtype == 'USB':
            device_strings.append(f'usb:{usb_count}')
            usb_count += 1
        elif dtype == 'PCI':
            device_strings.append(f'pci:{pci_count}')
            pci_count += 1
    return device_strings


def run_worker(device: str, num_warmup: int, num_iters: int, out_q: Queue, cpu_id: int = None) -> None:
    result: Dict[str, Any] = {'device': device, 'ok': False}
    try:
        if cpu_id is not None and hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, {cpu_id})
                result['cpu_affinity'] = cpu_id
            except Exception as e:
                result['cpu_affinity_error'] = repr(e)
        it = make_interpreter(MODEL_PATH, device=device)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        # 构造固定输入
        if inp['dtype'].__name__ == 'uint8':
            x = np.random.randint(0, 256, inp['shape'], dtype=np.uint8)
        else:
            x = np.random.randint(-128, 128, inp['shape'], dtype=np.int8)

        # 预热
        for _ in range(num_warmup):
            it.set_tensor(inp['index'], x)
            it.invoke()

        times = []
        for _ in range(num_iters):
            it.set_tensor(inp['index'], x)
            t0 = time.perf_counter()
            it.invoke()  # 仅计 invoke
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)

        arr = np.array(times, dtype=np.float64)
        result.update({
            'ok': True,
            'count': int(arr.size),
            'mean_ms': float(arr.mean()),
            'p50_ms': float(np.percentile(arr, 50)),
            'p90_ms': float(np.percentile(arr, 90)),
            'min_ms': float(arr.min()),
            'max_ms': float(arr.max()),
        })
    except Exception as e:
        result['error'] = repr(e)
    finally:
        out_q.put(result)


def run_worker_unbound(num_warmup: int, num_iters: int, out_q: Queue, cpu_id: int = None) -> None:
    result: Dict[str, Any] = {'device': '(unbound)', 'ok': False}
    try:
        if cpu_id is not None and hasattr(os, 'sched_setaffinity'):
            try:
                os.sched_setaffinity(0, {cpu_id})
                result['cpu_affinity'] = cpu_id
            except Exception as e:
                result['cpu_affinity_error'] = repr(e)
        it = make_interpreter(MODEL_PATH)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        if inp['dtype'].__name__ == 'uint8':
            x = np.random.randint(0, 256, inp['shape'], dtype=np.uint8)
        else:
            x = np.random.randint(-128, 128, inp['shape'], dtype=np.int8)
        for _ in range(num_warmup):
            it.set_tensor(inp['index'], x)
            it.invoke()
        times = []
        for _ in range(num_iters):
            it.set_tensor(inp['index'], x)
            t0 = time.perf_counter()
            it.invoke()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        result.update({
            'ok': True,
            'count': int(arr.size),
            'mean_ms': float(arr.mean()),
            'p50_ms': float(np.percentile(arr, 50)),
            'p90_ms': float(np.percentile(arr, 90)),
            'min_ms': float(arr.min()),
            'max_ms': float(arr.max()),
        })
    except Exception as e:
        result['error'] = repr(e)
    finally:
        out_q.put(result)


def main():
    # 两个设备；如不存在则会在子进程里报错
    devices = build_device_strings()
    meta: Dict[str, Any] = {
        'model': MODEL_PATH,
        'enum_devices': devices,
    }
    num_warmup = 50
    num_iters = 500

    q = Queue()
    procs: List[Process] = []
    # 选择两个 CPU 核心（若可用）
    try:
        nproc = os.cpu_count() or 2
    except Exception:
        nproc = 2
    cpu_pair = (0, 1) if nproc >= 2 else (0, 0)

    if len(devices) >= 2:
        for dev, cpu_id in zip(devices[:2], cpu_pair):
            procs.append(Process(target=run_worker, args=(dev, num_warmup, num_iters, q, cpu_id)))
    else:
        procs.append(Process(target=run_worker_unbound, args=(num_warmup, num_iters, q, cpu_pair[0])))
        procs.append(Process(target=run_worker_unbound, args=(num_warmup, num_iters, q, cpu_pair[1])))
        meta['fallback_unbound'] = True
    for p in procs:
        p.start()
    results = [q.get() for _ in procs]
    for p in procs:
        p.join()

    print(json.dumps({'meta': meta, 'results': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


