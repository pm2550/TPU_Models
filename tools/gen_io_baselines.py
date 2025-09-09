#!/usr/bin/env python3
import os
import json
import glob
from pathlib import Path
import subprocess
import sys


MODELS_BASE = "/home/10210/Desktop/OS/models_local/public"
OUTPUT_DIR = "/home/10210/Desktop/OS/baselines"

# 如需限制模型范围，可在此列出；默认扫描目录下 *_8seg_uniform_local
DEFAULT_MODELS = None  # e.g. ["resnet50_8seg_uniform_local", ...]


VENV_PY = "/home/10210/Desktop/OS/.venv/bin/python"


def get_io_bytes(model_path: str):
    """返回 (in_bytes, out_bytes) 或抛出异常。逐步尝试多种加载方式。"""
    last_err = None
    # 为了稳定性，使用虚拟环境子进程读取（可安全 allocate_tensors）
    py = VENV_PY if os.path.exists(VENV_PY) else sys.executable
    code = r'''
import json, sys
mp = sys.argv[1]
def nbytes(shape, dtype):
    n=1
    for d in shape:
        n*=int(d)
    s=str(dtype)
    if s.endswith('int8') or s.endswith('uint8'):
        bs=1
    elif s.endswith('float32'):
        bs=4
    elif s.endswith('int16') or s.endswith('uint16'):
        bs=2
    elif s.endswith('int32') or s.endswith('uint32'):
        bs=4
    elif s.endswith('int64') or s.endswith('uint64'):
        bs=8
    else:
        bs=1
    return int(n)*bs
def try_pycoral():
    try:
        from pycoral.utils.edgetpu import make_interpreter
        it=make_interpreter(mp)
        it.allocate_tensors()
        inp=it.get_input_details()[0]
        outp=it.get_output_details()[0]
        print(json.dumps({'in': nbytes(inp['shape'], inp['dtype']), 'out': nbytes(outp['shape'], outp['dtype'])}))
        return True
    except Exception:
        return False
def try_tfl_delegate():
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        it=Interpreter(model_path=mp, experimental_delegates=[load_delegate('libedgetpu.so.1')])
        it.allocate_tensors()
        inp=it.get_input_details()[0]
        outp=it.get_output_details()[0]
        print(json.dumps({'in': nbytes(inp['shape'], inp['dtype']), 'out': nbytes(outp['shape'], outp['dtype'])}))
        return True
    except Exception:
        return False
def try_tfl():
    try:
        from tflite_runtime.interpreter import Interpreter
        it=Interpreter(model_path=mp)
        it.allocate_tensors()
        inp=it.get_input_details()[0]
        outp=it.get_output_details()[0]
        print(json.dumps({'in': nbytes(inp['shape'], inp['dtype']), 'out': nbytes(outp['shape'], outp['dtype'])}))
        return True
    except Exception:
        return False
if not (try_pycoral() or try_tfl_delegate() or try_tfl()):
    print(json.dumps({'error': 'failed'}))
'''
    try:
        res = subprocess.run([py, '-c', code, model_path], capture_output=True, text=True, timeout=30)
        data = json.loads((res.stdout or '').strip() or '{}')
        if 'in' in data and 'out' in data:
            return int(data['in']), int(data['out'])
        raise RuntimeError(data.get('error') or (res.stderr[-200:] if res.stderr else 'unknown'))
    except Exception as e:
        raise RuntimeError(f"failed to read IO bytes: {e}")


def dtype_itemsize(dtype_obj) -> int:
    # 兼容 pycoral/tflite_runtime 中 dtype 字段，不依赖 numpy
    s = str(dtype_obj)
    # 常见类型映射
    if s.endswith('int8') or s.endswith('uint8'):
        return 1
    if s.endswith('float32'):
        return 4
    if s.endswith('int16') or s.endswith('uint16'):
        return 2
    if s.endswith('int32') or s.endswith('uint32'):
        return 4
    if s.endswith('int64') or s.endswith('uint64'):
        return 8
    # 默认按1字节处理，避免中断
    return 1


def num_elements(shape) -> int:
    n = 1
    for d in list(shape):
        try:
            n *= int(d)
        except Exception:
            return 0
    return int(n)


def find_seg_model_any(base_dir: str, seg_num: int):
    pats = [
        os.path.join(base_dir, f"seg{seg_num}_*_edgetpu.tflite"),
        os.path.join(base_dir, f"seg{seg_num}_int8_edgetpu.tflite"),
        os.path.join(base_dir, f"seg{seg_num}_*.tflite"),
        os.path.join(base_dir, f"seg{seg_num}.tflite"),
    ]
    for pat in pats:
        cands = sorted(glob.glob(pat))
        if cands:
            return cands[0]
    return None


def collect_standard_models():
    out: dict[str, dict] = {}
    models = (
        DEFAULT_MODELS
        if DEFAULT_MODELS is not None
        else [d for d in os.listdir(MODELS_BASE) if d.endswith('_8seg_uniform_local')]
    )
    for name in sorted(models):
        root_dir = os.path.join(MODELS_BASE, name, 'full_split_pipeline_local')
        if not os.path.isdir(root_dir):
            continue
        tfl_dir = os.path.join(root_dir, 'tflite')
        tpu_dir = os.path.join(root_dir, 'tpu')
        segs: dict[str, dict] = {}
        for i in range(1, 9):
            # 先找 tflite，再回退 tpu
            mp = None
            if os.path.isdir(tfl_dir):
                mp = find_seg_model_any(tfl_dir, i)
            if not mp and os.path.isdir(tpu_dir):
                mp = find_seg_model_any(tpu_dir, i)
            if not mp:
                continue
            try:
                in_bytes, out_bytes = get_io_bytes(mp)
                segs[f'seg{i}'] = {
                    'model_path': mp,
                    'in_bytes': in_bytes,
                    'out_bytes': out_bytes,
                }
            except Exception as e:
                segs[f'seg{i}'] = {'model_path': mp, 'error': str(e)}
        if segs:
            out[name] = {'segments': segs}
    return out


def collect_combo_models():
    out: dict[str, dict] = {}
    models = (
        DEFAULT_MODELS
        if DEFAULT_MODELS is not None
        else [d for d in os.listdir(MODELS_BASE) if d.endswith('_8seg_uniform_local')]
    )
    for name in sorted(models):
        combos: dict[str, dict] = {}
        for k in range(2, 8):  # 只需 K=2..7
            root_dir = os.path.join(MODELS_BASE, name, f'combos_K{k}_run1')
            if not os.path.isdir(root_dir):
                continue
            tfl_dir = os.path.join(root_dir, 'tflite')
            tpu_dir = os.path.join(root_dir, 'tpu')
            segs: dict[str, dict] = {}
            for i in range(1, 9):
                mp = None
                if os.path.isdir(tfl_dir):
                    mp = find_seg_model_any(tfl_dir, i)
                if not mp and os.path.isdir(tpu_dir):
                    mp = find_seg_model_any(tpu_dir, i)
                if not mp:
                    continue
                try:
                    in_bytes, out_bytes = get_io_bytes(mp)
                    segs[f'seg{i}'] = {
                        'model_path': mp,
                        'in_bytes': in_bytes,
                        'out_bytes': out_bytes,
                    }
                except Exception as e:
                    segs[f'seg{i}'] = {'model_path': mp, 'error': str(e)}
            if segs:
                combos[f'K{k}'] = {'segments': segs}
        if combos:
            out[name] = {'combos': combos}
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    std = collect_standard_models()
    combo = collect_combo_models()

    std_path = os.path.join(OUTPUT_DIR, 'models_io_baseline.json')
    combo_path = os.path.join(OUTPUT_DIR, 'combo_io_baseline.json')

    with open(std_path, 'w') as f:
        json.dump(std, f, indent=2, ensure_ascii=False)
    with open(combo_path, 'w') as f:
        json.dump(combo, f, indent=2, ensure_ascii=False)

    print(f"Wrote: {std_path}")
    print(f"Wrote: {combo_path}")


if __name__ == '__main__':
    main()


