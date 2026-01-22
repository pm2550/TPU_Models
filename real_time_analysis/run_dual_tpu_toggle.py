#!/usr/bin/env python3
import argparse
import json
import multiprocessing as mp
import os
import random
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "dual_tpu_toggle"
PASSWORD_FILE = ROOT.parent / "password.text"
OFFLINE_ALIGN = ROOT.parent / "tools" / "offline_align_usbmon_ref.py"
ANALYZE_ACTIVE = ROOT.parent / "analyze_usbmon_active.py"
VENV_PY = ROOT.parent / ".venv" / "bin" / "python"

# Use CLOCK_BOOTTIME for timestamps so they can be aligned with usbmon.
def now_boottime():
    try:
        return time.clock_gettime(time.CLOCK_BOOTTIME)
    except Exception:
        # Fallback; relative durations still correct, but usbmon alignment may fail.
        return time.perf_counter()


def parse_args():
    parser = argparse.ArgumentParser(description="Two TPUs toggling between two models.")
    parser.add_argument("--mn4", type=Path, default=ROOT / "model" / "test for cache" / "mn4.tflite")
    parser.add_argument("--mn5", type=Path, default=ROOT / "model" / "test for cache" / "mn5.tflite")
    parser.add_argument("--mn7", type=Path, default=ROOT.parent / "model" / "test for cache" / "mn7.tflite")
    parser.add_argument("--7m", type=Path, default=ROOT.parent / "model" / "test for cache" / "7m.tflite")
    parser.add_argument("--deeplab", type=Path, default=ROOT.parent / "models_local" / "public" / "deeplabv3_mnv2_dm05_pascal_quant_edgetpu.tflite")
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Run a single model only (ignore --mn5) instead of toggling between two.",
    )
    parser.add_argument(
        "--mixed-models",
        action="store_true",
        help="TPU0 runs --deeplab, TPU1 runs --mn7 (requires --single-model).",
    )
    parser.add_argument(
        "--sync-start",
        action="store_true",
        help="Synchronize once before the first iteration so both workers start at the same time.",
    )
    parser.add_argument(
        "--sync-release",
        action="store_true",
        help="Synchronize on every release/invoke start across workers (lock-step per iteration).",
    )
    parser.add_argument("--iterations", type=int, default=50, help="How many times to run each model per TPU.")
    parser.add_argument("--cpu-cores", type=int, nargs="+", default=[2, 3], help="CPU cores to pin the two processes.")
    parser.add_argument("--devices", type=str, nargs="+", default=["usb:0", "usb:1"], help="EdgeTPU device ids per process.")
    parser.add_argument(
        "--sync-create",
        action="store_true",
        help="Synchronize between workers: start create together and wait until all creates finish before allocate/invoke.",
    )
    parser.add_argument(
        "--sync-all",
        action="store_true",
        help="Stronger sync: align create start, create end, allocate end, and invoke start across workers.",
    )
    parser.add_argument(
        "--align-first-create",
        action="store_true",
        help="Only align the very first create across workers; measurement will report wall time starting after the slowest first create.",
    )
    parser.add_argument(
        "--start-offset-ms",
        type=float,
        default=0.0,
        help="Sleep this many milliseconds before starting work in worker idx>0 (creates a fixed phase offset).",
    )
    parser.add_argument(
        "--per-iter-offset-ms",
        type=float,
        default=0.0,
        help="Sleep this many milliseconds before each create in worker idx>0 (forces per-iteration phase offset).",
    )
    parser.add_argument(
        "--reuse-interpreter",
        action="store_true",
        help="Create interpreters once per model and reuse across iterations (warm-state).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup invokes per model when --reuse-interpreter is set.",
    )
    parser.add_argument(
        "--read-output",
        action="store_true",
        help="Call get_tensor after invoke and include it in total_ms.",
    )
    parser.add_argument(
        "--period-ms",
        type=float,
        default=0.0,
        help="If >0, run periodic releases using this period in milliseconds.",
    )
    parser.add_argument(
        "--burst-size",
        type=int,
        default=0,
        help="If >0, run in bursts of N invokes followed by idle time (ignored when period-ms > 0).",
    )
    parser.add_argument(
        "--idle-ms",
        type=float,
        default=0.0,
        help="Idle time in milliseconds after each burst (ignored when period-ms > 0).",
    )
    # usbmon capture options
    parser.add_argument(
        "--capture-usbmon",
        action="store_true",
        help="Enable usbmon capture during the run.",
    )
    parser.add_argument(
        "--usb-bus",
        type=int,
        default=2,
        help="USB bus number for usbmon capture (default: 2).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for usbmon capture files. If not specified, uses RESULTS_DIR.",
    )
    parser.add_argument(
        "--lead-s",
        type=float,
        default=1.0,
        help="Seconds to wait after starting usbmon before running invokes.",
    )
    parser.add_argument(
        "--tail-s",
        type=float,
        default=1.0,
        help="Seconds to wait after finishing invokes before stopping usbmon.",
    )
    return parser.parse_args()


# ===================== usbmon capture helpers =====================

def start_usbmon_capture(bus: int, outdir: Path):
    """Start usbmon capture in background. Returns (cat_proc, usbmon_file, time_map_file)."""
    usbmon_file = outdir / "usbmon.txt"
    time_map_file = outdir / "time_map.json"
    usbmon_node = f"/sys/kernel/debug/usb/usbmon/{bus}u"
    
    # Read password
    if not PASSWORD_FILE.exists():
        raise FileNotFoundError(f"Missing password file: {PASSWORD_FILE}")
    password = PASSWORD_FILE.read_text().strip()
    
    # Load usbmon module
    subprocess.run(
        ["sudo", "-S", "modprobe", "usbmon"],
        input=password + "\n", capture_output=True, text=True
    )
    
    # Clear capture file
    subprocess.run(
        ["sudo", "-S", "sh", "-c", f": > '{usbmon_file}'"],
        input=password + "\n", capture_output=True, text=True
    )
    
    # Start cat in background using shell
    cat_cmd = f"echo '{password}' | sudo -S sh -c \"cat '{usbmon_node}' > '{usbmon_file}'\""
    cat_proc = subprocess.Popen(
        cat_cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    
    time.sleep(0.2)  # Let capture start
    
    # Initialize time_map with placeholder - will be populated after invokes
    time_map_file.write_text(json.dumps({
        'usbmon_ref': None,
        'boottime_ref': now_boottime(),
        'time_map_ready': False,
    }))
    
    print(f"[usbmon] Capture started, time_map will be built after invokes")
    return cat_proc, usbmon_file, time_map_file


def finalize_time_map(usbmon_file: Path, time_map_file: Path, timeout_s: float = 5.0):
    """Read first usbmon timestamp and update time_map after invokes complete."""
    deadline = time.time() + timeout_s
    usb_ts = None
    while time.time() < deadline and usb_ts is None:
        try:
            with open(usbmon_file, 'r', errors='ignore') as f:
                for ln in f:
                    parts = ln.split()
                    if not parts:
                        continue
                    for idx in (1, 0):
                        if idx < len(parts):
                            try:
                                v = float(parts[idx])
                                usb_ts = v / 1e6 if v > 1e6 else v
                                break
                            except Exception:
                                pass
                    if usb_ts is not None:
                        break
        except FileNotFoundError:
            pass
        if usb_ts is None:
            time.sleep(0.02)
    
    bt_ref = now_boottime()
    time_map_file.write_text(json.dumps({
        'usbmon_ref': usb_ts,
        'boottime_ref': bt_ref,
        'time_map_ready': usb_ts is not None,
    }))
    
    ready = usb_ts is not None
    print(f"time_map_ready={ready}")
    if not ready:
        print("警告: time_map 同步失败，结果可能不准确")
    return ready


def stop_usbmon_capture(cat_proc: subprocess.Popen):
    """Stop usbmon capture."""
    if cat_proc is None:
        return
    
    password = PASSWORD_FILE.read_text().strip() if PASSWORD_FILE.exists() else ""
    
    # Kill the cat process (need sudo since it's running as root)
    try:
        subprocess.run(
            ["sudo", "-S", "pkill", "-f", f"cat.*/usbmon/"],
            input=password + "\n", capture_output=True, text=True
        )
    except Exception:
        pass
    
    try:
        cat_proc.terminate()
        cat_proc.wait(timeout=2)
    except Exception:
        try:
            cat_proc.kill()
        except Exception:
            pass
    
    print("cat 已停止")


def run_offline_align(usbmon_file: Path, invokes_file: Path, time_map_file: Path):
    """Run offline aligner to refine time_map."""
    if not OFFLINE_ALIGN.exists():
        print(f"[skip] offline_align: {OFFLINE_ALIGN} not found")
        return
    
    py_exec = str(VENV_PY) if VENV_PY.exists() else "python3"
    cmd = [py_exec, str(OFFLINE_ALIGN), str(usbmon_file), str(invokes_file), str(time_map_file), '--min-urb-bytes', '512']
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            msg = res.stderr.strip() or res.stdout.strip()
            print(f"[warn] offline align failed ({res.returncode}): {msg}")
        else:
            print("[done] offline_align completed")
    except Exception as exc:
        print(f"[warn] offline align exception: {exc}")


def run_analyze_active(usbmon_file: Path, invokes_file: Path, time_map_file: Path, outdir: Path):
    """Run analyze_usbmon_active.py to generate active_analysis.json."""
    if not ANALYZE_ACTIVE.exists():
        print(f"[skip] analyze_active: {ANALYZE_ACTIVE} not found")
        return
    
    py_exec = str(VENV_PY) if VENV_PY.exists() else "python3"
    env = os.environ.copy()
    env['STRICT_INVOKE_WINDOW'] = '1'
    env['SHIFT_POLICY'] = 'tail_last_BiC_guard_BoS'
    env['SEARCH_TAIL_MS'] = '40'
    env['SEARCH_HEAD_MS'] = '40'
    env['EXTRA_HEAD_EXPAND_MS'] = '10'
    env['MAX_SHIFT_MS'] = '50'
    env['SPAN_STRICT_PAIR'] = '1'
    env['MIN_URB_BYTES'] = '512'
    
    # analyze_usbmon_active.py takes exactly 3 args and prints JSON to stdout
    cmd = [
        py_exec, str(ANALYZE_ACTIVE),
        str(usbmon_file), str(invokes_file), str(time_map_file),
    ]
    analysis_file = outdir / 'active_analysis.json'
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        if res.returncode != 0:
            msg = res.stderr.strip() or res.stdout.strip()
            print(f"[warn] analyze_active failed ({res.returncode}): {msg[:200]}")
        else:
            # Write stdout (JSON) to file
            analysis_file.write_text(res.stdout)
            print("[done] analyze_active completed")
            # Print summary
            try:
                analysis = json.loads(res.stdout)
                invokes_data = analysis.get('per_invoke', [])
                if invokes_data:
                    # Fields are in seconds, convert to ms
                    invoke_ms_list = [inv.get('invoke_span_s', 0) * 1000 for inv in invokes_data]
                    union_ms_list = [inv.get('union_active_span_s', 0) * 1000 for inv in invokes_data]
                    in_ms_list = [inv.get('in_active_span_s', 0) * 1000 for inv in invokes_data]
                    out_ms_list = [inv.get('out_active_span_s', 0) * 1000 for inv in invokes_data]
                    bytes_in_list = [inv.get('bytes_in', 0) for inv in invokes_data]
                    bytes_out_list = [inv.get('bytes_out', 0) for inv in invokes_data]
                    if invoke_ms_list:
                        avg_invoke = sum(invoke_ms_list) / len(invoke_ms_list)
                        print(f"  invoke_ms: mean={avg_invoke:.2f}, min={min(invoke_ms_list):.2f}, max={max(invoke_ms_list):.2f}, n={len(invoke_ms_list)}")
                    if union_ms_list:
                        print(f"  union_active_ms: mean={sum(union_ms_list)/len(union_ms_list):.2f}")
                    if in_ms_list:
                        print(f"  in_active_ms: mean={sum(in_ms_list)/len(in_ms_list):.2f}")
                    if out_ms_list:
                        print(f"  out_active_ms: mean={sum(out_ms_list)/len(out_ms_list):.2f}")
                    if bytes_out_list:
                        print(f"  bytes_out: mean={sum(bytes_out_list)/len(bytes_out_list)/1e6:.2f} MB/invoke")
            except Exception:
                pass
    except Exception as exc:
        print(f"[warn] analyze_active exception: {exc}")


# ===================== interpreter / worker =====================


def make_interpreter(model_path: str, device: str):
    try:
        from pycoral.utils.edgetpu import make_interpreter as coral_make
    except Exception:
        coral_make = None

    if coral_make is not None:
        return coral_make(model_path, device=device)

    from tflite_runtime.interpreter import Interpreter, load_delegate

    delegate = load_delegate("libedgetpu.so.1", {"device": device})
    init_kwargs = {"model_path": model_path, "experimental_delegates": [delegate], "num_threads": 1}
    interpreter = Interpreter(**init_kwargs)
    return interpreter


def set_affinity(core: Optional[int]):
    if core is None:
        return
    proc = psutil.Process()
    try:
        proc.cpu_affinity([core])
    except AttributeError:
        os.sched_setaffinity(0, {core})


def run_worker(
    idx: int,
    device: str,
    core: Optional[int],
    models: List[str],
    iterations: int,
    queue: mp.Queue,
    sync_barrier: Optional[mp.Barrier] = None,
    sync_all: bool = False,
    first_barrier: Optional[mp.Barrier] = None,
    sync_start_barrier: Optional[mp.Barrier] = None,
    sync_release_barrier: Optional[mp.Barrier] = None,
    start_offset_ms: float = 0.0,
    per_iter_offset_ms: float = 0.0,
    reuse_interpreter: bool = False,
    warmup: int = 0,
    read_output: bool = False,
    period_ms: float = 0.0,
    burst_size: int = 0,
    idle_ms: float = 0.0,
):
    set_affinity(core)
    if idx > 0 and start_offset_ms > 0:
        time.sleep(start_offset_ms / 1000.0)
    rng = random.Random(1234 + idx)
    records = []
    first_create_end_ts: Optional[float] = None
    first_create_wait: float = 0.0
    period_s = period_ms / 1000.0 if period_ms > 0 else 0.0
    burst_enabled = period_s == 0 and burst_size > 0 and idle_ms > 0
    burst_counter = 0
    next_release = 0.0
    reuse_cache = {}
    if reuse_interpreter:
        for model_path in models:
            t0 = time.perf_counter()
            interpreter = make_interpreter(model_path, device)
            t_make_end = time.perf_counter()
            if first_create_end_ts is None:
                first_create_end_ts = t_make_end
                if first_barrier is not None:
                    wait_start = time.perf_counter()
                    first_barrier.wait()
                    first_create_wait = time.perf_counter() - wait_start
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            shape = inp["shape"]
            dtype = np.dtype(inp["dtype"])
            data = rng.random()
            if np.issubdtype(dtype, np.integer):
                input_tensor = np.full(shape, int(data * 10), dtype=dtype)
            else:
                input_tensor = np.full(shape, float(data), dtype=dtype)
            if warmup > 0:
                for _ in range(warmup):
                    interpreter.set_tensor(inp["index"], input_tensor)
                    interpreter.invoke()
                    if read_output:
                        _ = interpreter.get_tensor(out["index"])
            reuse_cache[model_path] = (interpreter, inp, out, input_tensor)
    if sync_start_barrier is not None:
        sync_start_barrier.wait()
    if period_s > 0:
        next_release = time.perf_counter()
    for _ in range(iterations):
        for model_path in models:
            if idx > 0 and per_iter_offset_ms > 0:
                time.sleep(per_iter_offset_ms / 1000.0)
            scheduled_release = None
            release_time = None
            if period_s > 0:
                scheduled_release = next_release
                now = time.perf_counter()
                if now < scheduled_release:
                    time.sleep(scheduled_release - now)
                release_time = time.perf_counter()
            elif sync_release_barrier is not None:
                sync_release_barrier.wait()
            if sync_barrier is not None:
                sync_barrier.wait()  # align creation start
            t0 = time.perf_counter()
            if reuse_interpreter:
                interpreter, inp, out, input_tensor = reuse_cache[model_path]
                t_make_end = t0
                wait_after_create = 0.0
                if sync_barrier is not None:
                    wait_start = time.perf_counter()
                    sync_barrier.wait()
                    wait_after_create = time.perf_counter() - wait_start
                t_alloc_end = t_make_end
                wait_after_alloc = 0.0
                if sync_barrier is not None and sync_all:
                    wait_start = time.perf_counter()
                    sync_barrier.wait()
                    wait_after_alloc = time.perf_counter() - wait_start
                t_input_end = t_alloc_end
                interpreter.set_tensor(inp["index"], input_tensor)
                t_set_end = time.perf_counter()
                wait_before_invoke = 0.0
                if sync_barrier is not None and sync_all:
                    wait_start = time.perf_counter()
                    sync_barrier.wait()
                    wait_before_invoke = time.perf_counter() - wait_start
            else:
                interpreter = make_interpreter(model_path, device)
                t_make_end = time.perf_counter()
                if first_create_end_ts is None:
                    first_create_end_ts = t_make_end
                    if first_barrier is not None:
                        wait_start = time.perf_counter()
                        first_barrier.wait()
                        first_create_wait = time.perf_counter() - wait_start
                wait_after_create = 0.0
                if sync_barrier is not None:
                    wait_start = time.perf_counter()
                    sync_barrier.wait()  # proceed only when all workers finished create
                    wait_after_create = time.perf_counter() - wait_start
                interpreter.allocate_tensors()
                t_alloc_end = time.perf_counter()
                wait_after_alloc = 0.0
                if sync_barrier is not None and sync_all:
                    wait_start = time.perf_counter()
                    sync_barrier.wait()  # align after allocate
                    wait_after_alloc = time.perf_counter() - wait_start
                inp = interpreter.get_input_details()[0]
                out = interpreter.get_output_details()[0]
                shape = inp["shape"]
                dtype = np.dtype(inp["dtype"])
                data = rng.random()
                # synthesize input; now record each stage (make/load → alloc → build input → set → invoke)
                if np.issubdtype(dtype, np.integer):
                    input_tensor = np.full(shape, int(data * 10), dtype=dtype)
                else:
                    input_tensor = np.full(shape, float(data), dtype=dtype)
                t_input_end = time.perf_counter()
                interpreter.set_tensor(inp["index"], input_tensor)
                t_set_end = time.perf_counter()
                wait_before_invoke = 0.0
                if sync_barrier is not None and sync_all:
                    wait_start = time.perf_counter()
                    sync_barrier.wait()  # align invoke start
                    wait_before_invoke = time.perf_counter() - wait_start
            t_invoke_start = time.perf_counter()
            bt_invoke_begin = now_boottime()
            interpreter.invoke()
            t_invoke_end = time.perf_counter()
            bt_invoke_end = now_boottime()
            read_output_ms = 0.0
            t1 = t_invoke_end
            if read_output:
                out_idx = out["index"]
                _ = interpreter.get_tensor(out_idx)
                t1 = time.perf_counter()
                read_output_ms = (t1 - t_invoke_end) * 1000.0
            wait_total = wait_after_create + wait_after_alloc + wait_before_invoke
            deadline_miss = False
            lateness_ms = 0.0
            release_jitter_ms = 0.0
            if period_s > 0 and scheduled_release is not None and release_time is not None:
                release_jitter_ms = (release_time - scheduled_release) * 1000.0
                deadline = scheduled_release + period_s
                lateness_ms = (t1 - deadline) * 1000.0
                deadline_miss = lateness_ms > 0
                next_release = scheduled_release + period_s
            records.append({
                "model": Path(model_path).name,
                "create_ms": 0.0 if reuse_interpreter else (t_make_end - t0) * 1000.0,
                "wait_after_create_ms": wait_after_create * 1000.0,
                "allocate_ms": 0.0 if reuse_interpreter else (t_alloc_end - t_make_end) * 1000.0,
                "wait_after_allocate_ms": wait_after_alloc * 1000.0,
                "build_input_ms": 0.0 if reuse_interpreter else (t_input_end - t_alloc_end) * 1000.0,
                "set_ms": (t_set_end - t_input_end) * 1000.0,
                "wait_before_invoke_ms": wait_before_invoke * 1000.0,
                "invoke_ms": (t_invoke_end - t_invoke_start) * 1000.0,
                "read_output_ms": read_output_ms,
                "total_ms": (t1 - t0) * 1000.0,  # includes waiting after create if sync enabled
                "work_total_ms": max(((t1 - t0) - wait_total) * 1000.0, 0.0),
                "boottime_invoke_begin": bt_invoke_begin,
                "boottime_invoke_end": bt_invoke_end,
                "scheduled_release_ts": scheduled_release,
                "release_ts": release_time,
                "start_ts": t0,
                "finish_ts": t1,
                "release_jitter_ms": release_jitter_ms,
                "lateness_ms": lateness_ms,
                "deadline_miss": deadline_miss,
            })
            if burst_enabled:
                burst_counter += 1
                if burst_counter % burst_size == 0:
                    time.sleep(idle_ms / 1000.0)
    queue.put({
        "device": device,
        "core": core,
        "records": records,
        "first_create_end_ts": first_create_end_ts,
        "first_create_wait_ms": first_create_wait * 1000.0,
    })


def main():
    args = parse_args()
    
    # Handle mixed-models: TPU0 runs deeplab, TPU1 runs mn7
    if args.mixed_models:
        if not args.single_model:
            print("Warning: --mixed-models implies --single-model, enabling it.")
            args.single_model = True
        models_per_worker = [
            [str(args.deeplab)],  # worker 0
            [str(args.mn7)],      # worker 1
        ]
        for wmodels in models_per_worker:
            for p in wmodels:
                if not Path(p).exists():
                    raise FileNotFoundError(f"Model not found: {p}")
        models = None  # signal to use per-worker models
    else:
        models = [str(args.mn4)]
        if not args.single_model:
            models.append(str(args.mn5))
        for p in models:
            if not Path(p).exists():
                raise FileNotFoundError(f"Model not found: {p}")
        models_per_worker = None

    # Determine output directory
    if args.outdir:
        outdir = args.outdir
    else:
        outdir = RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    # Start usbmon capture if requested
    cat_proc = None
    usbmon_file = None
    time_map_file = None
    if args.capture_usbmon:
        print(f"[usbmon] Starting capture on bus {args.usb_bus}...")
        cat_proc, usbmon_file, time_map_file = start_usbmon_capture(args.usb_bus, outdir)
        print(f"[usbmon] Lead time: {args.lead_s}s")
        time.sleep(args.lead_s)

    wall_start = time.time()
    perf_start = time.perf_counter()
    meta = {
        "models": models if models else [m for w in models_per_worker for m in w],
        "models_per_worker": models_per_worker,
        "devices": args.devices,
        "iterations": args.iterations,
        "cpu_cores": args.cpu_cores,
        "sync_create": args.sync_create,
        "sync_all": args.sync_all,
        "align_first_create": args.align_first_create,
        "reuse_interpreter": args.reuse_interpreter,
        "warmup": args.warmup,
        "read_output": args.read_output,
        "period_ms": args.period_ms,
        "burst_size": args.burst_size,
        "idle_ms": args.idle_ms,
        "single_model": args.single_model,
        "sync_start": args.sync_start,
        "sync_release": args.sync_release,
        "wall_start_ts": wall_start,
        "capture_usbmon": args.capture_usbmon,
        "usb_bus": args.usb_bus if args.capture_usbmon else None,
    }

    procs = []
    q = mp.Queue()
    sync_enabled = (args.sync_create or args.sync_all) and len(args.devices) > 1
    barrier = mp.Barrier(len(args.devices)) if sync_enabled else None
    first_barrier = mp.Barrier(len(args.devices)) if args.align_first_create and len(args.devices) > 1 else None
    sync_start_barrier = mp.Barrier(len(args.devices)) if args.sync_start and len(args.devices) > 1 else None
    sync_release_barrier = mp.Barrier(len(args.devices)) if args.sync_release and len(args.devices) > 1 else None

    for i, device in enumerate(args.devices):
        core = args.cpu_cores[i] if i < len(args.cpu_cores) else None
        # Use per-worker models if mixed-models, else shared models list
        worker_models = models_per_worker[i] if models_per_worker else models
        p = mp.Process(
            target=run_worker,
            args=(
                i,
                device,
                core,
                worker_models,
                args.iterations,
                q,
                barrier,
                args.sync_all,
                first_barrier,
                sync_start_barrier,
                sync_release_barrier,
                args.start_offset_ms,
                args.per_iter_offset_ms,
                args.reuse_interpreter,
                args.warmup,
                args.read_output,
                args.period_ms,
                args.burst_size,
                args.idle_ms,
            ),
        )
        p.start()
        procs.append(p)

    summaries = []
    for _ in procs:
        summaries.append(q.get())
    for p in procs:
        p.join()

    # Stop usbmon capture after tail wait
    if args.capture_usbmon and cat_proc:
        print(f"[usbmon] Tail time: {args.tail_s}s")
        time.sleep(args.tail_s)
        stop_usbmon_capture(cat_proc)
        # Finalize time_map with actual usbmon timestamps
        finalize_time_map(usbmon_file, time_map_file)

    wall_end = time.time()
    perf_end = time.perf_counter()
    meta["wall_end_ts"] = wall_end
    meta["wall_elapsed_ms"] = (wall_end - wall_start) * 1000.0
    meta["perf_elapsed_ms"] = (perf_end - perf_start) * 1000.0
    first_ends = [r.get("first_create_end_ts") for r in summaries if r.get("first_create_end_ts")]
    if first_ends:
        slowest_first_end = max(first_ends)
        meta["first_create_end_ts_max"] = slowest_first_end
        meta["wall_elapsed_from_first_create_ms"] = (perf_end - slowest_first_end) * 1000.0

    out_path = outdir / "invoke_only.json"
    out_obj = {"meta": meta, "results": summaries}
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2))
    # Also emit a simple invokes.json (BOOTTIME spans) for usbmon alignment.
    spans = []
    for entry in summaries:
        for r in entry.get("records", []):
            if "boottime_invoke_begin" in r and "boottime_invoke_end" in r:
                spans.append({
                    "begin": r["boottime_invoke_begin"],
                    "end": r["boottime_invoke_end"],
                    "device": entry.get("device"),
                    "model": r.get("model"),
                })
    invokes_path = outdir / "invokes.json"
    invokes_path.write_text(json.dumps({"name": "dual_tpu_toggle", "spans": spans}, ensure_ascii=False, indent=2))
    print(f"[done] wrote {out_path} and {invokes_path}")
    print(f"[wall] elapsed {meta['wall_elapsed_ms']:.2f} ms")
    for entry in summaries:
        times = [r["invoke_ms"] for r in entry["records"]]
        total_times = [r["total_ms"] for r in entry["records"]]
        create_times = [r["create_ms"] for r in entry["records"]]
        alloc_times = [r["allocate_ms"] for r in entry["records"]]
        build_times = [r["build_input_ms"] for r in entry["records"]]
        set_times = [r["set_ms"] for r in entry["records"]]
        waits = [r["wait_after_create_ms"] for r in entry["records"]]
        waits_alloc = [r.get("wait_after_allocate_ms", 0.0) for r in entry["records"]]
        waits_pre_invoke = [r.get("wait_before_invoke_ms", 0.0) for r in entry["records"]]
        work_totals = [r["work_total_ms"] for r in entry["records"]]
        avg_invoke = sum(times) / len(times)
        avg_total = sum(total_times) / len(total_times)
        avg_create = sum(create_times) / len(create_times)
        avg_alloc = sum(alloc_times) / len(alloc_times)
        avg_build = sum(build_times) / len(build_times)
        avg_set = sum(set_times) / len(set_times)
        avg_wait = sum(waits) / len(waits)
        avg_wait_alloc = sum(waits_alloc) / len(waits_alloc)
        avg_wait_pre_invoke = sum(waits_pre_invoke) / len(waits_pre_invoke)
        avg_work_total = sum(work_totals) / len(work_totals)
        miss_count = sum(1 for r in entry["records"] if r.get("deadline_miss"))
        lateness_vals = [r["lateness_ms"] for r in entry["records"] if r.get("lateness_ms", 0.0) > 0]
        avg_late = sum(lateness_vals) / len(lateness_vals) if lateness_vals else 0.0
        max_late = max(lateness_vals) if lateness_vals else 0.0
        print(
            f"device {entry['device']} core {entry['core']}: {len(times)} toggles, "
            f"avg create {avg_create:.2f} ms, alloc {avg_alloc:.2f} ms, "
            f"build {avg_build:.2f} ms, set {avg_set:.2f} ms, "
            f"invoke {avg_invoke:.2f} ms, wait_after_create {avg_wait:.2f} ms, "
            f"wait_after_alloc {avg_wait_alloc:.2f} ms, wait_before_invoke {avg_wait_pre_invoke:.2f} ms, "
            f"work_total {avg_work_total:.2f} ms, total {avg_total:.2f} ms, "
            f"miss {miss_count}, avg_late {avg_late:.2f} ms, max_late {max_late:.2f} ms"
        )

    # Run offline align and analyze if usbmon was captured
    if args.capture_usbmon and usbmon_file and time_map_file:
        print("\n[post] Running offline alignment...")
        run_offline_align(usbmon_file, invokes_path, time_map_file)
        print("[post] Running active analysis...")
        run_analyze_active(usbmon_file, invokes_path, time_map_file, outdir)
        print(f"\n[done] Output files in: {outdir}")
        print(f"  - usbmon.txt")
        print(f"  - time_map.json")
        print(f"  - invokes.json")
        print(f"  - invoke_only.json")
        print(f"  - active_analysis.json")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
