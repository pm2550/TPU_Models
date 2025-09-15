#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_capture_offline.sh <model_path> <name> <out_dir> <bus> <duration_s>

if [[ $# -lt 5 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir> <bus> <duration_s>" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
OUTDIR="$3"
BUS="$4"
DUR="$5"

PW="/home/10210/Desktop/OS/password.text"
USBMON_NODE="/sys/kernel/debug/usb/usbmon/${BUS}u"
CAP="$OUTDIR/usbmon.txt"
TM="$OUTDIR/time_map.json"
IV="$OUTDIR/invokes.json"
OUT_JSON="$OUTDIR/io_split_bt.json"
# 记录次数（默认 10，可通过环境变量 COUNT 覆盖，例如 COUNT=101）
COUNT="${COUNT:-10}"

# 选择 Python 解释器：
# - 时间映射与分析使用系统 python3（不依赖 pycoral）
# - 运行模型优先使用 .venv，其次系统 python3；模型内会优先尝试 pycoral，失败则回退 tflite_runtime
PY_SYS="$(command -v python3)"
PY_MAP="${PY_SYS:-/usr/bin/python3}"
PY_ANALYZE="${PY_SYS:-/usr/bin/python3}"

if [[ -x "/home/10210/Desktop/OS/.venv/bin/python" ]]; then
  RUN_PY="/home/10210/Desktop/OS/.venv/bin/python"
elif [[ -n "${PY_SYS}" ]]; then
  RUN_PY="${PY_SYS}"
else
  RUN_PY="/usr/bin/python3"
fi

mkdir -p "$OUTDIR"

if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2
  exit 1
fi

# 启动 usbmon 采集
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
cat "$PW" | sudo -S -p '' sh -c "cat '$USBMON_NODE' > '$CAP'" &
CATPID=$!
sleep 0.1

# 启动映射记录器：读取首条 usbmon 行的时间戳（尝试第2/第1列），与同刻 BOOTTIME 建立映射（阻塞至超时）
"$PY_MAP" - "$CAP" "$TM" <<'PY' &
import json,sys,time
cap, tm_path = sys.argv[1:]
usb_ts=None
deadline=time.time()+30.0
while time.time()<deadline and usb_ts is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            for ln in f:
                parts=ln.split()
                if not parts:
                    continue
                ts=None
                for idx in (1,0):
                    if idx < len(parts):
                        try:
                            v=float(parts[idx]); ts=v/1e6 if v>1e6 else v
                            break
                        except Exception:
                            pass
                if ts is not None:
                    usb_ts=ts
                    break
    except FileNotFoundError:
        pass
    if usb_ts is None:
        time.sleep(0.02)
try:
    bt_ref=time.clock_gettime(time.CLOCK_BOOTTIME)
except Exception:
    bt_ref=float(open('/proc/uptime').read().split()[0])
open(tm_path,'w').write(json.dumps({'usbmon_ref': usb_ts, 'boottime_ref': bt_ref}))
PY

# 等待 time_map.json 就绪（包含 usbmon_ref 与 boottime_ref）
"$PY_MAP" - "$TM" <<'PY'
import json,sys,time,os
tm_path=sys.argv[1]
deadline=time.time()+30.0
ready=False
while time.time()<deadline and not ready:
    try:
        j=json.load(open(tm_path))
        ready=(j.get('usbmon_ref') is not None) and (j.get('boottime_ref') is not None)
    except Exception:
        ready=False
    if not ready:
        time.sleep(0.02)
print('time_map_ready=', ready)
if not ready:
    print('警告: time_map 同步失败，结果可能不准确')
PY

# 采集前预留窗口（可通过 LEAD_S 指定，默认 3）
sleep "${LEAD_S:-3}"

# 运行模型：记录 COUNT 次 invoke 的 EPOCH 窗口（CLOCK_BOOTTIME）
"$RUN_PY" - "$MODEL_PATH" <<'PY' > "$IV"
import json, time, numpy as np, sys, os
m=sys.argv[1]

it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception as e:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    except Exception as e2:
        print('INTERPRETER_FAIL', repr(e), repr(e2))
        raise SystemExit(1)

it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
# 预热次数由环境变量 WARMUP 控制（默认0）
try:
    warm=int(os.environ.get('WARMUP','0'))
except Exception:
    warm=0
for _ in range(max(0, warm)):
    it.set_tensor(inp['index'], x)
    it.invoke()
    _ = it.get_tensor(it.get_output_details()[0]['index'])
spans=[]
# 结果消费策略：
# - 读取输出张量并计算一个轻量校验和，避免底层延迟传输/融合优化
# - 可选将结果写入临时文件（环境变量 WRITE_OUT=1 开启）
write_out = os.environ.get('WRITE_OUT','0') == '1'
out_path = os.environ.get('WRITE_OUT_PATH','') or '/tmp/edgetpu_out.bin'
try:
    cnt=int(os.environ.get('COUNT','10'))
except Exception:
    cnt=10
for i in range(cnt):
    # 记录 set/invoke/get 的细分时间戳（CLOCK_BOOTTIME）
    # 可选：每次轻微修改输入，避免底层对相同输入做缓存/去重（开启: MUTATE_INPUT=1）
    if os.environ.get('MUTATE_INPUT','0')=='1':
        try:
            # 对第一个元素做一次轻量按位异或，代价极小
            x.flat[0] = np.bitwise_xor(x.flat[0], np.array(i & 0x7, dtype=x.dtype))
        except Exception:
            pass
    # set
    t_set0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.set_tensor(inp['index'], x)
    t_set1=time.clock_gettime(time.CLOCK_BOOTTIME)
    # invoke
    t0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.invoke()
    t1=time.clock_gettime(time.CLOCK_BOOTTIME)
    # get 并强制消费
    t_get0=time.clock_gettime(time.CLOCK_BOOTTIME)
    out_info = it.get_output_details()[0]
    out = it.get_tensor(out_info['index'])
    # 轻量计算（校验和），避免优化跳过读取
    checksum = int(np.sum(out) % 2**32)
    t_get1=time.clock_gettime(time.CLOCK_BOOTTIME)
    if write_out:
        try:
            # 附加写入，避免频繁覆盖
            with open(out_path, 'ab') as f:
                f.write(out.tobytes())
        except Exception:
            pass
    spans.append({
        'begin': t0,
        'end': t1,
        'set_begin': t_set0,
        'set_end': t_set1,
        'get_begin': t_get0,
        'get_end': t_get1,
        'checksum': checksum,
        'out_shape': tuple(out.shape),
        'out_dtype': str(out.dtype),
    })
    # 推理间隔，避免长尾IO影响下次统计
    gap_ms = float(os.environ.get('INVOKE_GAP_MS', '0'))
    if gap_ms > 0:
        time.sleep(gap_ms / 1000.0)
print(json.dumps({'name': r"${NAME}", 'spans': spans}))
PY

# 等待到持续时间结束
sleep "$DUR"

# 停止采集并修正权限
kill "$CATPID" 2>/dev/null || true
sleep 0.1 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$TM" "$IV" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$TM" "$IV" 2>/dev/null || true

# 分析（可选快速口径）：默认禁用；设置 ONLY_ANALYZER=0 可启用快速口径以兼容旧流程
if [[ "${ONLY_ANALYZER:-1}" != "1" ]]; then
"$PY_ANALYZE" - <<'PY' | tee "$OUT_JSON"
import json, os, re, subprocess
CAP=r"$CAP"; IV=r"$IV"; TM=r"$TM"; OUTDIR=r"$OUTDIR"
tool=r"/home/10210/Desktop/OS/tools/correct_per_invoke_stats.py"
py=r"/home/10210/Desktop/OS/.venv/bin/python"
# 检查 time_map 是否有效
tm_ok = False
try:
    tmj = json.load(open(TM))
    tm_ok = (tmj.get('usbmon_ref') is not None) and (tmj.get('boottime_ref') is not None)
except Exception:
    tm_ok = False

if tm_ok:
    # 有效 time_map：用严格配对
    cmd = [py, tool, CAP, IV, TM, "--mode", "bulk_complete", "--include", "full", "--extra", "0.000"]
    txt = subprocess.run(cmd, capture_output=True, text=True).stdout
    mode = "strict"
else:
    # 无效 time_map：用 overlap 回退
    cmd = [py, tool, CAP, IV, TM, "--mode", "bulk_complete", "--include", "overlap", "--extra", "0.010"]
    txt = subprocess.run(cmd, capture_output=True, text=True).stdout
    mode = "overlap_fallback"
# 保存原始输出
os.makedirs(OUTDIR, exist_ok=True)
open(os.path.join(OUTDIR, f'correct_{mode}_stdout.txt'), 'w').write(txt)

# 解析并格式化
m = re.search(r"JSON_PER_INVOKE:\s*(\[.*?\])", txt, re.S)
if m:
    arr = json.loads(m.group(1))
    result = []
    for i, x in enumerate(arr, 1):
        result.append({
            'invoke': i,
            'bytes_in': int(x.get('bytes_in', 0) or 0),
            'bytes_out': int(x.get('bytes_out', 0) or 0),
            'in_ms': round(float(x.get('in_active_s') or 0) * 1000, 3),
            'out_ms': round(float(x.get('out_active_s') or 0) * 1000, 3),
            'union_ms': round(float(x.get('union_active_s') or 0) * 1000, 3)
        })
    summary = {'mode': mode, 'time_map_ok': tm_ok, 'results': result}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(os.path.join(OUTDIR, 'io_correct_per_invoke.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
else:
    print(json.dumps({'mode': mode, 'time_map_ok': tm_ok, 'error': 'no JSON_PER_INVOKE found'}, ensure_ascii=False))
PY

echo "完成，输出: $OUT_JSON"
fi

# 运行严格窗口分析器（权威口径），生成 active_analysis_strict.json 便于后续统一消费
ANA_PY="/home/10210/Desktop/OS/analyze_usbmon_active.py"
if [[ -f "$ANA_PY" ]]; then
    echo "运行严格窗口分析器: $ANA_PY"
    STRICT_INVOKE_WINDOW="${STRICT_INVOKE_WINDOW:-1}" \
    SHIFT_POLICY="${SHIFT_POLICY:-in_tail_or_out_head}" \
    CLUSTER_GAP_MS="${CLUSTER_GAP_MS:-0.1}" \
    SEARCH_TAIL_MS="${SEARCH_TAIL_MS:-20}" \
    SEARCH_HEAD_MS="${SEARCH_HEAD_MS:-10}" \
    MAX_SHIFT_MS="${MAX_SHIFT_MS:-30}" \
    "$PY_ANALYZE" "$ANA_PY" "$CAP" "$IV" "$TM" > "$OUTDIR/active_analysis_strict.json" || echo "严格分析器执行失败，已跳过"
fi


