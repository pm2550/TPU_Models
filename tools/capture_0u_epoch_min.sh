#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="$1"         # 输出目录
MODEL_PATH="$2"      # 模型路径
COUNT="${3:-20}"     # 次数
DUR_S="${4:-8}"      # 采集时长（秒）

PW_FILE="/home/10210/Desktop/OS/password.text"
USBNODE="/sys/kernel/debug/usb/usbmon/0u"

if [[ ! -f "$PW_FILE" ]]; then echo "missing $PW_FILE" >&2; exit 1; fi
mkdir -p "$OUT_DIR"
USB_TXT="$OUT_DIR/usbmon.txt"
IV_JSON="$OUT_DIR/invokes.json"

# 1) 启动 usbmon(0u) 采集
cat "$PW_FILE" | sudo -S -p '' modprobe usbmon
cat "$PW_FILE" | sudo -S -p '' sh -c ": > '$USB_TXT'"
cat "$PW_FILE" | sudo -S -p '' sh -c "cat '$USBNODE' >> '$USB_TXT'" &
CAP_PID=$!
sleep 1

# 2) 仅记录 invoke 窗口（EPOCH 时间）
PY_VENV="/home/10210/Desktop/OS/.venv/bin/python"
PY_SYS="$(command -v python3 || echo /usr/bin/python3)"
PYRUN="$PY_SYS"
if [[ -x "$PY_VENV" ]]; then PYRUN="$PY_VENV"; fi

"$PYRUN" - "$MODEL_PATH" "$COUNT" "$IV_JSON" <<'PY'
import json, sys, time, numpy as np
m=sys.argv[1]
cnt=int(sys.argv[2])
out=sys.argv[3]
it=None
try:
    from pycoral.utils.edgetpu import make_interpreter
    it=make_interpreter(m)
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter, load_delegate
        it=Interpreter(model_path=m, experimental_delegates=[load_delegate('libedgetpu.so.1')])
    except Exception as e2:
        print('INTERPRETER_FAIL', repr(e2)); raise SystemExit(1)
it.allocate_tensors()
inp=it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
for _ in range(5):
    it.set_tensor(inp['index'], x); it.invoke()
spans=[]
for i in range(cnt):
    it.set_tensor(inp['index'], x)
    t0=time.time()
    it.invoke()
    t1=time.time()
    spans.append({'begin': t0, 'end': t1})
open(out,'w').write(json.dumps({'name':'invoke_only','spans':spans}))
print('INVOKES_OK', len(spans))
PY

# 3) 等待到持续时间结束并停止采集
sleep "$DUR_S"
kill "$CAP_PID" 2>/dev/null || true
sleep 0.1 || true
cat "$PW_FILE" | sudo -S -p '' chown "$USER:$USER" "$USB_TXT" "$IV_JSON" 2>/dev/null || true
cat "$PW_FILE" | sudo -S -p '' chmod 0644 "$USB_TXT" "$IV_JSON" 2>/dev/null || true

# 4) 打印 Bo/Bi 计数与逐次统计
echo '--- counts ---'
grep -c ' Bo:' "$USB_TXT" || true
grep -c ' Bi:' "$USB_TXT" || true
grep -c ' Co:' "$USB_TXT" || true
grep -c ' Ci:' "$USB_TXT" || true
echo '--- per-invoke (±5s) ---'
python3 /home/10210/Desktop/OS/tools/per_invoke_intersection_stats.py "$OUT_DIR" 5.0 | sed -n '1,80p'

