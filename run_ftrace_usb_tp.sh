#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_ftrace_usb_tp.sh <model_path> <name> <out_dir>

if [[ $# -lt 3 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir>" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
OUTDIR="$3"
mkdir -p "$OUTDIR"

PW="/home/10210/Desktop/OS/password.text"
TRACE="/sys/kernel/debug/tracing"

if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2; exit 1
fi

# 启用 USB 提交/完成 tracepoints
cat "$PW" | sudo -S -p '' sh -c "echo nop > $TRACE/current_tracer" || true
cat "$PW" | sudo -S -p '' sh -c "echo 0 > $TRACE/tracing_on" || true
for ev in usb/usb_submit_urb usb/usb_complete_urb; do
  cat "$PW" | sudo -S -p '' sh -c "echo 0 > $TRACE/events/$ev/enable" || true
done
cat "$PW" | sudo -S -p '' sh -c ": > $TRACE/trace" || true
for ev in usb/usb_submit_urb usb/usb_complete_urb; do
  cat "$PW" | sudo -S -p '' sh -c "echo 1 > $TRACE/events/$ev/enable" || true
done
cat "$PW" | sudo -S -p '' sh -c "echo 1 > $TRACE/tracing_on" || true

# 允许用户态写 trace_marker 以标记窗口
cat "$PW" | sudo -S -p '' chmod o+rw $TRACE/trace_marker || true

INVOKES_JSON="$OUTDIR/invokes.json"

"/home/10210/Desktop/OS/.venv/bin/python" - "$MODEL_PATH" <<'PY' > "$INVOKES_JSON"
import json, time, numpy as np, sys
from pycoral.utils.edgetpu import make_interpreter
m=sys.argv[1]
interpreter = make_interpreter(m)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    arr = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    arr = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)

marker_path = '/sys/kernel/debug/tracing/trace_marker'
# 预热
for _ in range(50):
    interpreter.set_tensor(inp['index'], arr)
    interpreter.invoke()

spans=[]
for i in range(10):
    with open(marker_path,'w') as f: f.write(f'INV_BEGIN i={i}\n')
    t0=time.time()
    interpreter.set_tensor(inp['index'], arr)
    interpreter.invoke()
    t1=time.time()
    with open(marker_path,'w') as f: f.write(f'INV_END i={i}\n')
    spans.append({'begin': t0, 'end': t1})

print(json.dumps({'name': r"${NAME}", 'spans': spans}))
PY

TRACE_OUT="$OUTDIR/trace.txt"
cat "$PW" | sudo -S -p '' sh -c "echo 0 > $TRACE/tracing_on" || true
cat "$PW" | sudo -S -p '' sh -c "cat $TRACE/trace > '$TRACE_OUT'" || true
for ev in usb/usb_submit_urb usb/usb_complete_urb; do
  cat "$PW" | sudo -S -p '' sh -c "echo 0 > $TRACE/events/$ev/enable" || true
done

cat "$PW" | sudo -S -p '' chmod 0644 "$TRACE_OUT" "$INVOKES_JSON" 2>/dev/null || true
chown "$USER:$USER" "$TRACE_OUT" "$INVOKES_JSON" 2>/dev/null || true

echo "已保存: $INVOKES_JSON, $TRACE_OUT"




