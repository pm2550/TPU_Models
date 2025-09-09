#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir>" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
TR="/sys/kernel/tracing"
OUTDIR="$3"
mkdir -p "$OUTDIR"

# 需要 sudo 权限
sudo -n sh -c "echo 0 > $TR/tracing_on; echo nop > $TR/current_tracer; echo global > $TR/trace_clock; echo 0 > $TR/events/enable"

for e in xhci-hcd/xhci_handle_transfer xhci-hcd/xhci_urb_enqueue xhci-hcd/xhci_urb_giveback; do
  sudo -n sh -c "echo 1 > $TR/events/$e/enable"
done

sudo -n sh -c "echo 0 > $TR/tracing_on; : > $TR/trace; echo 1 > $TR/tracing_on"

TMPJSON="$OUTDIR/invokes.json"

# 使用 FIFO 将用户态标记写入 trace_marker（需要 root 的 tee 在后台运行）
FIFO="/tmp/trace_marker_fifo_$$"
rm -f "$FIFO" && mkfifo "$FIFO"
sudo -n bash -lc "cat '$FIFO' > '$TR/trace_marker'" &
TEEPID=$!

"/home/10210/Desktop/OS/.venv/bin/python" - "$MODEL_PATH" "$FIFO" <<'PY' >"$TMPJSON"
import json, time as t, numpy as np
from pycoral.utils.edgetpu import make_interpreter
import sys
m = sys.argv[1]
it = make_interpreter(m); it.allocate_tensors(); inp = it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, inp['shape'], dtype=np.int8)
for _ in range(100): it.set_tensor(inp['index'], x); it.invoke()
ts=[]
fifo_path = sys.argv[2]
with open(fifo_path, 'w') as f:
    for i in range(10):
        it.set_tensor(inp['index'], x)
        f.write('INV_BEGIN\n'); f.flush()
        s=t.perf_counter(); it.invoke(); e=t.perf_counter()
        f.write('INV_END\n'); f.flush()
        ts.append((e-s)*1000)
print(json.dumps({'name': r"${NAME}", 'invokes_ms': ts}))
PY

sudo -n sh -c "echo 0 > $TR/tracing_on"
kill "$TEEPID" 2>/dev/null || true
rm -f "$FIFO"

# 保存完整 trace 到文件并修正权限
sudo -n cp "$TR/trace" "$OUTDIR/trace.txt"
sudo -n chown "$USER:$USER" "$OUTDIR/trace.txt" "$TMPJSON" 2>/dev/null || true
sudo -n chmod 0644 "$OUTDIR/trace.txt" "$TMPJSON" 2>/dev/null || true

# 解析窗口内每次 invoke 的 xhci 事件计数并保存
/usr/bin/env python3 - "$OUTDIR/trace.txt" <<'PY' > "$OUTDIR/window_counts.json"
import json, re, sys
trace_path = sys.argv[1]
invokes = []
cur = None
with open(trace_path, 'r') as f:
    for line in f:
        if 'tracing_mark_write: INV_BEGIN' in line:
            cur = {'begin_line': line.strip(), 'xhci_handle_transfer':0, 'xhci_urb_enqueue':0, 'xhci_urb_giveback':0}
        elif 'tracing_mark_write: INV_END' in line:
            if cur is not None:
                cur['end_line'] = line.strip()
                invokes.append(cur)
                cur = None
        elif cur is not None:
            if 'xhci_handle_transfer' in line:
                cur['xhci_handle_transfer'] += 1
            elif 'xhci_urb_enqueue' in line:
                cur['xhci_urb_enqueue'] += 1
            elif 'xhci_urb_giveback' in line:
                cur['xhci_urb_giveback'] += 1
print(json.dumps({'invokes': invokes}, ensure_ascii=False, indent=2))
PY

echo "已保存: $TMPJSON, $OUTDIR/trace.txt, $OUTDIR/window_counts.json"

