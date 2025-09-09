#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_invoke_boottime_full.sh <model_path> <name> <out_dir>

if [[ $# -lt 3 ]]; then
  echo "用法: $0 <model_path> <name> <out_dir>" >&2
  exit 1
fi

MODEL_PATH="$1"
NAME="$2"
OUTDIR="$3"

PW="/home/10210/Desktop/OS/password.text"
USBMON_NODE="/sys/kernel/debug/usb/usbmon/0u"
CAP="$OUTDIR/usbmon.txt"
TM="$OUTDIR/time_map.json"
IV="$OUTDIR/invokes.json"
OUT_JSON="$OUTDIR/io_split_bt.json"

mkdir -p "$OUTDIR"

if [[ ! -f "$PW" ]]; then
  echo "缺少密码文件: $PW" >&2
  exit 1
fi

# 启动 usbmon 聚合采集
cat "$PW" | sudo -S -p '' modprobe usbmon || true
cat "$PW" | sudo -S -p '' sh -c ": > '$CAP'" || true
cat "$PW" | sudo -S -p '' sh -c "cat '$USBMON_NODE' > '$CAP'" &
CATPID=$!
sleep 0.05

# 同步建立 usbmon_ref 与 CLOCK_BOOTTIME 的映射，并记录 10 次 invoke 的 BOOTTIME 窗口
"/home/10210/Desktop/OS/.venv/bin/python" - "$CAP" "$TM" "$IV" "$MODEL_PATH" <<'PY'
import json,sys,time,re
import numpy as np
from pycoral.utils.edgetpu import make_interpreter

cap, tm_path, iv_path, model = sys.argv[1:]

# 尝试在同一时刻读取 usbmon 最新行的第二列与 BOOTTIME
usb_ts = None
pat = re.compile(r"^\s*\S+\s+([0-9]+(?:\.[0-9]+)?)\s+")
deadline = time.time() + 10.0
while time.time() < deadline and usb_ts is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            lines = f.readlines()
            # 从尾部往前找最近的 C/S 行
            for ln in reversed(lines[-2000:] if len(lines)>2000 else lines):
                m = pat.match(ln)
                if m:
                    v = float(m.group(1))
                    usb_ts = v/1e6 if v>1e6 else v
                    break
    except FileNotFoundError:
        pass
    if usb_ts is None:
        time.sleep(0.05)
if usb_ts is None:
    raise SystemExit('no usbmon ts captured')

try:
    bt_ref = time.clock_gettime(time.CLOCK_BOOTTIME)
except Exception:
    bt_ref = float(open('/proc/uptime').read().split()[0])

json.dump({'usbmon_ref': usb_ts, 'boottime_ref': bt_ref}, open(tm_path,'w'))

# 预热+记录 invoke 窗口（BOOTTIME）
it = make_interpreter(model)
it.allocate_tensors()
inp = it.get_input_details()[0]
if inp['dtype'].__name__=='uint8':
    x = np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x = np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)
for _ in range(50):
    it.set_tensor(inp['index'], x); it.invoke()
spans=[]
for i in range(10):
    it.set_tensor(inp['index'], x)
    t0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.invoke()
    t1=time.clock_gettime(time.CLOCK_BOOTTIME)
    spans.append({'begin': t0, 'end': t1})
json.dump({'name': r"${NAME}", 'spans': spans}, open(iv_path,'w'))
print('SYNC_OK', json.dumps({'usbmon_ref':usb_ts,'boottime_ref':bt_ref,'span0':spans[0]['end']-spans[0]['begin']}))
PY

sleep 0.05
kill "$CATPID" 2>/dev/null || true
sleep 0.05 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$TM" "$IV" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$TM" "$IV" 2>/dev/null || true

# 分析：严格窗口与 ±10ms 扩窗
"/home/10210/Desktop/OS/.venv/bin/python" - "$CAP" "$IV" "$TM" <<'PY' | tee "$OUT_JSON"
import json,sys,re
cap, ivp, tmp = sys.argv[1:]
tm=json.load(open(tmp)); iv=json.load(open(ivp))['spans']
u=float(tm['usbmon_ref']); b=float(tm['boottime_ref'])
recs=[]
with open(cap,'r',errors='ignore') as f:
    for ln in f:
        parts=ln.split()
        if len(parts)<3: continue
        try:
            ts=float(parts[1]); ts=ts/1e6 if ts>1e6 else ts
        except: continue
        tok=None; idx=None
        for i,t in enumerate(parts):
            if re.match(r'^[BC][io]:\d+:', t): tok=t[:2]; idx=i; break
        if not tok: continue
        m=re.search(r'len=(\d+)', ln); nb=0
        if m: nb=int(m.group(1))
        elif idx is not None and idx+2<len(parts):
            try: nb=int(parts[idx+2])
            except: nb=0
        recs.append((ts,tok,nb))

def stat(win, extra=0.0):
    b0 = win['begin'] - b + u - extra
    e0 = win['end']   - b + u + extra
    bi=bo=ci=co=0; binb=boutb=0
    for ts,d,nb in recs:
        if b0<=ts<=e0:
            if d=='Bi': bi+=1; binb+=nb
            elif d=='Bo': bo+=1; boutb+=nb
            elif d=='Ci': ci+=1; binb+=nb
            elif d=='Co': co+=1; boutb+=nb
    span=e0-b0
    toMB=lambda x: x/(1024*1024.0)
    return dict(span_s=span,Bi=bi,Bo=bo,Ci=ci,Co=co,
                bytes_in=binb, bytes_out=boutb,
                MBps_in=(toMB(binb)/span if span>0 else 0.0),
                MBps_out=(toMB(boutb)/span if span>0 else 0.0))

strict=[stat(iv[i],0.0) for i in range(len(iv))]
loose=[stat(iv[i],0.010) for i in range(len(iv))]

def agg(arr):
    s={'bytes_in':0,'bytes_out':0,'span_s':0.0}
    for x in arr:
        s['bytes_in']+=x['bytes_in']; s['bytes_out']+=x['bytes_out']; s['span_s']+=x['span_s']
    toMB=lambda x: x/(1024*1024.0)
    s['MBps_in']=toMB(s['bytes_in'])/s['span_s'] if s['span_s']>0 else 0.0
    s['MBps_out']=toMB(s['bytes_out'])/s['span_s'] if s['span_s']>0 else 0.0
    return s

print(json.dumps({'strict_first':strict[0],'loose_first':loose[0],'strict_overall':agg(strict),'loose_overall':agg(loose)}, ensure_ascii=False, indent=2))
PY

echo "完成，输出: $OUT_JSON"


