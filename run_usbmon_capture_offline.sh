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

# 启动映射记录器：读取首条 usbmon 行的第二列，与同刻 BOOTTIME 建立映射
"$PY_MAP" - "$CAP" "$TM" <<'PY' &
import json,sys,time,re
cap, tm_path = sys.argv[1:]
usb_ts=None
pat=re.compile(r"^\s*\S+\s+([0-9]+(?:\.[0-9]+)?)\s+")
deadline=time.time()+10.0
while time.time()<deadline and usb_ts is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            lines=f.readlines()
            for ln in lines:
                m=pat.match(ln)
                if m:
                    v=float(m.group(1))
                    usb_ts=v/1e6 if v>1e6 else v
                    break
    except FileNotFoundError:
        pass
    if usb_ts is None:
        time.sleep(0.02)
if usb_ts is None:
    open(tm_path,'w').write(json.dumps({'usbmon_ref': None, 'boottime_ref': None}))
    raise SystemExit(0)
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
deadline=time.time()+5.0
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
PY

# 采集前预留窗口（可通过 LEAD_S 指定，默认 0）
sleep "${LEAD_S:-0}"

# 运行模型：记录 COUNT 次 invoke 的 EPOCH 窗口（改为 CLOCK_REALTIME，避免依赖 time_map）
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
for _ in range(10):
    it.set_tensor(inp['index'], x); it.invoke()
spans=[]
try:
    cnt=int(os.environ.get('COUNT','10'))
except Exception:
    cnt=10
for i in range(cnt):
    it.set_tensor(inp['index'], x)
    t0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.invoke()
    t1=time.clock_gettime(time.CLOCK_BOOTTIME)
    spans.append({'begin': t0, 'end': t1})
print(json.dumps({'name': r"${NAME}", 'spans': spans}))
PY

# 等待到持续时间结束
sleep "$DUR"

# 停止采集并修正权限
kill "$CATPID" 2>/dev/null || true
sleep 0.1 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$TM" "$IV" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$TM" "$IV" 2>/dev/null || true

# 分析（严格窗口与 ±10ms 扩窗）
"$PY_ANALYZE" - "$CAP" "$IV" "$TM" <<'PY' | tee "$OUT_JSON"
import json,sys,re,os
cap, ivp, tmp = sys.argv[1:]
tm=json.load(open(tmp)) if os.path.exists(tmp) else {'usbmon_ref':None,'boottime_ref':None}
iv=json.load(open(ivp))['spans']
usb_ref=tm.get('usbmon_ref'); bt_ref=tm.get('boottime_ref')
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
    if usb_ref is None or bt_ref is None:
        return {'note':'no_ref','span_s':win['end']-win['begin'],'Bi':0,'Bo':0,'Ci':0,'Co':0,'bytes_in':0,'bytes_out':0,'MBps_in':0.0,'MBps_out':0.0}
    b0 = win['begin'] - bt_ref + usb_ref - extra
    e0 = win['end']   - bt_ref + usb_ref + extra
    bi=bo=ci=co=0; binb=boutb=0
    for ts,d,nb in recs:
        if b0<=ts<=e0:
            if d=='Bi': bi+=1; binb+=nb
            elif d=='Bo': bo+=1; boutb+=nb
            elif d=='Ci': ci+=1; binb+=nb
            elif d=='Co': co+=1; boutb+=nb
    span=e0-b0
    toMB=lambda x: x/(1024*1024.0)
    return dict(span_s=span,Bi=bi,Bo=bo,Ci=ci,Co=co,bytes_in=binb,bytes_out=boutb,MBps_in=(toMB(binb)/span if span>0 else 0.0),MBps_out=(toMB(boutb)/span if span>0 else 0.0))

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


