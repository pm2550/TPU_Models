#!/usr/bin/env bash
set -euo pipefail

# 用法: ./run_usbmon_capture_offline_setget.sh <model_path> <name> <out_dir> <bus> <duration_s>

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
SET_JSON="$OUTDIR/set_spans.json"
INV_JSON="$OUTDIR/invoke_spans.json"
GET_JSON="$OUTDIR/get_spans.json"

# 记录次数（默认 10，可通过环境变量 COUNT 覆盖，例如 COUNT=101）
COUNT="${COUNT:-10}"

# 选择 Python 解释器：
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
sleep 0.2

# 建立 usbmon_ref 与 CLOCK_BOOTTIME 的映射（后台，从尾部反向读取，等待更久）
"$PY_MAP" - "$CAP" "$TM" <<'PY' &
import json,sys,time,re
cap, tm_path = sys.argv[1:]
usb_ts=None
pat=re.compile(r"^\s*\S+\s+([0-9]+(?:\.[0-9]+)?)\s+")
deadline=time.time()+12.0
while time.time()<deadline and usb_ts is None:
    try:
        with open(cap,'r',errors='ignore') as f:
            lines=f.readlines()
            tail = lines[-8000:] if len(lines)>8000 else lines
            for ln in reversed(tail):
                m=pat.match(ln)
                if m:
                    v=float(m.group(1))
                    usb_ts=v/1e6 if v>1e6 else v
                    break
    except FileNotFoundError:
        pass
    if usb_ts is None:
        time.sleep(0.05)
if usb_ts is None:
    open(tm_path,'w').write(json.dumps({'usbmon_ref': None, 'boottime_ref': None}))
    raise SystemExit(0)
try:
    bt_ref=time.clock_gettime(time.CLOCK_BOOTTIME)
except Exception:
    bt_ref=float(open('/proc/uptime').read().split()[0])
open(tm_path,'w').write(json.dumps({'usbmon_ref': usb_ts, 'boottime_ref': bt_ref}))
PY

# 运行模型：记录 COUNT 次 set/invoke/get 的 BOOTTIME 窗口
"$RUN_PY" - "$MODEL_PATH" "$COUNT" <<'PY' > "$SET_JSON.tmp"
import json, time, numpy as np, sys, os
m=sys.argv[1]
try:
    cnt=int(sys.argv[2])
except Exception:
    cnt=10

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
outs=it.get_output_details()
out_idx = outs[0]['index'] if outs else None

if inp['dtype'].__name__=='uint8':
    x=np.random.randint(0,256, size=inp['shape'], dtype=np.uint8)
else:
    x=np.random.randint(-128,128, size=inp['shape'], dtype=np.int8)

# 预热
for _ in range(50):
    it.set_tensor(inp['index'], x)
    it.invoke()
    if out_idx is not None:
        try:
            it.get_tensor(out_idx)
        except Exception:
            pass

set_spans=[]; inv_spans=[]; get_spans=[]
for i in range(cnt):
    t0=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.set_tensor(inp['index'], x)
    t1=time.clock_gettime(time.CLOCK_BOOTTIME)
    set_spans.append({'begin': t0, 'end': t1})

    t2=time.clock_gettime(time.CLOCK_BOOTTIME)
    it.invoke()
    t3=time.clock_gettime(time.CLOCK_BOOTTIME)
    inv_spans.append({'begin': t2, 'end': t3})

    if out_idx is not None:
        t4=time.clock_gettime(time.CLOCK_BOOTTIME)
        try:
            it.get_tensor(out_idx)
        except Exception:
            pass
        t5=time.clock_gettime(time.CLOCK_BOOTTIME)
        get_spans.append({'begin': t4, 'end': t5})

print(json.dumps({'name':'set', 'spans': set_spans}))
print('\n===SPLIT===')
print(json.dumps({'name':'invoke', 'spans': inv_spans}))
print('\n===SPLIT===')
print(json.dumps({'name':'get', 'spans': get_spans}))
PY

# 拆分三段 JSON
awk 'BEGIN{p=0} /===SPLIT===/{p++; next} {print > ((p==0?"'$SET_JSON'":(p==1?"'$INV_JSON'":"'$GET_JSON'")))}' "$SET_JSON.tmp" || true
rm -f "$SET_JSON.tmp"

# 等待到持续时间结束
sleep "$DUR"

# 停止采集并修正权限
kill "$CATPID" 2>/dev/null || true
sleep 0.1 || true
cat "$PW" | sudo -S -p '' chown "$USER:$USER" "$CAP" "$TM" "$SET_JSON" "$INV_JSON" "$GET_JSON" 2>/dev/null || true
cat "$PW" | sudo -S -p '' chmod 0644 "$CAP" "$TM" "$SET_JSON" "$INV_JSON" "$GET_JSON" 2>/dev/null || true

# 若 time_map 中为 None，则在采集完成后兜底重算一次映射
"$PY_ANALYZE" - "$CAP" "$TM" <<'PY'
import json,sys,re
cap, tm_path = sys.argv[1:]
try:
    tm=json.load(open(tm_path))
except Exception:
    tm={'usbmon_ref':None,'boottime_ref':None}
if tm.get('usbmon_ref') is None or tm.get('boottime_ref') is None:
    usb_ts=None
    pat=re.compile(r"^\s*\S+\s+([0-9]+(?:\.[0-9]+)?)\s+")
    with open(cap,'r',errors='ignore') as f:
        lines=f.readlines()
        tail = lines[-16000:] if len(lines)>16000 else lines
        for ln in reversed(tail):
            m=pat.match(ln)
            if m:
                v=float(m.group(1))
                usb_ts=v/1e6 if v>1e6 else v
                break
    if usb_ts is not None:
        try:
            import time as _t
            bt=_t.clock_gettime(_t.CLOCK_BOOTTIME)
        except Exception:
            bt=float(open('/proc/uptime').read().split()[0])
        open(tm_path,'w').write(json.dumps({'usbmon_ref': usb_ts, 'boottime_ref': bt}))
print('OK')
PY

# 分析（set/invoke/get 各自严格窗口与 ±10ms 扩窗总览 + 纯活跃并集）
"$PY_ANALYZE" - "$CAP" "$SET_JSON" "$TM" <<'PY' | tee "$OUTDIR/set_io_split_bt.json"
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

print(json.dumps({'strict_first':strict[0] if strict else {},'loose_first':loose[0] if loose else {},'strict_overall':agg(strict),'loose_overall':agg(loose)}, ensure_ascii=False, indent=2))
PY

"$PY_ANALYZE" /home/10210/Desktop/OS/analyze_usbmon_out_active.py "$CAP" "$SET_JSON" "$TM" > "$OUTDIR/set_out_active_union.json" || true
"$PY_ANALYZE" /home/10210/Desktop/OS/analyze_usbmon_in_active.py  "$CAP" "$SET_JSON" "$TM" > "$OUTDIR/set_in_active_union.json"  || true

"$PY_ANALYZE" - "$CAP" "$INV_JSON" "$TM" <<'PY' | tee "$OUTDIR/invoke_io_split_bt.json"
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

print(json.dumps({'strict_first':strict[0] if strict else {},'loose_first':loose[0] if loose else {},'strict_overall':agg(strict),'loose_overall':agg(loose)}, ensure_ascii=False, indent=2))
PY

"$PY_ANALYZE" /home/10210/Desktop/OS/analyze_usbmon_out_active.py "$CAP" "$INV_JSON" "$TM" > "$OUTDIR/invoke_out_active_union.json" || true
"$PY_ANALYZE" /home/10210/Desktop/OS/analyze_usbmon_in_active.py  "$CAP" "$INV_JSON" "$TM" > "$OUTDIR/invoke_in_active_union.json"  || true

"$PY_ANALYZE" - "$CAP" "$GET_JSON" "$TM" <<'PY' | tee "$OUTDIR/get_io_split_bt.json"
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

print(json.dumps({'strict_first':strict[0] if strict else {},'loose_first':loose[0] if loose else {},'strict_overall':agg(strict),'loose_overall':agg(loose)}, ensure_ascii=False, indent=2))
PY

"$PY_ANALYZE" /home/10210/Desktop/OS/analyze_usbmon_out_active.py "$CAP" "$GET_JSON" "$TM" > "$OUTDIR/get_out_active_union.json" || true
"$PY_ANALYZE" /home/10210/Desktop/OS/analyze_usbmon_in_active.py  "$CAP" "$GET_JSON" "$TM" > "$OUTDIR/get_in_active_union.json"  || true

echo "完成，输出目录: $OUTDIR"


