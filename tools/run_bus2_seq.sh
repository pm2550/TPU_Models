#!/usr/bin/env bash
set -euo pipefail

MD="/home/10210/Desktop/OS/layered models/resnet101_balanced/tflite"
OUTROOT="/home/10210/Desktop/OS/results/resnet101_balanced_usbmon"
BUS="2"
SEG_LIST=(0 1 2 3 4 5 6)

PW="/home/10210/Desktop/OS/password.text"
USBNODE="/sys/kernel/debug/usb/usbmon/${BUS}u"

mkdir -p "$OUTROOT"

if [[ ! -f "$PW" ]]; then echo "[PRECHECK] 缺少密码文件: $PW"; exit 1; fi
if [[ ! -e "$USBNODE" ]]; then echo "[PRECHECK] 缺少 usbmon 节点: $USBNODE"; exit 1; fi

echo "[PRECHECK] OK: BUS=$BUS, COUNT=100, DUR=12s"

for S in "${SEG_LIST[@]}"; do
  MODEL="$MD/resnet101_seg${S}_int8.tflite"
  OUTDIR="$OUTROOT/seg${S}"
  mkdir -p "$OUTDIR"
  if [[ ! -f "$MODEL" ]]; then echo "[seg${S}] 缺少模型 $MODEL"; exit 1; fi
  echo "=== CAP seg${S} ==="
  COUNT=100 GET_OUTPUT=1 /home/10210/Desktop/OS/run_usbmon_capture_offline.sh "$MODEL" "resnet101_seg${S}" "$OUTDIR" "$BUS" 12
  echo "=== ANALYZE seg${S} ==="
  python3 /home/10210/Desktop/OS/analyze_usbmon_out_active.py "$OUTDIR/usbmon.txt" "$OUTDIR/invokes.json" "$OUTDIR/time_map.json" > "$OUTDIR/out_active_union.json" || true
  python3 /home/10210/Desktop/OS/analyze_usbmon_in_active.py  "$OUTDIR/usbmon.txt" "$OUTDIR/invokes.json" "$OUTDIR/time_map.json" > "$OUTDIR/in_active_union.json"  || true
  python3 /home/10210/Desktop/OS/summarize_warm_usbmon.py "$OUTDIR/out_active_union.json" "$OUTDIR/in_active_union.json" > "$OUTDIR/warm_summary.json" || true
  echo "--- seg${S} warm 摘要 ---"
  if [[ -f "$OUTDIR/warm_summary.json" ]]; then cat "$OUTDIR/warm_summary.json"; else echo '(无 warm_summary.json)'; fi
  echo
done

echo "[BATCH] 全部完成，目录: $OUTROOT"



