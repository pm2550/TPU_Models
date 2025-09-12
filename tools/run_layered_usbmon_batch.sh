#!/usr/bin/env bash
set -euo pipefail

# 批量对 layered models/tpu 下的所有 .tflite 进行 USBMon 采集
# - warm: 预热后记录100次
# - cold: 轮换模型，每次只记录1次，直至每模型累计100次

ROOT="/home/10210/Desktop/OS"
LDIR="$ROOT/layered models/tpu"
OUT_BASE="$ROOT/results/layered_usbmon"
BUS="${BUS:-2}"

if [[ ! -d "$LDIR" ]]; then
  echo "missing layered models dir: $LDIR" >&2; exit 1
fi

mapfile -t MODELS < <(find "$LDIR" -maxdepth 1 -type f -name '*edgetpu.tflite' | sort)
if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "no tflite models found under $LDIR" >&2; exit 1
fi

echo "== models =="
for m in "${MODELS[@]}"; do echo "$m"; done

# warm: 预热10，记录100
echo "== warm run (WARMUP=10 COUNT=100) =="
for mdl in "${MODELS[@]}"; do
  name=$(basename "${mdl%.tflite}")
  od="$OUT_BASE/$name/warm"
  mkdir -p "$od"
  echo "-- warm $name -> $od"
  WARMUP=10 COUNT=100 CAP_DUR=12 bash "$ROOT/run_usbmon_model.sh" "$mdl" "$name" "$od" "$BUS" | sed -n '1,4p'
done

# cold: 轮换模型，每次 COUNT=1，直至每模型100次
echo "== cold run (rotate models, COUNT=1 × 100 each) =="
TARGET=100
declare -A CNT
for mdl in "${MODELS[@]}"; do CNT["$mdl"]=0; done

round=0
while :; do
  done_all=1
  for mdl in "${MODELS[@]}"; do
    cur=${CNT["$mdl"]}
    if (( cur < TARGET )); then
      done_all=0
      name=$(basename "${mdl%.tflite}")
      cur=$((cur+1)); CNT["$mdl"]=$cur
      od="$OUT_BASE/$name/cold/run$(printf '%03d' "$cur")"
      mkdir -p "$od"
      echo "-- cold $name #$cur -> $od"
      WARMUP=0 COUNT=1 CAP_DUR=6 bash "$ROOT/run_usbmon_model.sh" "$mdl" "$name" "$od" "$BUS" | sed -n '1,3p'
    fi
  done
  (( done_all == 1 )) && break
  round=$((round+1))
done

echo "== batch done =="

