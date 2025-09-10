#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/10210/Desktop/OS"
LDIR="$ROOT/layered models/tpu"
OUT_ROOT="$ROOT/results/layered_usbmon"
BUS="0"  # 0u 全总线
WARMUP="${WARMUP:-10}"
COUNT="${COUNT:-1000}"

mkdir -p "$OUT_ROOT"

shopt -s nullglob
models=("$LDIR"/*.tflite)

for mdl in "${models[@]}"; do
  [ -f "$mdl" ] || continue
  base="$(basename "$mdl" .tflite)"
  outdir="$OUT_ROOT/$base"
  mkdir -p "$outdir"
  echo "== capture $base =="
  WARMUP="$WARMUP" COUNT="$COUNT" bash "$ROOT/run_usbmon_model.sh" "$mdl" "$base" "$outdir" "$BUS"
done

echo "saved to: $OUT_ROOT"

