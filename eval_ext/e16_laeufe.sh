#!/bin/bash
# E16: Sequential 5-loop replication runs of the main evaluation matrix.
cd "$(dirname "$0")/.." || exit 1
OUT=results/e16
mkdir -p "$OUT"
CUTOFF=0545

run() {
  local nr="$1" model="$2" mode="$3" app="$4"
  local now
  now=$(date +%H%M)
  if [ "$now" -ge "$CUTOFF" ]; then
    echo "$(date '+%F %T') Run $nr skipped (after 05:45)" | tee -a "$OUT/ablauf.txt"
    return
  fi
  local log="$OUT/lauf${nr}_${model}_${mode}_${app}.log"
  echo "$(date '+%F %T') Run $nr start: $model $mode $app" | tee -a "$OUT/ablauf.txt"
  python3 auto_screen.py test --app "$app" --model "$model" --mode "$mode" --loops 5 > "$log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') Run $nr end: exit $rc" | tee -a "$OUT/ablauf.txt"
}

run 1 Qwen_trained_e16 som katapp
run 2 Qwen_trained_e16 som bluesky
run 3 Qwen_trained_e16 dg katapp
run 4 Qwen_trained_e16 dg bluesky
run 5 Qwen_untrain_e16 som katapp
run 6 Qwen_untrain_e16 som bluesky
echo "$(date '+%F %T') E16 finished" | tee -a "$OUT/ablauf.txt"
