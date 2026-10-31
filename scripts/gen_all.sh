#!/usr/bin/env bash
# Generate the full SFT dataset (A/B/C) and validate it.
#
#   export DEEPSEEK_API_KEY=sk-...
#   bash scripts/gen_all.sh            # sequential, resumable
#
# Notes
#   * runs are resumable: rerun the same command after an interruption
#   * each type has its own budget guard (RMB, off-peak estimate)
#   * off-peak = outside Mon-Fri 09:00-12:00 / 14:00-18:00 (Asia/Shanghai)
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DEEPSEEK_API_KEY:?export DEEPSEEK_API_KEY first}"
WORKERS="${WORKERS:-12}"

# Use the project venv explicitly: `python` may resolve to conda/system python.
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="$(command -v python3)"
fi
echo "python: $PY"

hour=$(date +%H); dow=$(date +%u)
if [ "$dow" -le 5 ] && { [ "$hour" -ge 9 ] && [ "$hour" -lt 12 ] || [ "$hour" -ge 14 ] && [ "$hour" -lt 18 ]; }; then
  echo "!! peak hours now: cost is 2x. Ctrl-C and rerun after 18:00 or on the weekend."
  sleep 5
fi

mkdir -p data/sft

echo "== A: format samples (comparison questions)"
"$PY" scripts/gen_sft_deepseek.py --type A --n 1200 --workers "$WORKERS" \
  --out data/sft/raw_a.jsonl --budget 4

echo "== B: two-hop samples (bridge, 50% oracle-guided first hop)"
"$PY" scripts/gen_sft_deepseek.py --type B --n 2600 --workers "$WORKERS" \
  --out data/sft/raw_b.jsonl --budget 8

echo "== C: abstain samples (fake entities)"
"$PY" scripts/gen_sft_deepseek.py --type C --n 400 --workers 8 \
  --out data/sft/raw_c.jsonl --budget 2

echo "== validate + split"
"$PY" scripts/validate_sft.py \
  --inputs data/sft/raw_a.jsonl data/sft/raw_b.jsonl data/sft/raw_c.jsonl

echo "== done"
wc -l data/sft/train.jsonl data/sft/val.jsonl
