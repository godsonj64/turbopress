#!/usr/bin/env bash
# Run every method in its own venv (avoids transformers/CUDA version conflicts),
# then print the merged table. Linux GPU box. Usage:  bash run_all.sh [MODEL]
set -u
MODEL="${1:-Qwen/Qwen3-1.7B}"
BASE="transformers>=4.51 datasets accelerate"
mkdir -p results

run() {                       # run <venv> <pip-extras> <command...>
  local name="$1"; shift
  local extras="$1"; shift
  echo "==================  $name  =================="
  python -m venv ".venv_$name" || return
  # shellcheck disable=SC1090
  . ".venv_$name/bin/activate"
  pip install -q --upgrade pip
  # shellcheck disable=SC2086
  pip install -q $BASE $extras || { echo "  install failed for $name"; deactivate; return; }
  "$@" || echo "  $name run failed (continuing)"
  deactivate
}

run tp   "turbopress[llm]>=0.4.3" python run_turbopress.py --model "$MODEL" --bits 2,3,4
run hqq  "hqq"                    python run_hqq.py        --model "$MODEL" --bits 2,3,4
run gptq "gptqmodel"              python run_gptq.py       --model "$MODEL" --bits 2,3,4
run bnb  "bitsandbytes"           python run_bnb.py        --model "$MODEL"
run awq  "autoawq"                python run_awq.py        --model "$MODEL"

echo
python -m venv .venv_cmp && . .venv_cmp/bin/activate && pip install -q >/dev/null 2>&1
python compare.py
