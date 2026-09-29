#!/usr/bin/env bash
# Full competition: data v3.1, seed 42, Race, nothink selection.
set -euo pipefail
if [ "$#" -lt 1 ]; then
    echo "Usage: bash scripts/run_competition.sh MODEL_PATH [backend options]" >&2
    exit 2
fi
MODEL_PATH="$1"
shift
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
python3 "${ROOT_DIR}/scripts/ray-vllm/evaluate.py" \
    --model_path "$MODEL_PATH" \
    --data_dir "${BENCHMARK_DATA_DIR:-${ROOT_DIR}/../data}" \
    --output_dir "${BENCHMARK_RESULTS_DIR:-${ROOT_DIR}/results/datav3.1_seed42_race_nothink}" \
    "$@" \
    --version v3.1 --seed 42 --race_mode true --enable_thinking false --sample_size full
