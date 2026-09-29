#!/bin/bash
# General sweep script for think / nothink inference benchmarking.
#
# Supports sweeping over:
#   1. model size / model path
#   2. task
#   3. worker batch size
#   4. beam size
#   5. mode (think / nothink)
#
# Default behavior is SERIAL and safest:
#   - one probe at a time
#   - one local Ray per probe
#
# Parallel behavior is opt-in and uses a SHARED Ray cluster:
#   - one worker loop per GPU
#   - each GPU loop runs its assigned probes serially
#   - each probe connects to the same Ray cluster with --ray_address auto
#   - avoids spawning multiple independent local Ray clusters concurrently
#
# Example:
#   MODEL_SPECS="1.7b=/path/to/1.7b 0.6b=/path/to/0.6b" \
#   TASKS="video_v2.1 locallife_v0419.4" \
#   BEAMS="32 64" \
#   BATCHES="64 128 256 512" \
#   MODES="think nothink" \
#   PARALLEL=0 \
#   bash compare_think_nothink.sh

set -u

# ---- Configurable env vars ----------------------------------------------------
DEFAULT_MODEL_PATH="./models/model"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
MODEL_SPECS="${MODEL_SPECS:-default=${MODEL_PATH}}"

BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-./workspace}"
VERSION="${VERSION:-v3.1}"
DATA_VERSION="${DATA_VERSION:-v3.1}"

TASK="${TASK:-video_v2.1}"
TASKS="${TASKS:-$TASK}"
BEAM="${BEAM:-64}"
BEAMS="${BEAMS:-$BEAM}"
BATCHES="${BATCHES:-64 128 256 512}"
MODES="${MODES:-think nothink}"

SAMPLE_SIZE="${SAMPLE_SIZE:-}"              # empty = full data
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
SEED="${SEED:-42}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"

GPU_ID="${GPU_ID:-0}"                      # serial mode GPU
PARALLEL="${PARALLEL:-0}"                  # 1 = shard combos across GPUs
PARALLEL_GPUS="${PARALLEL_GPUS:-0 1 2 3 4 5 6 7}"
STAGGER_SECONDS="${STAGGER_SECONDS:-10}"
RAY_ADDRESS="${RAY_ADDRESS:-local}"        # local for serial; auto for shared-cluster parallel
AUTO_START_SHARED_RAY="${AUTO_START_SHARED_RAY:-1}"
SHARED_RAY_NUM_GPUS="${SHARED_RAY_NUM_GPUS:-}"
DRY_RUN="${DRY_RUN:-0}"

TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BENCHMARK_DATA_DIR}/think_compare/${VERSION}/sweep_${TS}}"
CSV_PATH="${CSV_PATH:-${OUTPUT_ROOT}/results.csv}"

# ---- Setup --------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd)"
cd "$SCRIPT_DIR/../.."

mkdir -p "$OUTPUT_ROOT"
CSV_HEADER="model_tag,model_path,task,mode,beam,batch,gpu,ray_address,result,wall_time_s,infer_time_s,input_tokens,output_tokens,decode_tok_per_s,peak_gpu_mem_mb,notes"
echo "$CSV_HEADER" > "$CSV_PATH"

DATA_DIR="${BENCHMARK_TASK_DATA_DIR:-${BENCHMARK_DATA_DIR}/data/data_${DATA_VERSION}}"
COMBOS=()

echo "========== Config =========="
echo "MODEL_SPECS:       $MODEL_SPECS"
echo "TASKS:             $TASKS"
echo "BEAMS:             $BEAMS"
echo "BATCHES:           $BATCHES"
echo "MODES:             $MODES"
echo "BENCHMARK_DATA_DIR:$BENCHMARK_DATA_DIR"
echo "DATA_DIR:          $DATA_DIR"
echo "VERSION:           $VERSION"
echo "PARALLEL:          $PARALLEL"
echo "PARALLEL_GPUS:     $PARALLEL_GPUS"
echo "GPU_ID:            $GPU_ID"
echo "RAY_ADDRESS:       $RAY_ADDRESS"
echo "AUTO_START_SHARED_RAY: $AUTO_START_SHARED_RAY"
echo "SAMPLE_SIZE:       ${SAMPLE_SIZE:-<full data>}"
echo "GPU_MEM_UTIL:      $GPU_MEM_UTIL"
echo "OUTPUT_ROOT:       $OUTPUT_ROOT"
echo "CSV:               $CSV_PATH"
echo "DRY_RUN:           $DRY_RUN"
echo "============================"

# ---- Helpers ------------------------------------------------------------------
sanitize_tag() {
    printf '%s' "$1" | tr '/: ' '___' | tr -cd '[:alnum:]_.=-'
}

short_hash() {
    if command -v md5sum >/dev/null 2>&1; then
        printf '%s' "$1" | md5sum | cut -c1-8
    else
        printf '%s' "$1" | md5 | awk '{print substr($NF, 1, 8)}'
    fi
}

extract_model_tag() {
    local spec=$1
    if [[ "$spec" == *=* ]]; then
        printf '%s\n' "${spec%%=*}"
    else
        basename "$spec"
    fi
}

extract_model_path() {
    local spec=$1
    if [[ "$spec" == *=* ]]; then
        printf '%s\n' "${spec#*=}"
    else
        printf '%s\n' "$spec"
    fi
}

ensure_shared_ray_cluster() {
    if [ "$RAY_ADDRESS" != "auto" ]; then
        return 0
    fi
    if ray status >/dev/null 2>&1; then
        echo "[ray] existing shared cluster detected, reusing it"
        return 0
    fi
    if [ "$AUTO_START_SHARED_RAY" != "1" ]; then
        echo "[ray] ERROR: RAY_ADDRESS=auto but no existing cluster detected and AUTO_START_SHARED_RAY=0" >&2
        return 1
    fi

    local num_gpus_to_start
    if [ -n "$SHARED_RAY_NUM_GPUS" ]; then
        num_gpus_to_start="$SHARED_RAY_NUM_GPUS"
    else
        local gpu_arr=($PARALLEL_GPUS)
        num_gpus_to_start="${#gpu_arr[@]}"
        if [ "$num_gpus_to_start" -le 0 ]; then
            num_gpus_to_start=1
        fi
    fi

    echo "[ray] starting shared local cluster with $num_gpus_to_start GPUs"
    ray start --head --num-gpus="$num_gpus_to_start" >/dev/null
    sleep 3
    ray status >/dev/null 2>&1
}

build_combos() {
    local model_spec model_tag model_path task mode beam batch
    for model_spec in $MODEL_SPECS; do
        model_tag=$(extract_model_tag "$model_spec")
        model_path=$(extract_model_path "$model_spec")
        for task in $TASKS; do
            for mode in $MODES; do
                for beam in $BEAMS; do
                    for batch in $BATCHES; do
                        COMBOS+=("${model_tag}|${model_path}|${task}|${mode}|${beam}|${batch}")
                    done
                done
            done
        done
    done
}

append_row() {
    local row=$1
    local row_file=$2
    echo "$row" > "$row_file"
}

print_command_preview() {
    local cmd=$1
    echo "[dry-run] $cmd"
}

# ---- Run one probe ------------------------------------------------------------
run_one() {
    local model_tag=$1
    local model_path=$2
    local task=$3
    local mode=$4
    local beam=$5
    local batch=$6
    local phys_gpu=$7

    local safe_model_tag safe_task trial_dir log gpu_log row_file
    safe_model_tag=$(sanitize_tag "$model_tag")
    safe_task=$(sanitize_tag "$task")
    trial_dir="${OUTPUT_ROOT}/${safe_model_tag}/${safe_task}/${mode}_beam${beam}_bs${batch}"
    log="${trial_dir}/run.log"
    gpu_log="${trial_dir}/gpu.log"
    row_file="${trial_dir}/row.csv"
    mkdir -p "$trial_dir"

    local thinking_arg=""
    local extra_beam_args=""
    local sampling_args=""
    if [ "$mode" = "think" ]; then
        thinking_arg="--enable_thinking"
        extra_beam_args="--num_return_thinking_sequences 1"
        sampling_args="--temperature 0.6 --top_p 0.95 --top_k 20 --presence_penalty 1.0"
    else
        sampling_args="--temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5"
    fi

    local max_lp=384
    if [ $(( beam * 2 )) -gt $max_lp ]; then max_lp=$(( beam * 2 )); fi

    local sample_size_arg=""
    if [ -n "$SAMPLE_SIZE" ]; then sample_size_arg="--sample_size $SAMPLE_SIZE"; fi

    local max_model_len_arg=""
    if [ -n "$MAX_MODEL_LEN" ]; then max_model_len_arg="--max_model_len $MAX_MODEL_LEN"; fi

    local nvsmi_pid=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$phys_gpu" -l 1 \
            > "$gpu_log" 2>/dev/null &
        nvsmi_pid=$!
    fi

    local ray_mode_args=""
    local launch_prefix=""
    local ray_tmp=""
    if [ "$RAY_ADDRESS" = "auto" ]; then
        ray_mode_args="--ray_address auto --gpu_ids $phys_gpu --num_gpus 1"
    else
        local ray_key ray_hash
        ray_key="${model_tag}|${task}|${mode}|${beam}|${batch}|${phys_gpu}|$$"
        ray_hash=$(short_hash "$ray_key")
        ray_tmp="/tmp/raycmp_${ray_hash}"
        mkdir -p "$ray_tmp"
        launch_prefix="RAY_TMPDIR=\"$ray_tmp\" CUDA_VISIBLE_DEVICES=\"$phys_gpu\""
        ray_mode_args="--ray_address local --num_gpus 1"
    fi

    local beam_args=""
    if [ "$beam" -gt 1 ]; then
        beam_args="--num_beams \"$beam\" --num_return_sequences \"$beam\" --max_logprobs \"$max_lp\""
    fi

    local cmd="python3 -u scripts/ray-vllm/evaluate.py \
        --task_types \"$task\" \
        --version \"$VERSION\" \
        --model_path \"$model_path\" \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"${trial_dir}/results\" \
        --dtype bfloat16 \
        --gpu_memory_utilization \"$GPU_MEM_UTIL\" \
        --tensor_parallel_size 1 \
        --worker_batch_size \"$batch\" \
        $beam_args \
        $thinking_arg $extra_beam_args $sampling_args \
        $sample_size_arg \
        $max_model_len_arg \
        --seed \"$SEED\" \
        $ray_mode_args \
        --overwrite"

    echo "[start] model=$model_tag task=$task mode=$mode beam=$beam batch=$batch gpu=$phys_gpu"

    local t0 t1 rc
    if [ "$DRY_RUN" = "1" ]; then
        print_command_preview "$launch_prefix $cmd"
        rc=0
        t0=$(date +%s.%N)
        t1="$t0"
    else
        t0=$(date +%s.%N)
        if [ -n "$launch_prefix" ]; then
            eval "$launch_prefix $cmd" > "$log" 2>&1
        else
            eval "$cmd" > "$log" 2>&1
        fi
        rc=$?
        t1=$(date +%s.%N)
    fi

    if [ -n "$nvsmi_pid" ]; then
        kill "$nvsmi_pid" 2>/dev/null || true
        wait "$nvsmi_pid" 2>/dev/null || true
    fi

    local wall peak_mem result notes infer_time input_tok output_tok decode_tps
    wall=$(python3 -c "print(f'{$t1 - $t0:.2f}')")
    if [ -s "$gpu_log" ]; then
        peak_mem=$(sort -n "$gpu_log" 2>/dev/null | tail -1)
    else
        peak_mem=""
    fi

    infer_time=""
    input_tok=""
    output_tok=""
    decode_tps=""
    if [ "$DRY_RUN" = "1" ]; then
        result="DRY_RUN"
        notes=""
    elif [ $rc -eq 0 ]; then
        local gen_json
        gen_json=$(find "${trial_dir}/results" -name '*_generated.json' 2>/dev/null | head -1)
        if [ -n "$gen_json" ] && [ -f "$gen_json" ]; then
            read -r infer_time input_tok output_tok decode_tps <<<"$(python3 - "$gen_json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
agg = d.get("mfu_stats_aggregate", {}) or {}
t = agg.get("total_time", []) or []
ti = agg.get("total_input_tokens", []) or []
to = agg.get("total_output_tokens", []) or []
infer = sum(t) if t else 0.0
in_tok = sum(ti) if ti else 0
out_tok = sum(to) if to else 0
tps = (out_tok / infer) if infer > 0 else 0.0
print(f"{infer:.2f} {in_tok} {out_tok} {tps:.2f}")
PY
)"
        fi
        result="OK"
        notes=""
    else
        result="FAIL"
        if grep -qi -e "out of memory" -e "OutOfMemoryError" -e "CUDA OOM" "$log" 2>/dev/null; then
            notes="OOM"
        elif grep -qi "Failed to connect to GCS" "$log" 2>/dev/null; then
            notes="GCS_CONNECT_FAIL"
        elif grep -qi "RayActorError" "$log" 2>/dev/null; then
            notes="RayActorError"
        else
            notes="non-OOM fail (rc=$rc)"
        fi
    fi

    append_row \
        "${model_tag},${model_path},${task},${mode},${beam},${batch},${phys_gpu},${RAY_ADDRESS},${result},${wall},${infer_time},${input_tok},${output_tok},${decode_tps},${peak_mem},${notes}" \
        "$row_file"

    echo "[done] model=$model_tag task=$task mode=$mode beam=$beam batch=$batch gpu=$phys_gpu -> $result wall=${wall}s infer=${infer_time}s decode_tps=${decode_tps} peak_mem=${peak_mem}MB ${notes}"
}

run_gpu_shard() {
    local shard_index=$1
    local shard_count=$2
    local phys_gpu=$3
    local combo
    local idx=0
    for combo in "${COMBOS[@]}"; do
        if [ $(( idx % shard_count )) -ne "$shard_index" ]; then
            idx=$(( idx + 1 ))
            continue
        fi
        IFS='|' read -r model_tag model_path task mode beam batch <<< "$combo"
        run_one "$model_tag" "$model_path" "$task" "$mode" "$beam" "$batch" "$phys_gpu"
        idx=$(( idx + 1 ))
    done
}

# ---- Main ---------------------------------------------------------------------
build_combos

if [ "${#COMBOS[@]}" -eq 0 ]; then
    echo "[ERROR] no combos built; check MODEL_SPECS/TASKS/BEAMS/BATCHES/MODES" >&2
    exit 1
fi

echo "[combos] total=${#COMBOS[@]}"

if [ "$PARALLEL" = "1" ]; then
    if [ "$RAY_ADDRESS" != "auto" ]; then
        echo "[ERROR] PARALLEL=1 requires RAY_ADDRESS=auto to avoid concurrent local Ray clusters" >&2
        exit 1
    fi
    ensure_shared_ray_cluster

    GPU_ARR=($PARALLEL_GPUS)
    NUM_GPUS_AVAIL=${#GPU_ARR[@]}
    if [ "$NUM_GPUS_AVAIL" -le 0 ]; then
        echo "[ERROR] PARALLEL=1 but PARALLEL_GPUS is empty" >&2
        exit 1
    fi

    pids=()
    for i in "${!GPU_ARR[@]}"; do
        phys=${GPU_ARR[$i]}
        ( run_gpu_shard "$i" "$NUM_GPUS_AVAIL" "$phys" ) &
        pids+=($!)
        if [ $i -lt $(( NUM_GPUS_AVAIL - 1 )) ] && [ "$STAGGER_SECONDS" -gt 0 ]; then
            sleep "$STAGGER_SECONDS"
        fi
    done
    echo "[parallel] launched ${#pids[@]} GPU workers; waiting ..."
    for pid in "${pids[@]}"; do wait "$pid"; done
else
    run_gpu_shard 0 1 "$GPU_ID"
fi

# ---- Merge rows ----------------------------------------------------------------
for combo in "${COMBOS[@]}"; do
    IFS='|' read -r model_tag _ task mode beam batch <<< "$combo"
    row_file="${OUTPUT_ROOT}/$(sanitize_tag "$model_tag")/$(sanitize_tag "$task")/${mode}_beam${beam}_bs${batch}/row.csv"
    if [ -f "$row_file" ]; then
        cat "$row_file" >> "$CSV_PATH"
    fi
done

echo ""
echo "========== Done =========="
echo "CSV: $CSV_PATH"
cat "$CSV_PATH"

echo ""
echo "========== Best Config Per (model, task, mode) =========="
python3 - "$CSV_PATH" <<'PY'
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1])))
ok_rows = [r for r in rows if r["result"] == "OK" and r["decode_tok_per_s"]]
groups = {}
for row in ok_rows:
    key = (row["model_tag"], row["task"], row["mode"])
    groups.setdefault(key, []).append(row)

for key in sorted(groups):
    model_tag, task, mode = key
    rs = groups[key]
    best_tps = max(rs, key=lambda r: float(r["decode_tok_per_s"]))
    best_infer = min(rs, key=lambda r: float(r["infer_time_s"]) if r["infer_time_s"] else 1e18)
    print(f"[model={model_tag} task={task} mode={mode}]")
    print(
        f"  max throughput: beam={best_tps['beam']} batch={best_tps['batch']} "
        f"decode_tok_per_s={best_tps['decode_tok_per_s']} infer_time_s={best_tps['infer_time_s']}"
    )
    print(
        f"  min infer_time: beam={best_infer['beam']} batch={best_infer['batch']} "
        f"infer_time_s={best_infer['infer_time_s']} decode_tok_per_s={best_infer['decode_tok_per_s']}"
    )
PY
