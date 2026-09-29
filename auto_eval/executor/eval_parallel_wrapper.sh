#!/bin/bash

# ======================================================================
# Wrapper Script for eval_parallel.sh
# ======================================================================
# Purpose: Adapt eval_script.py parameters to eval_parallel.sh
#
# This script receives parameters from eval_script.py and generates
# a task configuration file for the scheduler.
# ======================================================================

# Receive parameters from eval_script.py
# $1: model_path
# $2: output_suffix
# $3: enable_thinking (true/false)
# $4: tasks (comma-separated or "all")
# $5: rollout_args
# $6: rollout_mode (e.g. "Final", "Normal 128", etc.)
# $7: overwrite (true/false)
MODEL_PATH="$1"
OUTPUT_SUFFIX="$2"
ENABLE_THINKING="$3"
TASKS="$4"
ROLLOUT_ARGS="$5"
ROLLOUT_MODE="${6:-""}"
OVERWRITE="${7:-false}"

# Get script directory
SCRIPT_DIR=$(cd $(dirname $0); pwd)

# Read environment variables set by eval_script.py
DATA_VERSION="${DATA_VERSION:-v3.1}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-./workspace}"
BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-./workspace}"
DATA_DIR="${BENCHMARK_TASK_DATA_DIR:-${BENCHMARK_DATA_DIR}/data/data_${DATA_VERSION}}"
VERSION="${VERSION:-v3.1}"

# Set output and log directories
BASE_OUTPUT_DIR="${BENCHMARK_DATA_DIR}/results/${VERSION}/results_${OUTPUT_SUFFIX}"
BASE_LOG_DIR="${BENCHMARK_LOG_DIR}/auto_eval_logs/${VERSION}"

# Set parallel execution log directory (same as what will be passed to eval_parallel.sh)
PARALLEL_LOG_DIR="${BASE_LOG_DIR}/parallel_${OUTPUT_SUFFIX}"

# Ensure log directory exists
mkdir -p "${PARALLEL_LOG_DIR}"

# Create temporary task config file in parallel log directory
TEMP_CONFIG="${PARALLEL_LOG_DIR}/task_config.txt"

echo "Generating task configuration for the scheduler..."
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Output: $BASE_OUTPUT_DIR"

# Build thinking arguments
THINKING_ARGS=""
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS="--enable_thinking"
fi

# Build overwrite argument
OVERWRITE_ARG=""
if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_ARG="--overwrite"
fi

# Set temperature parameters based on thinking mode
if [ "$ENABLE_THINKING" = "true" ]; then
    TEMP_PARAMS="--temperature 0.6 --top_p 0.95 --top_k 20 --presence_penalty 1.0"
else
    TEMP_PARAMS="--temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5"
fi

# Parse tasks and create task configuration
# Convert comma-separated tasks or "all" to individual task commands

# Change to project root
cd $SCRIPT_DIR/../..

# Read task groups from environment variables (set by eval_script.py)
EXECUTOR_CATEGORIES="${EXECUTOR_CATEGORIES:-}"

# Per-category execution parameters
get_batch_size() {
    case "$1" in
        recommendation) echo 250 ;;
        *) echo 250 ;;
    esac
}

# Build ALL_TASKS from all categories
ALL_TASKS=""
for category in $EXECUTOR_CATEGORIES; do
    var_name="EXECUTOR_TASKS_${category}"
    ALL_TASKS="$ALL_TASKS ${!var_name}"
done

# Determine which tasks to run
if [ "$TASKS" = "all" ] || [ -z "$TASKS" ]; then
    SELECTED_TASKS="$ALL_TASKS"
else
    # Convert comma-separated to space-separated
    SELECTED_TASKS=$(echo "$TASKS" | tr ',' ' ')
fi

# Build reverse lookup: task -> category
declare -A TASK_CATEGORY
for category in $EXECUTOR_CATEGORIES; do
    var_name="EXECUTOR_TASKS_${category}"
    for task in ${!var_name}; do
        TASK_CATEGORY[$task]="$category"
    done
done

# Generate task config file
> "$TEMP_CONFIG"  # Clear file

echo "# Auto-generated task configuration for output_suffix: $OUTPUT_SUFFIX" >> "$TEMP_CONFIG"
echo "# Generated at: $(date)" >> "$TEMP_CONFIG"
echo "" >> "$TEMP_CONFIG"

# For each selected task, generate a command
for task in $SELECTED_TASKS; do
    category="${TASK_CATEGORY[$task]:-}"
    if [ -z "$category" ]; then
        echo "Warning: task '$task' not found in any category, skipping"
        continue
    fi

    GPU_MEM="0.8"
    BATCH_SIZE=$(get_batch_size "$category")

    # Build the command
    CMD="python3 -u scripts/ray-vllm/evaluate.py"
    CMD="$CMD --task_types $task"
    CMD="$CMD --gpu_memory_utilization $GPU_MEM"
    CMD="$CMD --model_path \"$MODEL_PATH\""
    CMD="$CMD --data_dir \"$BENCHMARK_DATA_DIR\""
    CMD="$CMD --output_dir \"$BASE_OUTPUT_DIR\""
    CMD="$CMD --dtype bfloat16"
    CMD="$CMD --worker_batch_size $BATCH_SIZE"
    CMD="$CMD --ray_address local"

    # Add optional arguments
    if [ -n "$THINKING_ARGS" ]; then
        CMD="$CMD $THINKING_ARGS"
    fi
    if [ -n "$OVERWRITE_ARG" ]; then
        CMD="$CMD $OVERWRITE_ARG"
    fi

    # Add category-specific arguments
    if [ "$category" = "general" ]; then
        CMD="$CMD $TEMP_PARAMS"
    fi

    # Add rollout args for all tasks
    if [ -n "$ROLLOUT_ARGS" ]; then
        CMD="$CMD $ROLLOUT_ARGS"
    fi

    # Print the command for debugging
    echo "[CMD] $task: $CMD" >> "${PARALLEL_LOG_DIR}/commands.log"

    # Add to config file (format: task_name|||command)
    echo "[CMD] $task ($category, batch=$BATCH_SIZE): $CMD"
    echo "$task|||$CMD" >> "$TEMP_CONFIG"
done

echo "Task configuration generated: $TEMP_CONFIG"
echo "Total tasks: $(grep -c '|||' "$TEMP_CONFIG")"

# Call eval_parallel.sh
bash "${SCRIPT_DIR}/eval_parallel.sh" \
    --config "$TEMP_CONFIG" \
    --log_dir "${PARALLEL_LOG_DIR}"

# Capture exit code
EXIT_CODE=$?
# Config file is kept in log directory for reference
# Location: ${PARALLEL_LOG_DIR}/task_config.txt

# Exit with the same code as the script
exit $EXIT_CODE
