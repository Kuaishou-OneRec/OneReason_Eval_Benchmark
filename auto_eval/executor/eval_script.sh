#!/bin/bash

# Set common variables
MODEL_PATH=$1
ENABLE_THINKING=$3
TASKS=$4  # Comma-separated list of tasks or "all"
ROLLOUT_ARGS=$5 # Extra arguments for rollout mode (e.g. --num_beams 0 ...)
ROLLOUT_MODE=${6:-""}  # Rollout mode string (e.g. "Final", "Normal 128", etc.)
OVERWRITE=${7:-false}  # true or false - whether to overwrite existing results
EXTPARAMS=${8:-""}  # Extra parameters to pass to the evaluation script

# Read configuration from environment variables (set by eval_script.py)
# Fallback to local workspace paths if not set
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-./workspace}"
BENCHMARK_LOG_DIR="${BENCHMARK_LOG_DIR:-./workspace}"
VERSION="${VERSION:-v3.1}"
DATA_VERSION="${DATA_VERSION:-v3.1}"

DATA_DIR="${BENCHMARK_TASK_DATA_DIR:-${BENCHMARK_DATA_DIR}/data/data_${DATA_VERSION}}"

# Set output and log directories
BASE_OUTPUT_DIR="${BENCHMARK_DATA_DIR}/results/${VERSION}/results_${2}"
BASE_LOG_NAME="${BENCHMARK_LOG_DIR}/auto_eval_logs/${VERSION}/$2"

# Create output directory and log directory
mkdir -p "$(dirname "${BASE_LOG_NAME}")"
mkdir -p "$BASE_OUTPUT_DIR"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
cd $SCRIPT_DIR/../..

# Function to check if a task is in the selected tasks list
task_selected() {
    local task=$1
    if [ "$TASKS" = "all" ] || [ -z "$TASKS" ]; then
        return 0  # All tasks selected
    fi
    # Check if task is in the comma-separated list
    if echo ",$TASKS," | grep -q ",$task,"; then
        return 0  # Task is selected
    fi
    return 1  # Task not selected
}

# Build task lists for each group based on selected tasks
build_task_list() {
    local group_tasks="$1"
    local selected=""
    for task in $group_tasks; do
        if task_selected "$task"; then
            if [ -z "$selected" ]; then
                selected="$task"
            else
                selected="$selected $task"
            fi
        fi
    done
    echo "$selected"
}

# Read task groups from environment variables (set by eval_script.py)
# EXECUTOR_CATEGORIES: space-separated list of category names
# EXECUTOR_TASKS_{category}: space-separated task names for each category
EXECUTOR_CATEGORIES="${EXECUTOR_CATEGORIES:-}"

# Per-category execution parameters
get_batch_size() {
    case "$1" in
        recommendation) echo 250 ;;
        *) echo 250 ;;
    esac
}

get_extra_args() {
    case "$1" in
        general) echo "$TEMP_PARAMS" ;;
        recommendation) echo "$ROLLOUT_ARGS" ;;
        *) echo "$ROLLOUT_ARGS" ;;
    esac
}


# Build selected task lists for each category
declare -A CATEGORY_SELECTED
for category in $EXECUTOR_CATEGORIES; do
    var_name="EXECUTOR_TASKS_${category}"
    all_tasks="${!var_name}"
    CATEGORY_SELECTED[$category]=$(build_task_list "$all_tasks")
done

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

# Set temperature parameters based on thinking mode (for general tasks)
if [ "$ENABLE_THINKING" = "true" ]; then
    TEMP_PARAMS="--temperature 0.6 --top_p 0.95 --top_k 20 --presence_penalty 1.0"
else
    TEMP_PARAMS="--temperature 0.7 --top_p 0.8 --top_k 20 --presence_penalty 1.5"
fi

# Write debug info to log file
{
    echo "========== Task Configuration =========="
    echo "VERSION: $VERSION"
    echo "DATA_VERSION: $DATA_VERSION"
    echo "DATA_DIR: $DATA_DIR"
    echo "MODEL_PATH: $MODEL_PATH"
    echo "BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
    echo "Enable Thinking: $ENABLE_THINKING"
    echo "Thinking args: $THINKING_ARGS"
    echo "Temp params: $TEMP_PARAMS"
    echo "Selected Tasks: $TASKS"
    echo "Rollout args: $ROLLOUT_ARGS"
    echo "Rollout mode: $ROLLOUT_MODE"
    echo "Overwrite: $OVERWRITE"
    echo "Categories: $EXECUTOR_CATEGORIES"
    for category in $EXECUTOR_CATEGORIES; do
        batch_size=$(get_batch_size "$category")
        extra_args=$(get_extra_args "$category")
        echo "  [$category] batch=$batch_size extra_args='$extra_args' tasks: ${CATEGORY_SELECTED[$category]}"
    done
    echo "========================================"
} >> "${BASE_LOG_NAME}.log"

echo "Thinking args: $THINKING_ARGS"
echo "Overwrite arg: $OVERWRITE_ARG"


# Run all category groups - only selected tasks
echo "Running selected task groups"

for category in $EXECUTOR_CATEGORIES; do
    selected="${CATEGORY_SELECTED[$category]}"
    if [ -n "$selected" ]; then
        batch_size=$(get_batch_size "$category")
        extra_args=$(get_extra_args "$category")
        CMD="python3 -u scripts/ray-vllm/evaluate.py \
            --task_types $selected \
            --gpu_memory_utilization 0.8 \
            --model_path $MODEL_PATH \
            --data_dir $DATA_DIR \
            --output_dir ${BASE_OUTPUT_DIR} \
            --dtype bfloat16 \
            --worker_batch_size $batch_size \
            --version $VERSION \
            $THINKING_ARGS $OVERWRITE_ARG $extra_args $EXTPARAMS"
        echo "Starting category [$category]: $selected (batch=$batch_size)"
        echo "[CMD] $CMD" >> "${BASE_LOG_NAME}.log"
        eval $CMD >> "${BASE_LOG_NAME}.log" 2>&1

        if [ $? -ne 0 ]; then
            echo "Category [$category] execution failed"
            exit 1
        fi
        echo "Category [$category] completed"
    else
        echo "No tasks selected for category [$category], skipping"
    fi
done

echo "All tasks completed successfully"
