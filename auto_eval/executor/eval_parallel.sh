#!/bin/bash

# ======================================================================
# Multi-Processor Parallel Evaluation Script
# ======================================================================
# Purpose: Execute AIR-Bench evaluation tasks in parallel on N processors
# 
# Architecture:
# - N processors (each processor = multiple machines forming a Ray cluster)
# - Centralized task queue on processor 0's head node
# - Each processor runs 1 task at a time (using all GPUs in that processor)
# - Independent Ray cluster per processor
# - Dynamic task allocation based on processor availability
#
# Usage:
#   bash eval_parallel.sh --config task_config.txt
# ======================================================================

set -e  # Exit on error

# ======================================================================
# Configuration
# ======================================================================

# Read configuration from environment variables (compatible with eval_script.py)
# Fallback to defaults if not set

# Multi-processor configuration
NUM_PROCESSORS="${NUM_PROCESSORS:-2}"  # Number of processors
MACHINES_PER_PROCESSOR="${MACHINES_PER_PROCESSOR:-1}"  # Machines per processor

# Default values
CONFIG_FILE=""
LOG_DIR_DEFAULT="${BENCHMARK_LOG_DIR:-./workspace}/auto_eval_logs/${VERSION}/parallel"
LOG_DIR=""
RAY_PORT=6379
CONDA_ENV="verl"
PROJECT_ROOT=$(cd $(dirname $0)/../..; pwd)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --ray_port)
            RAY_PORT="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --config <task_config_file> [--log_dir <dir>] [--ray_port <port>] [--conda_env <env>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: --config is required"
    echo "Usage: $0 --config <task_config_file>"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ======================================================================
# Helper Functions (defined early for use throughout script)
# ======================================================================

# Log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# ======================================================================
# Initialize
# ======================================================================

# Set log directory (use --log_dir if provided, otherwise use default)
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$LOG_DIR_DEFAULT"
fi

# Create log directory
mkdir -p "$LOG_DIR"

# Main log file
MAIN_LOG="$LOG_DIR/main.log"
echo "=== Multi-Processor Parallel Evaluation Started at $(date) ===" | tee "$MAIN_LOG"

# Log configuration
{
    echo "========== Configuration =========="
    echo "LOG_DIR: $LOG_DIR"
    echo "PROJECT_ROOT: $PROJECT_ROOT"
    echo "CONDA_ENV: $CONDA_ENV"
    echo "RAY_PORT: $RAY_PORT"
    echo "NUM_PROCESSORS: $NUM_PROCESSORS"
    echo "MACHINES_PER_PROCESSOR: $MACHINES_PER_PROCESSOR"
    echo "===================================="
} | tee -a "$MAIN_LOG"

# Read machine list from hostfile
if [[ ! -f /etc/mpi/hostfile ]]; then
    echo "Error: /etc/mpi/hostfile not found" | tee -a "$MAIN_LOG"
    exit 1
fi

# Backup original hostfile (will be overwritten during processor initialization)
HOSTFILE_BACKUP="/etc/mpi/hostfile_backup"
cp /etc/mpi/hostfile "$HOSTFILE_BACKUP"
log "Backed up original hostfile to $HOSTFILE_BACKUP"

# Read all machines from hostfile
ALL_MACHINES=($(awk '{print $1}' /etc/mpi/hostfile))
TOTAL_MACHINES=${#ALL_MACHINES[@]}

log "Total machines in hostfile: $TOTAL_MACHINES"
log "Number of processors: $NUM_PROCESSORS"
log "Machines per processor: $MACHINES_PER_PROCESSOR"

# Validate configuration
REQUIRED_MACHINES=$((NUM_PROCESSORS * MACHINES_PER_PROCESSOR))
if [[ $TOTAL_MACHINES -lt $REQUIRED_MACHINES ]]; then
    echo "Error: Not enough machines in hostfile" | tee -a "$MAIN_LOG"
    echo "Required: $REQUIRED_MACHINES (${NUM_PROCESSORS} processors × ${MACHINES_PER_PROCESSOR} machines)" | tee -a "$MAIN_LOG"
    echo "Available: $TOTAL_MACHINES" | tee -a "$MAIN_LOG"
    exit 1
fi

# Create processor groups
declare -a PROCESSORS
for ((i=0; i<NUM_PROCESSORS; i++)); do
    start_idx=$((i * MACHINES_PER_PROCESSOR))
    
    # Get machines for this processor
    proc_machines=()
    for ((j=0; j<MACHINES_PER_PROCESSOR; j++)); do
        idx=$((start_idx + j))
        proc_machines+=("${ALL_MACHINES[$idx]}")
    done
    
    # Store processor info (head node is first machine)
    PROCESSORS[$i]="${proc_machines[0]}"  # Head node of this processor
    
    log "Processor $i: head=${proc_machines[0]}, machines=${proc_machines[*]}"
done

# ======================================================================
# Additional Helper Functions
# ======================================================================

# Execute command on remote machine
remote_exec() {
    local machine=$1
    local cmd=$2
    
    ssh -n "$machine" "bash -c '
        # Try multiple conda initialization methods
        if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
            source ~/anaconda3/etc/profile.d/conda.sh
        elif command -v conda &> /dev/null; then
            eval \"\$(conda shell.bash hook)\"
        else
            export PATH=\"\$HOME/anaconda3/bin:\$PATH\"
        fi
        
        # Activate environment if conda is available
        if command -v conda &> /dev/null; then
            conda activate $CONDA_ENV || true
        fi
        
        cd $PROJECT_ROOT
        $cmd
    '"
}

# ======================================================================
# Ray Management
# ======================================================================

# Initialize Ray cluster for a processor
init_ray_for_processor() {
    local processor_id=$1
    local head_node="${PROCESSORS[$processor_id]}"
    
    log "Initializing Ray cluster for processor $processor_id (head: $head_node)..."
    
    # Calculate machine indices for this processor
    local start_idx=$((processor_id * MACHINES_PER_PROCESSOR))
    
    # Build hostfile content for this processor
    local hostfile_content=""
    for ((j=0; j<MACHINES_PER_PROCESSOR; j++)); do
        local idx=$((start_idx + j))
        local machine="${ALL_MACHINES[$idx]}"
        
        # Get the full line from BACKUP hostfile (including slot information)
        # Use backup to avoid reading already-modified hostfile
        local full_line=$(awk -v machine="$machine" '$1 == machine {print; exit}' "$HOSTFILE_BACKUP")
        
        log "  Machine $j for processor $processor_id: $full_line"
        
        if [[ $j -eq 0 ]]; then
            hostfile_content="$full_line"
        else
            hostfile_content="${hostfile_content}\n${full_line}"
        fi
    done
    
    log "Hostfile content for processor $processor_id:"
    echo -e "$hostfile_content" | while read line; do
        log "  | $line"
    done
    
    # Create hostfile on head node
    log "Creating hostfile on processor $processor_id head node ($head_node)..."
    ssh -n "$head_node" "echo -e '${hostfile_content}' > /etc/mpi/hostfile" 2>&1 | tee "$LOG_DIR/ray_processor_${processor_id}.log"
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log "ERROR: Failed to create hostfile on processor $processor_id"
        return 1
    fi
    
    # Verify hostfile was created correctly
    log "Verifying hostfile on processor $processor_id..."
    ssh -n "$head_node" "cat /etc/mpi/hostfile" 2>&1 | while read line; do
        log "  Remote hostfile | $line"
    done
    
    # Initialize Ray cluster using init_ray_cluster.sh
    log "Starting Ray cluster on processor $processor_id..."
    
    # Get the project root on the remote machine
    remote_exec "$head_node" "bash ${PROJECT_ROOT}/scripts/init_ray_cluster.sh" >> "$LOG_DIR/ray_processor_${processor_id}.log" 2>&1 &
    local init_pid=$!
    
    # Wait for initialization to complete
    wait $init_pid
    local init_exit_code=$?
    
    if [[ $init_exit_code -eq 0 ]]; then
        log "Ray cluster initialized successfully on processor $processor_id"
    else
        log "ERROR: Failed to initialize Ray cluster on processor $processor_id (exit code: $init_exit_code)"
        log "ERROR: Check log file for details: $LOG_DIR/ray_processor_${processor_id}.log"
        # Show last 20 lines of error log
        tail -20 "$LOG_DIR/ray_processor_${processor_id}.log" | while read line; do
            log "  | $line"
        done
        return 1
    fi
    
    # Verify Ray status
    remote_exec "$head_node" "ray status" >> "$LOG_DIR/ray_processor_${processor_id}.log" 2>&1
    
    return 0
}

# Stop Ray cluster for a processor
stop_ray_for_processor() {
    local processor_id=$1
    local head_node="${PROCESSORS[$processor_id]}"
    
    log "Stopping Ray cluster on processor $processor_id (head: $head_node)..."
    
    # Calculate machine indices for this processor
    local start_idx=$((processor_id * MACHINES_PER_PROCESSOR))
    
    # Stop Ray on all machines in this processor (in reverse order)
    for ((j=$((MACHINES_PER_PROCESSOR-1)); j>=0; j--)); do
        local idx=$((start_idx + j))
        local machine="${ALL_MACHINES[$idx]}"
        
        log "Stopping Ray on $machine (processor $processor_id)..."
        remote_exec "$machine" "ray stop" >> "$LOG_DIR/ray_processor_${processor_id}.log" 2>&1 || true
    done
    
    log "Ray cluster stopped on processor $processor_id"
}

# ======================================================================
# Task Queue Management
# ======================================================================

TASK_QUEUE_FILE="$LOG_DIR/task_queue.txt"
TASK_QUEUE_LOCK="$LOG_DIR/task_queue.lock"

# Initialize task queue from config file
init_task_queue() {
    log "Initializing task queue from $CONFIG_FILE..."
    
    # Copy config to task queue (filter out empty lines and comment lines)
    # Format: task_name|||command
    grep -v '^[[:space:]]*$' "$CONFIG_FILE" | grep -v '^[[:space:]]*#' > "$TASK_QUEUE_FILE"
    
    local total_tasks=$(wc -l < "$TASK_QUEUE_FILE")
    log "Total tasks in queue: $total_tasks"
}

# Get next task from queue (with file lock)
get_next_task() {
    (
        flock -x 200
        
        # Check if queue is empty
        if [[ ! -s "$TASK_QUEUE_FILE" ]]; then
            echo ""
            return
        fi
        
        # Get first line
        local task=$(head -1 "$TASK_QUEUE_FILE")
        
        # Remove first line
        sed -i.bak '1d' "$TASK_QUEUE_FILE"
        rm -f "${TASK_QUEUE_FILE}.bak"
        
        echo "$task"
    ) 200>"$TASK_QUEUE_LOCK"
}

# Get remaining tasks count
get_remaining_tasks_count() {
    if [[ -f "$TASK_QUEUE_FILE" ]]; then
        wc -l < "$TASK_QUEUE_FILE" | tr -d ' '
    else
        echo "0"
    fi
}

# ======================================================================
# Processor State Management
# ======================================================================

# Mark processor as busy
mark_processor_busy() {
    local processor_id=$1
    local task_id=$2
    local task_name=$3
    echo "busy" > "$LOG_DIR/processor_${processor_id}_status.txt"
    echo "$task_id" > "$LOG_DIR/processor_${processor_id}_task_id.txt"
    echo "$task_name" > "$LOG_DIR/processor_${processor_id}_task_name.txt"
}

# Mark processor as idle
mark_processor_idle() {
    local processor_id=$1
    echo "idle" > "$LOG_DIR/processor_${processor_id}_status.txt"
    rm -f "$LOG_DIR/processor_${processor_id}_task_id.txt"
    rm -f "$LOG_DIR/processor_${processor_id}_task_name.txt"
    rm -f "$LOG_DIR/processor_${processor_id}_task.pid"
}

# Check if processor is idle
is_processor_idle() {
    local processor_id=$1
    local status_file="$LOG_DIR/processor_${processor_id}_status.txt"
    
    # If status file doesn't exist, processor is idle
    if [[ ! -f "$status_file" ]]; then
        return 0  # true
    fi
    
    local status=$(cat "$status_file")
    if [[ "$status" == "idle" ]]; then
        return 0  # true
    else
        return 1  # false
    fi
}

# Get processor current task ID
get_processor_task_id() {
    local processor_id=$1
    local task_id_file="$LOG_DIR/processor_${processor_id}_task_id.txt"
    
    if [[ -f "$task_id_file" ]]; then
        cat "$task_id_file"
    else
        echo "none"
    fi
}

# Get processor current task name
get_processor_task_name() {
    local processor_id=$1
    local task_name_file="$LOG_DIR/processor_${processor_id}_task_name.txt"
    
    if [[ -f "$task_name_file" ]]; then
        cat "$task_name_file"
    else
        echo "none"
    fi
}

# ======================================================================
# Task Execution
# ======================================================================

TASK_COUNTER=0
FAILED_TASKS=0
declare -a FAILED_TASK_LIST

# Execute task on processor
execute_task_on_processor() {
    local processor_id=$1
    local task_name=$2
    local task_cmd=$3
    local task_id=$4
    
    local head_node="${PROCESSORS[$processor_id]}"
    local task_log="$LOG_DIR/task_${task_id}_${task_name}.log"
    
    log "[Processor $processor_id] Starting task ${task_id} (${task_name})"
    log "[Processor $processor_id] Head node: $head_node"
    log "[Processor $processor_id] Command: $task_cmd"
    
    # Modify task command to use ray_address=local
    # Ensure the command uses local Ray instance
    if [[ ! "$task_cmd" =~ "--ray_address" ]]; then
        task_cmd="$task_cmd --ray_address local"
    fi
    
    # Execute on head node of the processor
    remote_exec "$head_node" "$task_cmd" > "$task_log" 2>&1 &
    local pid=$!
    
    # Save PID
    echo "$pid" > "$LOG_DIR/processor_${processor_id}_task.pid"
    
    log "[Processor $processor_id] Task ${task_id} (${task_name}) started with PID $pid"
}

# Check task completion on all processors
check_task_completion() {
    for ((i=0; i<NUM_PROCESSORS; i++)); do
        local pid_file="$LOG_DIR/processor_${i}_task.pid"
        
        # Skip if no task is running
        if [[ ! -f "$pid_file" ]]; then
            continue
        fi
        
        local pid=$(cat "$pid_file")
        local task_id=$(get_processor_task_id "$i")
        local task_name=$(get_processor_task_name "$i")
        
        # Check if process is still running
        if ! kill -0 "$pid" 2>/dev/null; then
            # Task completed - get exit status
            wait "$pid" 2>/dev/null
            local exit_code=$?

            log "[Processor $i] Task ${task_id} (${task_name}) completed (PID $pid, exit code: $exit_code)"
            mark_processor_idle "$i"

            # Check task exit status from log and exit code
            local task_log="$LOG_DIR/task_${task_id}_${task_name}.log"
            local task_failed=0

            # Check if log contains "Failed tasks: 0" (indicates success)
            if grep -q "Failed tasks: 0" "$task_log"; then
                task_failed=0
            # If log contains "Failed tasks:" with non-zero value, it's a failure
            elif grep -E "Failed tasks: [1-9][0-9]*" "$task_log"; then
                task_failed=1
            fi

            # Check exit code (non-zero usually means failure)
            if [[ $exit_code -ne 0 ]]; then
                task_failed=1
            fi

            if [[ $task_failed -eq 1 ]]; then
                log "[Processor $i] ERROR: Task ${task_id} (${task_name}) FAILED. Check log: $task_log"
                FAILED_TASKS=$((FAILED_TASKS + 1))
                FAILED_TASK_LIST+=("${task_name}")
            else
                log "[Processor $i] SUCCESS: Task ${task_id} (${task_name}) completed successfully"
            fi
        fi
    done
}

# Check if any tasks are still running
has_running_tasks() {
    for ((i=0; i<NUM_PROCESSORS; i++)); do
        local pid_file="$LOG_DIR/processor_${i}_task.pid"
        
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                return 0  # true - has running tasks
            fi
        fi
    done
    
    return 1  # false - no running tasks
}

# ======================================================================
# Main Execution
# ======================================================================

# Cleanup function
cleanup() {
    local exit_status=$?  # Capture the exit status before cleanup
    log "Cleaning up..."

    # Stop Ray on all processors
    for ((i=0; i<NUM_PROCESSORS; i++)); do
        stop_ray_for_processor "$i"
    done

    # Restore original hostfile
    if [[ -f "$HOSTFILE_BACKUP" ]]; then
        cp "$HOSTFILE_BACKUP" /etc/mpi/hostfile
        rm -f "$HOSTFILE_BACKUP"
        log "Restored original hostfile"
    fi

    log "Cleanup completed"

    # Preserve the original exit status
    exit $exit_status
}

# Register cleanup on exit
trap cleanup EXIT

# Main function
main() {
    log "=== Starting Multi-Processor Parallel Evaluation ==="
    
    # 1. Initialize Ray on all processors
    log "Step 1: Initializing Ray clusters on all processors..."
    for ((i=0; i<NUM_PROCESSORS; i++)); do
        init_ray_for_processor "$i"
        if [[ $? -ne 0 ]]; then
            log "ERROR: Failed to initialize processor $i, aborting..."
            exit 1
        fi
    done
    
    # 2. Initialize task queue
    log "Step 2: Initializing task queue..."
    init_task_queue
    
    # 3. Initialize processor states
    log "Step 3: Initializing processor states..."
    for ((i=0; i<NUM_PROCESSORS; i++)); do
        mark_processor_idle "$i"
    done
    
    # 4. Main scheduling loop
    log "Step 4: Starting task scheduling loop..."
    
    while true; do
        # Check task completion
        check_task_completion
        
        # Try to assign tasks to idle processors
        for ((i=0; i<NUM_PROCESSORS; i++)); do
            if is_processor_idle "$i"; then
                # Get next task (format: task_name|||command)
                task=$(get_next_task)
                
                if [[ -n "$task" ]]; then
                    # Parse task_name and command
                    task_name=$(echo "$task" | cut -d'|' -f1)
                    task_cmd=$(echo "$task" | cut -d'|' -f4-)
                    
                    # Assign task
                    TASK_COUNTER=$((TASK_COUNTER + 1))
                    mark_processor_busy "$i" "$TASK_COUNTER" "$task_name"
                    execute_task_on_processor "$i" "$task_name" "$task_cmd" "$TASK_COUNTER"
                fi
            fi
        done
        
        # Check if we should exit
        remaining=$(get_remaining_tasks_count)
        if [[ "$remaining" -eq 0 ]] && ! has_running_tasks; then
            log "All tasks completed!"
            break
        fi
        
        # Progress report
        running_count=0
        for ((i=0; i<NUM_PROCESSORS; i++)); do
            if ! is_processor_idle "$i"; then
                running_count=$((running_count + 1))
            fi
        done
        
        log "Progress: $running_count running, $remaining queued"
        
        # Wait before next iteration
        sleep 30
    done
    
    log "=== Multi-Processor Parallel Evaluation Completed ==="

    # Report final statistics
    local total_tasks=$TASK_COUNTER
    local successful_tasks=$((total_tasks - FAILED_TASKS))

    log "========== Task Execution Summary =========="
    log "Total tasks: $total_tasks"
    log "Successful tasks: $successful_tasks"
    log "Failed tasks: $FAILED_TASKS"

    if [[ $FAILED_TASKS -gt 0 ]]; then
        log "Failed task list:"
        for failed_task in "${FAILED_TASK_LIST[@]}"; do
            log "  - $failed_task"
        done
    fi

    log "============================================"
    log "Logs are saved in: $LOG_DIR"
    log "Check individual task logs: $LOG_DIR/task_*.log"

    # Set exit code based on failed tasks
    if [[ $FAILED_TASKS -gt 0 ]]; then
        log "ERROR: Some tasks failed. Exiting with error code 1."
        exit 1
    else
        log "SUCCESS: All tasks completed successfully."
        exit 0
    fi
}

# Run main function
main
