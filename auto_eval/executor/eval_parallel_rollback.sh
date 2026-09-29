#!/bin/bash

# ======================================================================
# Multi-Processor Parallel Evaluation Rollback Script
# ======================================================================
# Purpose: Clean up resources and restore state after eval_parallel.sh
#          execution (normal completion, interruption, or failure)
#
# This script will:
# - Stop Ray clusters on all machines
# - Restore original hostfile from backup
# - Terminate any running task processes
# - Clean up state files (while preserving logs)
#
# Usage:
#   bash eval_parallel_rollback.sh [--log_dir <dir>]
# ======================================================================

set -e  # Exit on error

# ======================================================================
# Configuration
# ======================================================================

HOSTFILE_PATH="/etc/mpi/hostfile"
HOSTFILE_BACKUP="/etc/mpi/hostfile_backup"
LOG_DIR=""
PROJECT_ROOT=$(cd $(dirname $0)/../..; pwd)
CONDA_ENV="verl"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--log_dir <dir>] [--conda_env <env>]"
            echo ""
            echo "Options:"
            echo "  --log_dir <dir>     Path to the log directory (optional, will auto-detect if not provided)"
            echo "  --conda_env <env>   Conda environment name (default: verl)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# ======================================================================
# Auto-detect log directory if not provided
# ======================================================================

if [[ -z "$LOG_DIR" ]]; then
    echo "Log directory not specified, searching for recent parallel execution logs..."

    # Search for parallel log directories
    SEARCH_PATHS=(
        "${PROJECT_ROOT}/auto_eval_logs/*/parallel"
        "${BENCHMARK_LOG_DIR:-./workspace}/auto_eval_logs/*/parallel"
    )

    FOUND_DIRS=()
    for pattern in "${SEARCH_PATHS[@]}"; do
        for dir in $pattern; do
            if [[ -d "$dir" ]] && [[ -f "$dir/main.log" ]]; then
                FOUND_DIRS+=("$dir")
            fi
        done
    done

    if [[ ${#FOUND_DIRS[@]} -eq 0 ]]; then
        echo "No log directories found. You can specify one with --log_dir"
        echo "Proceeding with basic cleanup (Ray stop and hostfile restore only)..."
        LOG_DIR=""
    elif [[ ${#FOUND_DIRS[@]} -eq 1 ]]; then
        LOG_DIR="${FOUND_DIRS[0]}"
        echo "Found log directory: $LOG_DIR"
    else
        echo "Multiple log directories found:"
        for i in "${!FOUND_DIRS[@]}"; do
            echo "  [$i] ${FOUND_DIRS[$i]}"
        done
        echo ""
        read -p "Select log directory (0-$((${#FOUND_DIRS[@]}-1))): " selection
        LOG_DIR="${FOUND_DIRS[$selection]}"
        echo "Using log directory: $LOG_DIR"
    fi
fi

# ======================================================================
# Setup logging
# ======================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [[ -n "$LOG_DIR" ]]; then
    ROLLBACK_LOG="$LOG_DIR/rollback_${TIMESTAMP}.log"
else
    # Use temporary log location if LOG_DIR not found
    ROLLBACK_LOG="/tmp/eval_parallel_rollback_${TIMESTAMP}.log"
fi

# Log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$ROLLBACK_LOG"
}

echo "=== Multi-Processor Parallel Evaluation Rollback Started ===" | tee "$ROLLBACK_LOG"
log "Rollback log: $ROLLBACK_LOG"
log ""

# ======================================================================
# Helper Functions
# ======================================================================

# Get local machine IP/hostname
get_local_machine() {
    # Try to get the first machine in hostfile as reference
    if [[ -f "$HOSTFILE_PATH" ]]; then
        awk 'NR==1 {print $1}' "$HOSTFILE_PATH"
    elif [[ -f "$HOSTFILE_BACKUP" ]]; then
        awk 'NR==1 {print $1}' "$HOSTFILE_BACKUP"
    else
        echo "localhost"
    fi
}

LOCAL_MACHINE=$(get_local_machine)

# Check if a machine is local
is_local_machine() {
    local machine=$1
    [[ "$machine" == "$LOCAL_MACHINE" ]] || [[ "$machine" == "localhost" ]] || [[ "$machine" == "127.0.0.1" ]]
}

# Execute command on remote machine or locally
remote_exec() {
    local machine=$1
    local cmd=$2

    # Check if this is the local machine
    if is_local_machine "$machine"; then
        # Execute locally
        bash -c "
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
        " 2>&1
    else
        # Execute via SSH
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
        '" 2>&1
    fi
}

# ======================================================================
# Step 1: Check if backup hostfile exists
# ======================================================================

log "============================================"
log "Step 1: Checking backup hostfile"
log "============================================"

if [[ ! -f "$HOSTFILE_BACKUP" ]]; then
    log "WARNING: Backup hostfile not found: $HOSTFILE_BACKUP"
    log "This may mean:"
    log "  - eval_parallel.sh was never run"
    log "  - Backup was already restored"
    log "  - Backup was manually deleted"
    log ""

    # Try to get machine list from current hostfile instead
    if [[ -f "$HOSTFILE_PATH" ]]; then
        log "Using current hostfile instead: $HOSTFILE_PATH"
        HOSTFILE_TO_USE="$HOSTFILE_PATH"
        RESTORE_NEEDED=false
    else
        log "ERROR: No hostfile found at all. Cannot proceed."
        exit 1
    fi
else
    log "Backup hostfile found: $HOSTFILE_BACKUP"
    HOSTFILE_TO_USE="$HOSTFILE_BACKUP"
    RESTORE_NEEDED=true
fi

log ""

# ======================================================================
# Step 2: Read machine list from hostfile
# ======================================================================

log "============================================"
log "Step 2: Reading machine list"
log "============================================"

mapfile -t ALL_MACHINES < <(awk '{print $1}' "$HOSTFILE_TO_USE")
TOTAL_MACHINES=${#ALL_MACHINES[@]}

log "Found $TOTAL_MACHINES machines:"
for machine in "${ALL_MACHINES[@]}"; do
    log "  - $machine"
done
log ""

# ======================================================================
# Step 3: Stop Ray on all machines
# ======================================================================

log "============================================"
log "Step 3: Stopping Ray on all machines"
log "============================================"

ray_stopped_count=0
ray_not_running_count=0
ray_failed_count=0

for machine in "${ALL_MACHINES[@]}"; do
    log "Stopping Ray on $machine..."

    # Try to stop Ray (ignore errors if Ray is not running)
    if remote_exec "$machine" "ray stop" &> /dev/null; then
        log "  [✓] Ray stopped successfully on $machine"
        ray_stopped_count=$((ray_stopped_count + 1))
    else
        # Check if Ray was actually running
        if remote_exec "$machine" "ray status" &> /dev/null; then
            log "  [✗] Failed to stop Ray on $machine"
            ray_failed_count=$((ray_failed_count + 1))
        else
            log "  [○] Ray was not running on $machine"
            ray_not_running_count=$((ray_not_running_count + 1))
        fi
    fi
done

log ""
log "Ray cleanup summary:"
log "  - Successfully stopped: $ray_stopped_count"
log "  - Not running: $ray_not_running_count"
log "  - Failed to stop: $ray_failed_count"
log ""

# ======================================================================
# Step 4: Restore original hostfile
# ======================================================================

log "============================================"
log "Step 4: Restoring original hostfile"
log "============================================"

if [[ "$RESTORE_NEEDED" == true ]]; then
    log "Restoring hostfile from backup..."

    if cp "$HOSTFILE_BACKUP" "$HOSTFILE_PATH"; then
        log "  [✓] Hostfile restored: $HOSTFILE_PATH"

        # Remove backup after successful restore
        if rm -f "$HOSTFILE_BACKUP"; then
            log "  [✓] Backup file removed: $HOSTFILE_BACKUP"
        else
            log "  [!] Warning: Could not remove backup file"
        fi
    else
        log "  [✗] Failed to restore hostfile"
        exit 1
    fi
else
    log "Hostfile restore not needed (backup doesn't exist)"
fi

log ""

# ======================================================================
# Step 5: Terminate running task processes
# ======================================================================

log "============================================"
log "Step 5: Terminating running task processes"
log "============================================"

tasks_killed=0
tasks_not_running=0

if [[ -n "$LOG_DIR" ]] && [[ -d "$LOG_DIR" ]]; then
    # Find all PID files
    for pid_file in "$LOG_DIR"/processor_*_task.pid; do
        if [[ -f "$pid_file" ]]; then
            pid=$(cat "$pid_file")
            processor_id=$(basename "$pid_file" | sed 's/processor_\(.*\)_task.pid/\1/')

            log "Checking task process for processor $processor_id (PID: $pid)..."

            if kill -0 "$pid" 2>/dev/null; then
                # Process is running, kill it
                if kill "$pid" 2>/dev/null; then
                    log "  [✓] Killed task process (PID: $pid)"
                    tasks_killed=$((tasks_killed + 1))
                else
                    log "  [!] Warning: Could not kill process (PID: $pid)"
                fi
            else
                log "  [○] Task process not running (PID: $pid)"
                tasks_not_running=$((tasks_not_running + 1))
            fi
        fi
    done

    log ""
    log "Task termination summary:"
    log "  - Killed: $tasks_killed"
    log "  - Not running: $tasks_not_running"
else
    log "No log directory available, skipping task termination"
fi

log ""

# ======================================================================
# Step 6: Clean up state files
# ======================================================================

log "============================================"
log "Step 6: Cleaning up state files"
log "============================================"

state_files_removed=0

if [[ -n "$LOG_DIR" ]] && [[ -d "$LOG_DIR" ]]; then
    # Remove processor state files
    for state_file in "$LOG_DIR"/processor_*_status.txt \
                      "$LOG_DIR"/processor_*_task_id.txt \
                      "$LOG_DIR"/processor_*_task_name.txt \
                      "$LOG_DIR"/processor_*_task.pid; do
        if [[ -f "$state_file" ]]; then
            if rm -f "$state_file"; then
                log "  [✓] Removed: $(basename "$state_file")"
                state_files_removed=$((state_files_removed + 1))
            fi
        fi
    done

    # Remove task queue files
    for queue_file in "$LOG_DIR"/task_queue.txt \
                      "$LOG_DIR"/task_queue.lock; do
        if [[ -f "$queue_file" ]]; then
            if rm -f "$queue_file"; then
                log "  [✓] Removed: $(basename "$queue_file")"
                state_files_removed=$((state_files_removed + 1))
            fi
        fi
    done

    log ""
    log "State files removed: $state_files_removed"

    # Count preserved log files
    log_files_count=$(find "$LOG_DIR" -name "*.log" -type f | wc -l | tr -d ' ')
    log "Log files preserved: $log_files_count"
else
    log "No log directory available, skipping state file cleanup"
fi

log ""

# ======================================================================
# Final Summary
# ======================================================================

log "============================================"
log "Rollback Complete!"
log "============================================"
log ""
log "Summary:"
log "  - Total machines processed: $TOTAL_MACHINES"
log "  - Ray instances stopped: $ray_stopped_count"
log "  - Ray instances not running: $ray_not_running_count"
log "  - Ray stop failures: $ray_failed_count"
log "  - Task processes killed: $tasks_killed"
log "  - Task processes not running: $tasks_not_running"
log "  - State files removed: $state_files_removed"
log "  - Hostfile restored: $([ "$RESTORE_NEEDED" == true ] && echo "Yes" || echo "Not needed")"
if [[ -n "$LOG_DIR" ]]; then
    log "  - Log directory: $LOG_DIR (preserved)"
fi
log "  - Rollback log: $ROLLBACK_LOG"
log ""

if [[ $ray_failed_count -gt 0 ]]; then
    log "WARNING: Some Ray instances failed to stop. You may need to manually check:"
    for machine in "${ALL_MACHINES[@]}"; do
        if remote_exec "$machine" "ray status" &> /dev/null; then
            log "  - $machine: Ray may still be running"
        fi
    done
    log ""
fi

log "=== Rollback completed successfully ==="
