#!/bin/bash

# This script should be run on the first machine in hostfile
# It will:
# 1. Backup the original hostfile
# 2. Group machines in pairs and create new hostfiles on each odd-row machine
# 3. Start tmux sessions on odd-row machines and run sentinel workers
#
# Usage: bash setup_hostfile_and_workers.sh [BASE_WORKER_ID] [MACHINES_PER_WORKER] [PARALLEL_MODE]
#   BASE_WORKER_ID: Base value for WORKER_ID (default: 20000)
#                   First worker will be BASE_WORKER_ID+0, second BASE_WORKER_ID+1, etc.
#   MACHINES_PER_WORKER: Number of machines in each group (default: 2)
#   PARALLEL_MODE: Value for PARALLEL_MODE environment variable (default: true)
#
# Examples:
#   bash setup_hostfile_and_workers.sh                     # Uses defaults: base=20000, group_size=2, test=true
#   bash setup_hostfile_and_workers.sh 30000               # Uses base=30000, group_size=2, test=true
#   bash setup_hostfile_and_workers.sh 30000 3             # Uses base=30000, group_size=3, test=true
#   bash setup_hostfile_and_workers.sh 30000 3 false       # Uses base=30000, group_size=3, test=false
#   bash setup_hostfile_and_workers.sh 20000 4 true        # Uses base=20000, group_size=4, test=true

set -e


# Configuration
HOSTFILE_PATH="/etc/mpi/hostfile"
BACKUP_PATH="/etc/mpi/hostfile_backup_ori"
TMUX_SESSION="sent"


# Get script and parent directory
script_dir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname ${script_dir})

# Setup logging
log_dir="${parent_dir}/worker_setup_logs"
mkdir -p "${log_dir}"

# Read first node from hostfile
FIRST_NODE=$(awk 'NR==1 {print $1}' ${HOSTFILE_PATH})

read -p "Setup or rollback? (setup/rollback): " SETUP_OR_ROLLBACK
SETUP_OR_ROLLBACK=${SETUP_OR_ROLLBACK:-setup}

if [ "$SETUP_OR_ROLLBACK" == "setup" ]; then

    # Interactive parameter input
    echo "============================================"
    echo "Setup Configuration"
    echo "============================================"
    echo ""

    # Get BASE_WORKER_ID
    read -p "Enter BASE_WORKER_ID (default: 20000): " BASE_WORKER_ID
    BASE_WORKER_ID=${BASE_WORKER_ID:-20000}

    # Get MACHINES_PER_WORKER
    read -p "Enter MACHINES_PER_WORKER (default: 2): " MACHINES_PER_WORKER
    MACHINES_PER_WORKER=${MACHINES_PER_WORKER:-2}

    # Get PARALLEL_MODE
    read -p "Enter PARALLEL_MODE - true/false (default: true): " PARALLEL_MODE
    PARALLEL_MODE=${PARALLEL_MODE:-true}

    if [ "$PARALLEL_MODE" == "true" ]; then
        read -p "Enter NUM_PROCESSORS (default: 2): " NUM_PROCESSORS
        NUM_PROCESSORS=${NUM_PROCESSORS:-2}

        read -p "Enter MACHINES_PER_PROCESSOR (default: 1): " MACHINES_PER_PROCESSOR
        MACHINES_PER_PROCESSOR=${MACHINES_PER_PROCESSOR:-1}
    fi

    echo ""


    # Replace dots in IP address with underscores for filename
    log_file="${log_dir}/setup_${BASE_WORKER_ID}.log"

    # Create worker mapping file
    worker_mapping_file="${parent_dir}/worker_mapping.txt"

    # Redirect all output to both console and log file
    exec > >(tee -a "${log_file}")
    exec 2>&1

    echo "============================================"
    echo "Starting hostfile setup and worker deployment"
    echo "============================================"
    echo "Configuration:"
    echo "  - Base WORKER_ID: ${BASE_WORKER_ID}"
    echo "  - Machines per worker: ${MACHINES_PER_WORKER}"
    echo "  - Parallel mode: ${PARALLEL_MODE}"
    if [ "$PARALLEL_MODE" == "true" ]; then
        echo "  - Num processors: ${NUM_PROCESSORS}"
        echo "  - Machines per processor: ${MACHINES_PER_PROCESSOR}"
    fi
    echo "  - Hostfile path: ${HOSTFILE_PATH}"
    echo "  - Tmux session: ${TMUX_SESSION}"
    echo "  - Log file: ${log_file}"
    echo ""

    # Step 1: Backup hostfile on first node (local operation, no ssh needed)
    echo "[1/4] Backing up hostfile..."
    cp ${HOSTFILE_PATH} ${BACKUP_PATH} 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Backup created: ${BACKUP_PATH}"
    else
        echo "✗ Failed to backup hostfile"
        exit 1
    fi
    echo ""

    # Step 2: Read all lines from hostfile (local operation, no ssh needed)
    echo "[2/4] Reading hostfile..."
    # Read hostfile into array, preserving full lines
    mapfile -t ALL_LINES < ${HOSTFILE_PATH}
    total_lines=${#ALL_LINES[@]}
    echo "✓ Found ${total_lines} machines in hostfile"
    echo ""

    # Step 3: Calculate number of groups
    num_groups=$((total_lines / MACHINES_PER_WORKER))
    echo "[3/4] Creating ${num_groups} groups (${MACHINES_PER_WORKER} machines each)..."
    remainder=$((total_lines % MACHINES_PER_WORKER))
    if [ ${remainder} -ne 0 ]; then
        echo "Note: Total lines (${total_lines}) is not divisible by ${MACHINES_PER_WORKER}, last ${remainder} machine(s) will be skipped"
    fi
    echo ""

    # Step 4: Process each group
    echo "[4/4] Setting up hostfiles and starting workers..."
    echo ""

    for ((i=0; i<num_groups; i++)); do
        # Get all lines for this group
        start_idx=$((i * MACHINES_PER_WORKER))

        # Build the hostfile content for this group
        new_hostfile=""
        for ((j=0; j<MACHINES_PER_WORKER; j++)); do
            idx=$((start_idx + j))
            if [ $j -eq 0 ]; then
                new_hostfile="${ALL_LINES[$idx]}"
            else
                new_hostfile="${new_hostfile}\n${ALL_LINES[$idx]}"
            fi
        done

        # Extract IP address from first line of the group
        first_node=$(echo "${ALL_LINES[$start_idx]}" | awk '{print $1}')

        echo "--- Group $((i+1))/${num_groups}: ${first_node} ---"

        # Step 4.1: Create new hostfile on first node of the group
        echo "  [→] Creating hostfile on ${first_node}..."
        ssh -n ${first_node} "echo -e '${new_hostfile}' > ${HOSTFILE_PATH}" 2>&1
        if [ $? -ne 0 ]; then
            echo "  [✗] Failed to create hostfile on ${first_node}"
        fi
        echo "  [✓] Hostfile created"

        # Step 4.2: Check if tmux session already exists
        echo "  [→] Checking tmux session '${TMUX_SESSION}'..."
        # Use tmux ls and grep to check for specific session
        # If tmux server is not running, tmux ls will fail and we'll get empty result
        session_exists=$(ssh -n ${first_node} "tmux ls 2>/dev/null | grep -q '^${TMUX_SESSION}:' && echo 'yes' || echo 'no'")

        if [[ "${session_exists}" == "yes" ]]; then
            echo "  [⊗] Skipping: tmux session '${TMUX_SESSION}' already exists on ${first_node}"
            echo ""
            continue
        fi
        echo "  [✓] No existing session found"

        # Step 4.3: Calculate WORKER_ID (starting from 0)
        worker_id=$((BASE_WORKER_ID + i))
        echo "  [→] Starting worker with WORKER_ID=${worker_id}..."

        # Step 4.4: Create tmux session and run commands
        # Create detached tmux session
        ssh -n ${first_node} "tmux new-session -d -s ${TMUX_SESSION}" 2>&1

        # Send commands to the tmux session
        ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'export WORKER_ID=${worker_id}' C-m" 2>&1
        ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'export PARALLEL_MODE=${PARALLEL_MODE}' C-m" 2>&1

        # If PARALLEL_MODE is false, initialize Ray cluster
        if [[ "${PARALLEL_MODE}" == "false" ]]; then
            ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'bash scripts/init_ray_cluster.sh' C-m" 2>&1
        else
            ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'export NUM_PROCESSORS=${NUM_PROCESSORS}' C-m" 2>&1
            ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'export MACHINES_PER_PROCESSOR=${MACHINES_PER_PROCESSOR}' C-m" 2>&1
        fi

        ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'cd ${parent_dir}/auto_eval' C-m" 2>&1
        ssh -n ${first_node} "tmux send-keys -t ${TMUX_SESSION} 'python3 executor/sentinel_worker.py' C-m" 2>&1


        if [ $? -eq 0 ]; then
            echo "  [✓] Worker started successfully (WORKER_ID=${worker_id})"
            # Save worker mapping
            echo "${worker_id} ${first_node}" >> "${worker_mapping_file}"
        else
            echo "  [✗] Failed to start worker"
        fi

        echo ""
    done

    echo "============================================"
    echo "Setup complete!"
    echo "============================================"
    echo ""
    echo "Summary:"
    echo "  - Processed ${num_groups} groups"
    echo "  - Hostfile backup: ${BACKUP_PATH} on ${FIRST_NODE}"
    echo "  - Worker mapping saved to: ${worker_mapping_file}"
    echo "  - Log file saved to: ${log_file}"
    echo ""
    echo "To connect to a worker quickly:"
    echo "  bash ${script_dir}/connect_worker.sh"
    echo ""
    echo "To check worker status on a machine, run:"
    echo "  ssh <machine_ip> \"tmux attach -t ${TMUX_SESSION}\""
    echo ""
    echo "To view worker logs, use tmux commands:"
    echo "  ssh <machine_ip> \"tmux capture-pane -t ${TMUX_SESSION} -p\""
    echo ""
    echo "To view setup log:"
    echo "  cat ${log_file}"

else
    echo "============================================"
    echo "Starting rollback process"
    echo "============================================"
    echo ""

    # Check if backup exists
    if [ ! -f ${BACKUP_PATH} ]; then
        echo "✗ Backup file not found: ${BACKUP_PATH}"
        echo "Nothing to rollback."
        exit 1
    fi

    # Setup logging
    log_file="${log_dir}/rollback_$(date +%Y%m%d_%H%M%S).log"

    # Worker mapping file
    worker_mapping_file="${parent_dir}/worker_mapping.txt"

    # Redirect all output to both console and log file
    exec > >(tee -a "${log_file}")
    exec 2>&1

    echo "Configuration:"
    echo "  - Backup path: ${BACKUP_PATH}"
    echo "  - First node: ${FIRST_NODE}"
    echo "  - Tmux session to kill: ${TMUX_SESSION}"
    echo "  - Worker mapping file: ${worker_mapping_file}"
    echo "  - Log file: ${log_file}"
    echo ""

    # Step 1: Restore hostfile on first node
    echo "[1/3] Restoring hostfile on first node (${FIRST_NODE})..."
    cp ${BACKUP_PATH} ${HOSTFILE_PATH} 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Hostfile restored from backup"
    else
        echo "✗ Failed to restore hostfile"
        exit 1
    fi
    echo ""

    # Step 2: Read all machines from backup hostfile
    echo "[2/3] Reading backup hostfile to get list of all machines..."
    mapfile -t ALL_LINES < ${BACKUP_PATH}
    total_lines=${#ALL_LINES[@]}
    echo "✓ Found ${total_lines} machines in backup hostfile"
    echo ""

    # Step 3: Kill tmux sessions on all machines
    echo "[3/3] Killing tmux sessions on all machines..."
    echo ""

    killed_count=0
    not_found_count=0
    failed_count=0
    mapping_removed_count=0

    for ((i=0; i<total_lines; i++)); do
        # Extract IP address from line
        node=$(echo "${ALL_LINES[$i]}" | awk '{print $1}')

        echo "  [→] Checking ${node}..."

        # Check if this is the local machine
        is_local="no"
        if [[ "${node}" == "${FIRST_NODE}" ]] || [[ "${node}" == "localhost" ]] || [[ "${node}" == "127.0.0.1" ]]; then
            is_local="yes"
        fi

        # Check if tmux session exists
        if [[ "${is_local}" == "yes" ]]; then
            # Run locally without SSH
            session_exists=$(tmux ls 2>/dev/null | grep -q "^${TMUX_SESSION}:" && echo 'yes' || echo 'no')
        else
            # Run via SSH
            session_exists=$(ssh -n ${node} "tmux ls 2>/dev/null | grep -q '^${TMUX_SESSION}:' && echo 'yes' || echo 'no'" )
        fi

        if [[ "${session_exists}" == "yes" ]]; then
            # Send Ctrl-C to the tmux session first to interrupt running processes
            echo "  [→] Sending interrupt signal to tmux session..."
            if [[ "${is_local}" == "yes" ]]; then
                tmux send-keys -t ${TMUX_SESSION} C-c 2>&1 || true
            else
                ssh -n ${node} "tmux send-keys -t ${TMUX_SESSION} C-c" 2>&1 || true
            fi
            echo "  [✓] Interrupt signal sent"

            # Give processes a moment to handle the interrupt
            sleep 1

            # Stop Ray first (ignore errors if Ray is not running)
            echo "  [→] Stopping Ray on ${node}..."
            if [[ "${is_local}" == "yes" ]]; then
                ray stop 2>&1 || true
            else
                ssh -n ${node} "ray stop" 2>&1 || true
            fi
            echo "  [✓] Ray stop executed"

            # Kill the tmux session
            echo "  [→] Killing tmux session..."
            if [[ "${is_local}" == "yes" ]]; then
                tmux kill-session -t ${TMUX_SESSION} 2>&1
                kill_result=$?
            else
                ssh -n ${node} "tmux kill-session -t ${TMUX_SESSION}" 2>&1
                kill_result=$?
            fi

            if [ ${kill_result} -eq 0 ]; then
                echo "  [✓] Killed tmux session '${TMUX_SESSION}'"
                killed_count=$((killed_count + 1))

                # Remove worker mapping entry for this machine
                if [ -f "${worker_mapping_file}" ]; then
                    echo "  [→] Removing mapping entries for ${node}..."
                    # Create temp file and remove lines containing this node's IP
                    if grep -v " ${node}$" "${worker_mapping_file}" > "${worker_mapping_file}.tmp" 2>/dev/null; then
                        # Calculate removed lines safely
                        old_count=$(wc -l < "${worker_mapping_file}" 2>/dev/null || echo "0")
                        new_count=$(wc -l < "${worker_mapping_file}.tmp" 2>/dev/null || echo "0")
                        removed_lines=$((old_count - new_count))

                        mv "${worker_mapping_file}.tmp" "${worker_mapping_file}" 2>/dev/null || true

                        if [ ${removed_lines} -gt 0 ]; then
                            echo "  [✓] Removed ${removed_lines} mapping entry(ies)"
                            mapping_removed_count=$((mapping_removed_count + removed_lines))
                        else
                            echo "  [○] No mapping entry found for ${node}"
                        fi
                    else
                        echo "  [○] Failed to process mapping file"
                        rm -f "${worker_mapping_file}.tmp" 2>/dev/null || true
                    fi
                fi
            else
                echo "  [✗] Failed to kill tmux session"
                failed_count=$((failed_count + 1))
            fi
        else
            echo "  [○] No tmux session '${TMUX_SESSION}' found"
            not_found_count=$((not_found_count + 1))
        fi
        echo ""
    done

    echo "============================================"
    echo "Rollback complete!"
    echo "============================================"
    echo ""
    echo "Summary:"
    echo "  - Total machines: ${total_lines}"
    echo "  - Tmux sessions killed: ${killed_count}"
    echo "  - Tmux sessions not found: ${not_found_count}"
    echo "  - Failed operations: ${failed_count}"
    echo "  - Worker mapping entries removed: ${mapping_removed_count}"
    echo "  - Hostfile restored on: ${FIRST_NODE}"
    echo "  - Log file saved to: ${log_file}"
    echo ""

fi