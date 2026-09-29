#!/bin/bash

# Quick connect script to attach to a worker's tmux session
# Usage: bash connect_worker.sh [WORKER_ID]

set -e

# Get script and parent directory
script_dir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname ${script_dir})

# Configuration
MAPPING_FILE="${parent_dir}/worker_mapping.txt"
TMUX_SESSION="sent"

# Check if mapping file exists
if [ ! -f "${MAPPING_FILE}" ]; then
    echo "Error: Worker mapping file not found: ${MAPPING_FILE}"
    echo ""
    echo "Please run setup_hostfile_and_workers.sh first to generate the mapping file."
    exit 1
fi

# Check if mapping file has content (more than just comments)
worker_count=$(grep -v '^#' "${MAPPING_FILE}" | grep -v '^$' | wc -l)
if [ ${worker_count} -eq 0 ]; then
    echo "Error: No workers found in mapping file: ${MAPPING_FILE}"
    echo ""
    echo "The mapping file exists but contains no worker entries."
    echo "Please check if setup_hostfile_and_workers.sh completed successfully."
    exit 1
fi

# Display available workers
echo "============================================"
echo "Available Workers"
echo "============================================"
echo ""
grep -v '^#' "${MAPPING_FILE}" | grep -v '^$' | while read worker_id machine_ip; do
    echo "  WORKER_ID: ${worker_id}  =>  Machine: ${machine_ip}"
done
echo ""
echo "============================================"
echo ""

# Get WORKER_ID from command line argument or user input
if [ -n "$1" ]; then
    TARGET_WORKER_ID="$1"
    echo "Connecting to WORKER_ID: ${TARGET_WORKER_ID}"
else
    read -p "Enter WORKER_ID to connect: " TARGET_WORKER_ID
fi

# Validate input
if [ -z "${TARGET_WORKER_ID}" ]; then
    echo "Error: WORKER_ID cannot be empty"
    exit 1
fi

# Find the machine IP for the given WORKER_ID
MACHINE_IP=$(grep -v '^#' "${MAPPING_FILE}" | grep -v '^$' | grep "^${TARGET_WORKER_ID} " | awk '{print $2}')

if [ -z "${MACHINE_IP}" ]; then
    echo "Error: WORKER_ID ${TARGET_WORKER_ID} not found in mapping file"
    echo ""
    echo "Available WORKER_IDs:"
    grep -v '^#' "${MAPPING_FILE}" | grep -v '^$' | awk '{print "  - " $1}'
    exit 1
fi

# Check if tmux session exists on the target machine
echo ""
echo "Checking tmux session '${TMUX_SESSION}' on ${MACHINE_IP}..."
session_exists=$(ssh -n ${MACHINE_IP} "tmux ls 2>/dev/null | grep -q '^${TMUX_SESSION}:' && echo 'yes' || echo 'no'")

if [[ "${session_exists}" != "yes" ]]; then
    echo "Warning: tmux session '${TMUX_SESSION}' not found on ${MACHINE_IP}"
    echo ""
    read -p "Do you still want to connect? (y/n): " continue_connect
    if [[ "${continue_connect}" != "y" && "${continue_connect}" != "Y" ]]; then
        echo "Connection cancelled."
        exit 0
    fi
    echo ""
    echo "Connecting to ${MACHINE_IP} (without tmux attach)..."
    ssh -t ${MACHINE_IP}
else
    echo "Found tmux session '${TMUX_SESSION}'"
    echo ""
    echo "Connecting to WORKER_ID ${TARGET_WORKER_ID} on ${MACHINE_IP}..."
    echo "Press Ctrl+B then D to detach from tmux session"
    echo ""
    sleep 1

    # Connect and attach to tmux session
    ssh -t ${MACHINE_IP} "tmux attach -t ${TMUX_SESSION}"
fi
