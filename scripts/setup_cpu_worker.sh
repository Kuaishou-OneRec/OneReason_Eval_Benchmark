#!/bin/bash
# Setup CPU-only sentinel worker for API-based evaluation tasks.
#
# Usage: bash scripts/setup_cpu_worker.sh
#   - Prompts for queue ID (used as WORKER_ID for processing directory)
#   - Starts sentinel_worker.py --cpu in a tmux session
#
# The CPU worker:
#   - Polls incoming_cpu/ for API tasks (closed-source model evaluation)
#   - Executes tasks concurrently (default 5 parallel)
#   - Does NOT need GPU

set -e

TMUX_SESSION="sent_cpu"

# Get script and parent directory
script_dir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname ${script_dir})

echo "============================================"
echo "CPU Worker Setup"
echo "============================================"
echo ""

# Get WORKER_ID
read -p "Enter WORKER_ID for this CPU worker (e.g. 50000): " WORKER_ID
WORKER_ID=${WORKER_ID:-50000}

# Get max workers
read -p "Enter max concurrent tasks (default: 5): " MAX_WORKERS
MAX_WORKERS=${MAX_WORKERS:-5}

echo ""
echo "Configuration:"
echo "  - WORKER_ID: ${WORKER_ID}"
echo "  - Max concurrent tasks: ${MAX_WORKERS}"
echo "  - Tmux session: ${TMUX_SESSION}"
echo ""

# Check if tmux session already exists
session_exists=$(tmux ls 2>/dev/null | grep -q "^${TMUX_SESSION}:" && echo 'yes' || echo 'no')

if [[ "${session_exists}" == "yes" ]]; then
    echo "⊗ Tmux session '${TMUX_SESSION}' already exists. Kill it first:"
    echo "  tmux kill-session -t ${TMUX_SESSION}"
    exit 1
fi

# Create tmux session and start worker
tmux new-session -d -s ${TMUX_SESSION}
tmux send-keys -t ${TMUX_SESSION} "export CPU_WORKER_ID=${WORKER_ID}" C-m
tmux send-keys -t ${TMUX_SESSION} "cd ${parent_dir}/auto_eval" C-m
tmux send-keys -t ${TMUX_SESSION} "python3 executor/sentinel_worker.py --cpu --cpu-max-workers ${MAX_WORKERS}" C-m

echo "✓ CPU worker started!"
echo ""
echo "To attach: tmux attach -t ${TMUX_SESSION}"
echo "To stop:   tmux kill-session -t ${TMUX_SESSION}"
