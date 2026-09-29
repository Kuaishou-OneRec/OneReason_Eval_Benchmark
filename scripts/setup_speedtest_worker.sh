#!/bin/bash

# Setup and manage speed test sentinel worker
#
# Usage:
#   bash setup_speedtest_worker.sh          # Interactive setup/rollback
#
# The sentinel worker polls the speedtest queue and executes speed benchmarks
# using compare_think_nothink.sh. Completely independent from eval sentinel.

set -e

TMUX_SESSION="speedtest_sent"

# Get script and parent directory
script_dir=$(cd $(dirname $0); pwd)
parent_dir=$(dirname ${script_dir})

# Setup logging
log_dir="${parent_dir}/worker_setup_logs"
mkdir -p "${log_dir}"

read -p "Setup or rollback? (setup/rollback): " SETUP_OR_ROLLBACK
SETUP_OR_ROLLBACK=${SETUP_OR_ROLLBACK:-setup}

if [ "$SETUP_OR_ROLLBACK" == "setup" ]; then

    echo "============================================"
    echo "Speed Test Sentinel Setup"
    echo "============================================"
    echo ""

    # Check if tmux session already exists
    session_exists=$(tmux ls 2>/dev/null | grep -q "^${TMUX_SESSION}:" && echo 'yes' || echo 'no')
    if [[ "${session_exists}" == "yes" ]]; then
        echo "[⊗] Tmux session '${TMUX_SESSION}' already exists."
        echo "    To attach: tmux attach -t ${TMUX_SESSION}"
        echo "    To rollback first: bash $0 (select rollback)"
        exit 0
    fi

    log_file="${log_dir}/speedtest_setup_$(date +%Y%m%d_%H%M%S).log"

    # Redirect output to both console and log file
    exec > >(tee -a "${log_file}")
    exec 2>&1

    echo "[1/2] Creating tmux session '${TMUX_SESSION}'..."
    tmux new-session -d -s ${TMUX_SESSION}
    echo "[✓] Tmux session created"

    echo "[2/2] Starting speedtest sentinel worker..."
    tmux send-keys -t ${TMUX_SESSION} "cd ${parent_dir}/auto_eval" C-m
    tmux send-keys -t ${TMUX_SESSION} "python3 -m speedtest.sentinel_speedtest" C-m

    echo "[✓] Sentinel worker started"
    echo ""
    echo "============================================"
    echo "Setup complete!"
    echo "============================================"
    echo ""
    echo "To attach to the worker session:"
    echo "  tmux attach -t ${TMUX_SESSION}"
    echo ""
    echo "To check worker output:"
    echo "  tmux capture-pane -t ${TMUX_SESSION} -p"
    echo ""
    echo "Log file: ${log_file}"

else
    echo "============================================"
    echo "Speed Test Sentinel Rollback"
    echo "============================================"
    echo ""

    session_exists=$(tmux ls 2>/dev/null | grep -q "^${TMUX_SESSION}:" && echo 'yes' || echo 'no')

    if [[ "${session_exists}" == "yes" ]]; then
        echo "[→] Sending interrupt signal..."
        tmux send-keys -t ${TMUX_SESSION} C-c 2>&1 || true
        sleep 1

        echo "[→] Killing tmux session '${TMUX_SESSION}'..."
        tmux kill-session -t ${TMUX_SESSION} 2>&1
        echo "[✓] Session killed"
    else
        echo "[○] No tmux session '${TMUX_SESSION}' found, nothing to do."
    fi

    echo ""
    echo "Rollback complete."
fi
