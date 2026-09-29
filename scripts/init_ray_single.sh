#!/bin/bash

# ======================================================================
# Single-Node Ray Initialization Script
# ======================================================================
# Purpose: Initialize Ray in single-node mode (head only, no cluster)
#
# Usage:
#   bash init_ray_single.sh [port]
# ======================================================================

# Default port
PORT=${1:-6379}

# Initialize Conda
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="$HOME/anaconda3/bin:$PATH"
fi

echo "Starting Ray in single-node mode on port ${PORT}..."

# Start Ray head (single-node mode)
ray start --head --port=${PORT}

sleep 3

# Show Ray status
ray status

echo "Ray initialization completed"
