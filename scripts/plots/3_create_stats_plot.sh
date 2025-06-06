#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }


# Construct the log file name
LOG_FILE="../../logs/visualize_stats_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir ../../plots

python3 ../../src/medhelm/plots/3_visualize_stats.py \
    --stats_dir ../../data/stats/ \
    --output_dir ../../plots/