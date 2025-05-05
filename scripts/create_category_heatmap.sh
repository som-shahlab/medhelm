#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
LOG_FILE="../logs/category_heatmap_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir ../plots

python3 ../src/medhelm/plots/plot_category_heatmap.py \
    --leaderboard_path ../data/leaderboard.csv \
    --output_path ../plots/category_heatmap_aggregated.png

