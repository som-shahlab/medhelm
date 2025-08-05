#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
LOG_FILE="../../logs/category_heatmap_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir ../../plots

# Full heatmap

python3 ../../src/medhelm/plots/1_heatmap.py \
    --leaderboard_path ../../data/leaderboard.csv \

# Category heatmap

python3 ../../src/medhelm/plots/1_heatmap.py \
    --leaderboard_path ../../data/leaderboard.csv \
    --aggregated Category

python3 ../../src/medhelm/plots/1_heatmap.py \
    --leaderboard_path ../../data/leaderboard.csv \
    --aggregated Subcategory