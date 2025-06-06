#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }


# Construct the log file name
LOG_FILE="../../logs/cost_plots_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir ../../plots

# 1. Create Cost vs Mean Win rate plot (all)

python3 ../../src/medhelm/plots/2_costs.py \
    -c ../../data/costs.csv \
    -l ../../data/leaderboard.csv \
    -o ../../plots/cost_vs_win_rate.png