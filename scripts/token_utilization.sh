#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
LOG_FILE="../logs/token_utilization_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

python3 ../src/medhelm/token_utilization.py \
    -i ../data/costs.csv \
    -o ../data/token_utilization.csv
