#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }

mkdir ../results

# Construct the log file name
DATE=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="../logs/verify_leaderboard_$DATE.log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

python3 ../src/medhelm/verify_leaderboard.py \
    -i ~/Downloads/benchmark_output \
    -o ../results/leaderboard_verification_$DATE.csv
