#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }

mkdir ../results

# Construct the log file name
LOG_FILE="../logs/find_outlierts_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

BENCHMARK_OUTPUT=~/Downloads/benchmark_output

if [ ! -d "$BENCHMARK_OUTPUT" ]; then
    echo "Error: Directory $BENCHMARK_OUTPUT does not exist."
    exit 1
fi

python3 ../src/medhelm/find_outliers.py \
    -b $BENCHMARK_OUTPUT \
    -o ../results/outliers.csv
