#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }

# Construct the log file name
LOG_FILE="../logs/box_plots_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir ../plots

BENCHMARK_OUTPUT=~/Downloads/benchmark_output

if [ ! -d "$BENCHMARK_OUTPUT" ]; then
    echo "Error: Directory $BENCHMARK_OUTPUT does not exist."
    exit 1
fi

python3 ../src/medhelm/plots/plot_open_ended_box_plots_aggregated.py \
    -b $BENCHMARK_OUTPUT \
    -o ../plots
