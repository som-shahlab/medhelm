#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
cd "$SCRIPT_DIR/$DIR" || { echo "Failed to change directory"; exit 1; }


# Construct the log file name
LOG_FILE="../logs/cost_plots_$(date '+%Y-%m-%d_%H-%M-%S').log"
mkdir -p ../logs  # Ensure the logs directory exists
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir ../plots

# 1. Create Cost vs Mean Win rate plot (all)

python3 ../src/medhelm/plots/plot_costs.py \
    -c ../data/costs.csv \
    -l ../data/leaderboard.csv \
    -o ../plots/cost_vs_win_rate.png

# 2. Create Cost vs Mean Win rate plot (Clinical Decision Support)

python3 ../src/medhelm/plots/plot_costs.py \
    -c ../data/costs.csv \
    -l ../data/leaderboard_clinical_decision_support.csv \
    --category "Clinical Decision Support" \
    -o ../plots/cost_vs_win_rate_clinical_decision_support.png

# 3. Create Cost vs Mean Win rate plot (Clinical Note Generation)

python3 ../src/medhelm/plots/plot_costs.py \
    -c ../data/costs.csv \
    -l ../data/leaderboard_clinical_note_generation.csv \
    --category "Clinical Note Generation" \
    -o ../plots/cost_vs_win_rate_clinical_note_generation.png

# 4. Create Cost vs Mean Win rate plot (Patient Communication and Education)

python3 ../src/medhelm/plots/plot_costs.py \
    -c ../data/costs.csv \
    -l ../data/leaderboard_patient_communication_and_education.csv \
    --category "Patient Communication and Education" \
    -o ../plots/cost_vs_win_rate_patient_communication_and_education.png

# 5. Create Cost vs Mean Win rate plot (Medical Research Assistance)

python3 ../src/medhelm/plots/plot_costs.py \
    -c ../data/costs.csv \
    -l ../data/leaderboard_medical_research_assistance.csv \
    --category "Medical Research Assistance" \
    -o ../plots/cost_vs_win_rate_medical_research_assistance.png

# 6. Create Cost vs Mean Win rate plot (Administration and Workflow)

python3 ../src/medhelm/plots/plot_costs.py \
    -c ../data/costs.csv \
    -l ../data/leaderboard_administration_and_workflow.csv \
    --category "Administration and Workflow" \
    -o ../plots/cost_vs_win_rate_administration_and_workflow.png

