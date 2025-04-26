import os
import json
import pandas as pd
import re
import argparse

# Mapping dictionary for Benchmark renaming
benchmark_mapping = {
    "aci_bench": "ACIBench",
    "chw_care_plan": "NoteExtract",
    "clear": "CLEAR",
    "dischargeme": "DischargeMe",
    "ehr_sql": "EHR-SQL",
    "ehrshot": "EHRSHOT",
    "head_qa": "HEADQA",
    "med_dialog": "MedDialog",
    "medalign": "MedAlign",
    "medbullets": "MedBullets",
    "medcalc_bench": "MedCalcBench",
    "medec": "Medec",
    "medi_qa": "MediQA",
    "medication_qa": "MedicationQA",
    "mental_health": "MentalHealth",
    "mimic_bhc": "MIMICBHC",
    "mimic_rrs": "MIMICRRS",
    "mimiciv_billing_code": "MIMICIV-BillingCode",
    "mtsamples_procedures": "MTSamples-Procedures",
    "mtsamples_replicate": "MTSamples",
    "n2c2_ct_matching": "N2C2",
    "pubmed_qa": "PubMedQA",
    "race_based_med": "RaceBias",
    "medhallu": "MedHallu",
    "starr_patient_instructions": "PatientInstruct"
}

benchmark_to_question_type = {
    "aci_bench": "Open",
    "chw_care_plan": "Open",
    "clear": "Closed",
    "dischargeme": "Open",
    "ehr_sql": "Closed",
    "ehrshot": "Closed",
    "head_qa": "Closed",
    "med_dialog": "Open",
    "medalign": "Open",
    "medbullets": "Closed",
    "medcalc_bench": "Closed",
    "medec": "Closed",
    "medi_qa": "Open",
    "medication_qa": "Open",
    "mental_health": "Open",
    "mimic_bhc": "Open",
    "mimic_rrs": "Open",
    "mimiciv_billing_code": "Closed",
    "mtsamples_procedures": "Open",
    "mtsamples_replicate": "Open",
    "n2c2_ct_matching": "Closed",
    "pubmed_qa": "Closed",
    "race_based_med": "Closed",
    "medhallu": "Closed",
    "starr_patient_instructions": "Open",
}

def parse_directory_name(directory):
    benchmark, sep, rest = directory.partition(':')
    subset = None
    if "med_dialog" in benchmark:
        subset = benchmark.split(",")[-1]
        benchmark = "med_dialog"
    subset = None
    if sep:
        subset_part = rest.split('model=')[0].rstrip(',')
        subset = subset_part if subset_part else subset

    model_match = re.search(r'model=([^,]+)', directory)
    model_raw = model_match.group(1) if model_match else 'unknown_model'
    model_name = re.sub(r'^[^_]+_', '', model_raw).replace('-', '_')

    return benchmark, subset, model_name

def extract_tokens(stats_file):
    try:
        with open(stats_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        print(f"Error reading {stats_file}: {e}")
        return 0, 0, 0

    input_tokens = 0
    output_tokens = 0
    test_instances = 0

    for item in data:
        if isinstance(item, dict) and 'name' in item and 'sum' in item:
            name = item['name']['name']
            if name == 'num_prompt_tokens':
                input_tokens = item['sum']
            elif name == 'max_num_completion_tokens':
                output_tokens = item['sum']
            elif name == 'num_completions':
                test_instances = item['sum']

    return input_tokens, output_tokens, test_instances

def generate_table(base_dir):
    rows = []

    if not os.path.isdir(base_dir):
        print(f"Error: The base directory '{base_dir}' does not exist.")
        return pd.DataFrame()

    directories = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for directory in directories:
        stats_file = os.path.join(directory, 'stats.json')

        if not os.path.isfile(stats_file):
            print(f"Skipping {directory}: stats.json not found.")
            continue

        benchmark, _, model = parse_directory_name(os.path.basename(directory))
        input_tokens, output_tokens, test_instances = extract_tokens(stats_file)

        row = {
            'Model': model,
            'Benchmark': benchmark,
            'Question Type': benchmark_to_question_type.get(benchmark, 'Unknown'),
            'Total Input Tokens': input_tokens,
            'Total Output Tokens': output_tokens,
            'Test Instances': test_instances,
        }
        rows.append(row)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Generate and aggregate benchmark stats.")
    parser.add_argument("base_dir", type=str, help="Path to directory containing benchmark runs.")
    parser.add_argument("--output_file", type=str, default="aggregated_results.csv", help="Path to save aggregated CSV.")
    args = parser.parse_args()

    # Step 1: Generate raw stats
    table = generate_table(args.base_dir)

    if table.empty:
        print("No valid data found.")
        return

    # Step 2: Apply benchmark name mapping
    table['Benchmark'] = table['Benchmark'].map(benchmark_mapping).fillna(table['Benchmark'])

    # Step 3: Aggregate
    aggregated = table.groupby(['Model', 'Benchmark', 'Question Type'], dropna=False).agg({
        'Test Instances': 'sum',
        'Total Input Tokens': 'sum',
        'Total Output Tokens': 'sum',
    }).reset_index()

    # Step 4: Save aggregated result
    aggregated.to_csv(args.output_file, index=False)
    print(f"Aggregated results saved to {args.output_file}")

if __name__ == "__main__":
    main()
