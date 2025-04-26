import argparse
import os
from typing import Dict, List, Any, Set
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt


from helm.benchmark.run_spec import RunSpec
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.common.codec import from_json, to_json
from helm.common.hierarchical_logger import hlog
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.metrics.statistic import Stat


BENCHMARK_METRICS: Dict[str, str] = {
    "medcalc_bench": "medcalc_bench_accuracy",
    "clear": "exact_match",
    "mtsamples_replicate": "mtsamples_replicate_accuracy",
    "medec": "medec_error_flag_accuracy",
    "ehrshot": "exact_match",
    "head_qa": "exact_match",
    "medbullet": "exact_match",
    "medalign": "medalign_accuracy",
    "shc_ptbm_med": "exact_match",
    "shc_sei_med": "exact_match",
    "dischargeme": "dischargeme_accuracy",
    "aci_bench": "aci_bench_accuracy",
    "mimic_bhc": "mimic_bhc_accuracy",
    "mtsamples": "mtsamples_procedures_accuracy",
    "mimic_rrs": "mimic_rrs_accuracy",
    "chw_care_plan": "chw_care_plan_accuracy",
    "medication_qa": "medication_qa_accuracy",
    "starr_patient_instructions": "starr_patient_instructions_accuracy",
    "med_dialog": "med_dialog_accuracy",
    "shc_conf_med": "exact_match",
    "medi_qa": "medi_qa_accuracy",
    "mental_health": "mental_health_accuracy",
    "pubmed_qa": "exact_match",
    "ehr_sql": "ehr_sql_execution_accuracy",
    "shc_bmt_med": "exact_match",
    "race_based_med": "exact_match",
    "medhallu": "exact_match",
    "n2c2_ct_matching": "exact_match",
    "shc_gip_med": "exact_match",
    "mimiciv_billing_code": "mimiciv_billing_code_f1",
    "shc_sequoia_med": "exact_match",
    "shc_cdi_med": "exact_match",
    "shc_ent_med": "exact_match",
    "shc_privacy_med": "exact_match",
    "shc_proxy_med": "exact_match",
}

OPEN_ENDED_BENCHMARKS: Set[str] = {
    "aci_bench",
    "med_dialog",
    "medi_qa",
    "medication_qa",
    "mtsamples",
    "mtsamples_replicate",
    "medalign",
    "chw_care_plan"
    "mimic_rrs",
    "mimic_bhc",
    "dischargeme",
    "starr_patient_instructions",
    "mental_health"
}


PER_INSTANCE_STATS_FILE_NAME = "per_instance_stats.json"
SCENARIO_FILE_NAME = "scenario.json"
RUN_SPEC_FILE_NAME = "run_spec.json"
TEST = "test"


def read_per_instance_stats(per_instance_stats_path: str) -> List[PerInstanceStats]:
    if not os.path.exists(per_instance_stats_path):
        raise ValueError(f"Could not load [PerInstanceStats] from {per_instance_stats_path}")
    with open(per_instance_stats_path) as f:
        return from_json(f.read(), List[PerInstanceStats])
   
 
def read_run_spec(run_spec_path: str) -> RunSpec:
    if not os.path.exists(run_spec_path):
        raise ValueError(f"Could not load RunSpec from {run_spec_path}")
    with open(run_spec_path) as f:
        return from_json(f.read(), RunSpec)

    
def read_scenario(scenario_path: str) -> Dict[str, Any]:
    with open(scenario_path) as scenario_file:
        scenario = json.load(scenario_file)
    return scenario


def get_main_metric_stats(
    per_instance_stats_list: List[PerInstanceStats], main_metric_name: str
) -> List[Stat]:
    main_metric_stats: List[Stat] = []
    for per_instance_stats in per_instance_stats_list:
        for stat in per_instance_stats.stats:
            if stat.name.name == main_metric_name and stat.name.split == TEST:
                main_metric_stats.append(stat)
    main_metric_stats.sort(key=lambda x: x.sum)
    return main_metric_stats

def compute_statistics(
    main_metric_stats: List[Stat]
) -> Dict[str, float]:
    numpy_array = np.array([stat.sum for stat in main_metric_stats])
    mean= np.mean(numpy_array)
    median = np.median(numpy_array)
    stddev = np.std(numpy_array)
    variance = np.var(numpy_array)
    minimum = main_metric_stats[0].sum
    maximum = main_metric_stats[-1].sum
    return {
        "mean": mean,
        "median": median,
        "stddev": stddev,
        "variance": variance,
        "minimum": minimum,
        "maximum": maximum
    }
        

def generate_plots(
    runs_path: str,
    output_dir: str
) -> None:
    metric_data: Dict[str, Dict[str, List[float]]] = {}
    suite_names = [
        p for p in os.listdir(runs_path)
        if os.path.isdir(os.path.join(runs_path, p))
    ]

    run_dir_names = []

    for suite in suite_names:
        run_suite_path = os.path.join(runs_path, suite)
        for p in os.listdir(run_suite_path):
            full_path = os.path.join(run_suite_path, p)
            if p not in {"eval_cache", "groups"} and os.path.isdir(full_path):
                run_dir_names.append(os.path.join(suite, p))  # Keeping track of suite/run

    run_dir_names.sort()
    
    for run_dir_name in tqdm(run_dir_names, disable=None):
        per_instance_stats_path: str = os.path.join(runs_path, run_dir_name, PER_INSTANCE_STATS_FILE_NAME)
        scenario_path: str = os.path.join(runs_path, run_dir_name, SCENARIO_FILE_NAME)
        run_spec_path: str = os.path.join(runs_path, run_dir_name, RUN_SPEC_FILE_NAME)
        if not os.path.exists(per_instance_stats_path):
            hlog(f"WARNING: {run_dir_name} doesn't have {PER_INSTANCE_STATS_FILE_NAME}, skipping")
            continue
        if not os.path.exists(scenario_path):
            hlog(f"WARNING: {run_dir_name} doesn't have {SCENARIO_FILE_NAME}, skipping")
            continue
        if not os.path.exists(run_spec_path):
            hlog(f"WARNING: {run_dir_name} doesn't have {RUN_SPEC_FILE_NAME}, skipping")
            continue
        scenario = read_scenario(scenario_path)
        scenario_name = scenario["name"]
        if scenario_name not in OPEN_ENDED_BENCHMARKS:
            continue
        run_spec = read_run_spec(run_spec_path)
        model = run_spec.adapter_spec.model.split("/")[-1]
        if model not in metric_data:
            metric_data[model] = {}
        
        metric_name = BENCHMARK_METRICS.get(scenario_name)

        if not metric_name:
            hlog(f"WARNING: No benchmark metric defined for scenario {scenario_name}, skipping")
            continue

        per_instance_stats_list = read_per_instance_stats(per_instance_stats_path)
        main_metric_stats = get_main_metric_stats(per_instance_stats_list, metric_name)

        if scenario_name not in metric_data[model]:
            metric_data[model][scenario_name] = []
        metric_data[model][scenario_name].extend([stat.sum for stat in main_metric_stats])
        print(f"Metric statistics for {model},{scenario_name}:")
        print(compute_statistics(main_metric_stats))

    for model in metric_data.keys():
        generate_box_plots(metric_data, output_dir, model)


def generate_box_plots(
    metric_data: Dict[str, Dict[str,List[float]]],
    output_dir: str, model: str
):
    """Generate and save box plots for metric data."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    scenarios = list(metric_data[model].keys())
    data = [metric_data[model][scenario] for scenario in scenarios]

    plt.figure(figsize=(15, 8))
    plt.boxplot(data, labels=scenarios, showfliers=False)
    plt.xticks(rotation=90)
    plt.ylabel("Metric Value")
    plt.title(f"Performance of {model} on open-ended benchmarks")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{model}_box_plot.png")
    plt.savefig(plot_path)
    hlog(f"Box plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--benchmark-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Where to save the plots", default="plots"
    )
    args = parser.parse_args()
    benchmark_path = args.benchmark_path
    output_dir = args.output_dir
    runs_path = os.path.join(benchmark_path, "runs")
    generate_plots(runs_path, output_dir)


if __name__ == "__main__":
    main()
