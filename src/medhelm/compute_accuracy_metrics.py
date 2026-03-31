import argparse
import csv
import os
import json
import pandas as pd

from typing import Dict, List, Any
from tqdm import tqdm

from helm.benchmark.run_spec import RunSpec
from helm.benchmark.adaptation.scenario_state import ScenarioState
from helm.benchmark.adaptation.request_state import RequestState
from helm.benchmark.scenarios.scenario import Instance
from helm.common.codec import from_json
from helm.common.hierarchical_logger import hlog
from helm.benchmark.metrics.metric import PerInstanceStats

from medhelm.utils.constants import (
    BENCHMARK_NAME_MAPPING
)

PER_INSTANCE_STATS_FILE_NAME = "per_instance_stats.json"
SCENARIO_FILE_NAME = "scenario.json"
SCENARIO_STATE_FILE_NAME = "scenario_state.json"
RUN_SPEC_FILE_NAME = "run_spec.json"
INSTANCES_FILE_NAME = "instances.json"
TEST = "test"
MAIN_METRIC_RANGE = [1, 5]

BENCHMARKS = [
    "EHRSHOT", #yes, no
    # "ADHD-Behavior",
    # "ADHD-MedEffects",
    # "MedConfInfo"
    # "ProxySender" # This has 3 possible answers,
    # "PrivacyDetection",
    # "PubMedQA" # This has 3 possible answers,
    # "BMT-Status",
    "RaceBias", #yes, no
    "N2C2", #yes, no
    # "HospiceReferral",
    # "ClinicReferral",
    # "CDI-QA",
    # "ENT-Referral" # This has 3 possible answers
]


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


def read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load ScenarioState from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)

def read_instances(instance_state_path: str) -> List[Instance]:
    if not os.path.exists(instance_state_path):
        raise ValueError(f"Could not load [Instance] from {instance_state_path}")
    with open(instance_state_path) as f:
        return from_json(f.read(), List[Instance])


def get_run_dirs(runs_path: str) -> List[str]:
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
                run_dir_names.append(os.path.join(suite, p))
    run_dir_names.sort()
    return run_dir_names

def get_request_state(scenario_state: ScenarioState, id: str) -> RequestState:
    request_state = None
    for state in scenario_state.request_states:
        if state.instance.id == id:
            request_state = state
            break
    return request_state

def compute_labels(
    per_instance_stats_list: List[PerInstanceStats], 
    instances: List[Instance],
    scenario_name: str,
    model: str,
    subgroup: str
) -> Dict[str, float]:
    stats = []
    for i, stat in enumerate(per_instance_stats_list):
        instance = instances[i]
        assert stat.instance_id in instance.id, f"Instance ID mismatch: {stat.instance_id} != {instance.id}"

        # Find the label from instance.references where "correct" is in tags
        label = None
        for ref in instance.references:
            if "correct" in ref.tags:
                label = ref.output.text
                break

        # Find the "exact_match" stat
        exact_match_stat = None
        for s in stat.stats:
            if s.name.name == "exact_match":
                exact_match_stat = s
                break

        # Derive prediction: if sum == 1.0, prediction == label, else prediction != label
        if exact_match_stat is not None and label is not None:
            if exact_match_stat.sum == 1:
                prediction = label
            else:
                if label == "no":
                    prediction = "yes"
                else:
                    prediction = "no"
        else:
            prediction = None

        stats.append({
            "instance_id": stat.instance_id,
            "label": label,
            "prediction": prediction,
            "scenario_name": scenario_name,
            "model": model,
            "subgroup": subgroup
        })
    return stats


def get_classification_counts(metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    counts = {}
    for row in metrics:
        scenario_name = row["scenario_name"]
        model = row["model"]
        key = f"{scenario_name},{model}"
        if key not in counts:
            counts[key] = {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "TN": 0,
            }

        counts[key]["TP"] += row["label"] == "yes" and row["prediction"] == "yes"
        counts[key]["FP"] += row["label"] == "no" and row["prediction"] == "yes"
        counts[key]["FN"] += row["label"] == "yes" and row["prediction"] == "no"
        counts[key]["TN"] += row["label"] == "no" and row["prediction"] == "no"
    return counts


def get_precision_recall_f1(counts: Dict[str, Dict[str, int]]) -> List[Dict[str, float]]:
    results = []
    for key, metrics in counts.items():
        TP = metrics['TP']
        FP = metrics['FP']
        FN = metrics['FN']
        TN = metrics['TN']
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        scenario_name, model = key.split(",")
        results.append({
            'scenario': scenario_name,
            'model': model,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            "total": TP + FP + FN + TN,
        })
    return results


def main(
    runs_path: str,
    output_path: str
) -> None:
    metrics = []
    run_dir_names = get_run_dirs(runs_path)
    for run_dir_name in tqdm(run_dir_names, disable=None):
        scenario_path = os.path.join(runs_path, run_dir_name, SCENARIO_FILE_NAME)
        scenario = read_scenario(scenario_path)
        scenario_name = scenario["name"]
        run_spec_path = os.path.join(runs_path, run_dir_name, RUN_SPEC_FILE_NAME)
        run_spec = read_run_spec(run_spec_path)
        model = run_spec.adapter_spec.model.split("/")[-1].replace("-", "_")
        if scenario_name not in BENCHMARK_NAME_MAPPING:
            continue
        if BENCHMARK_NAME_MAPPING[scenario_name] not in BENCHMARKS:
            continue
        print(f"Computing metrics for: {BENCHMARK_NAME_MAPPING[scenario_name]}, {model}")
        per_instance_stats_path = os.path.join(runs_path, run_dir_name, PER_INSTANCE_STATS_FILE_NAME)
        per_instance_stats_list = read_per_instance_stats(per_instance_stats_path)
        instances_path = os.path.join(runs_path, run_dir_name, INSTANCES_FILE_NAME)
        instances = read_instances(instances_path)
        subgroup = ""
        if len(run_spec.scenario_spec.args) > 0:
            subgroup = run_spec.scenario_spec.args
            
        metrics.extend(
            compute_labels(
                per_instance_stats_list, 
                instances,
                scenario_name,
                model,
                subgroup
            ) 
        )
        
    counts = get_classification_counts(metrics)
    results = get_precision_recall_f1(counts)

    for metrics in results:
        print(f"{metrics['scenario']}, {metrics['model']}: Precision = {metrics['precision']:.4f}, Recall = {metrics['recall']:.4f}, F1 Score = {metrics['f1_score']:.4f}")

    print(f"Total benchmarks processed: {len(results)}")
    print("Writing results to output file:", output_path)
    df= pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", "-b", type=str, required=True, help="Path to the directory containing run outputs")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Output path for the accuracy metrics csv")
    args = parser.parse_args()

    main(
        runs_path=f"{args.benchmark_path}/runs", 
        output_path=args.output_path,
    )
