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
from helm.common.codec import from_json
from helm.common.hierarchical_logger import hlog
from helm.benchmark.metrics.metric import PerInstanceStats

from medhelm.utils.constants import (
    BENCHMARK_METRICS, 
    OPEN_ENDED_BENCHMARKS, 
    BENCHMARK_NAME_MAPPING
)

PER_INSTANCE_STATS_FILE_NAME = "per_instance_stats.json"
SCENARIO_FILE_NAME = "scenario.json"
SCENARIO_STATE_FILE_NAME = "scenario_state.json"
RUN_SPEC_FILE_NAME = "run_spec.json"
TEST = "test"
MAIN_METRIC_RANGE = [1, 5]


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


def get_metric_stats(per_instance_stats_list: List[PerInstanceStats], metric_name: str) -> Dict[str, float]:
    instance_to_value = {}
    for per_instance_stats in per_instance_stats_list:
        for stat in per_instance_stats.stats:
            if stat.name.name == metric_name and stat.name.split == TEST:
                instance_to_value[per_instance_stats.instance_id] = stat.sum
    return instance_to_value


def find_outliers(data: Dict[str, float]) -> Dict[str, float]:
    """
    Identifies outliers in a dataset based on IQR method.
    
    Args:
        data (dict): A dictionary where keys are instance IDs and values are scores.
    
    Returns:
        dict: A dictionary of outlier instance IDs and their corresponding scores.
    """
    if not data:
        return {}

    scores = list(data.values())
    sorted_scores = sorted(scores)
    
    # Calculate Q1 and Q3
    n = len(sorted_scores)
    q1 = sorted_scores[n // 4]
    q3 = sorted_scores[3 * n // 4]
    iqr = q3 - q1

    # Define outlier bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Identify outliers
    outliers = {k: v for k, v in data.items() if v < lower_bound or v > upper_bound}
    
    return outliers


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

def get_gold_response(request_state: RequestState) -> str:
    gold_response = None
    for reference_response in request_state.instance.references:
        if "correct" in reference_response.tags:
            gold_response = reference_response.output
            break
    return gold_response

def get_scenario_state_dict(scenario_states: List[Dict[str, Any]], id: str):
    scenario_state_dict = scenario_states[0]
    if len(scenario_states) == 1:
        return scenario_state_dict
    for state in scenario_states:
        if not state["suffix"]:
            if not "-" in id:
                scenario_state_dict = state
                break
        else:
            if state["suffix"] in id:
                scenario_state_dict = state
                break
    return scenario_state_dict
    

def aggregate_outliers_data(
    outliers: Dict[str, float], 
    secondary_scores: Dict[str, float],
    secondary_metric: str,
    scenario_states: List[Dict[str, Any]],
    scenario_name: str
) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for id, score in outliers.items():
        id_original = id
        if "-" in id_original:
            id_original = id_original.split("-")[0]
        state_dict = get_scenario_state_dict(scenario_states, id)
        
        scenario_state = state_dict["scenario_state"]
        run_dir = state_dict["run_dir_name"]
        request_state = get_request_state(scenario_state, id_original)
        gold_response = get_gold_response(request_state)
        annotations = request_state.annotations
        del annotations[scenario_name]["prompt_text"]
        data.append(
            {
                "run_dir": run_dir,
                "id": id_original,
                "jury_score": score,
                secondary_metric: secondary_scores[id],
                "prompt": request_state.request.prompt,
                "response": request_state.result.completions[0].text,
                "gold_response": gold_response,
                "jury_annotations": annotations
            }
        )
    return data

def main(runs_path: str, output_path: str) -> None:
    benchmark_instance_scores_main: Dict[str, Dict[str, List[float]]] = {}
    benchmark_instance_scores_secondary: Dict[str, Dict[str, List[float]]] = {}
    
    run_dir_names = get_run_dirs(runs_path)
    all_outliers_data: List[Dict[str, Any]]= []
    scenario_states = {}
    processed = {}
    for run_dir_name in tqdm(run_dir_names, disable=None):
        per_instance_stats_path = os.path.join(runs_path, run_dir_name, PER_INSTANCE_STATS_FILE_NAME)
        scenario_path = os.path.join(runs_path, run_dir_name, SCENARIO_FILE_NAME)
        scenario_state_path = os.path.join(runs_path, run_dir_name, SCENARIO_STATE_FILE_NAME)
        run_spec_path = os.path.join(runs_path, run_dir_name, RUN_SPEC_FILE_NAME)
        run_spec = read_run_spec(run_spec_path)
        model = run_spec.adapter_spec.model.split("/")[-1].replace("-", "_")
        
        scenario_state = read_scenario_state(scenario_state_path)
        if not (os.path.exists(per_instance_stats_path) and os.path.exists(scenario_path) and os.path.exists(run_spec_path)):
            hlog(f"WARNING: Missing files in {run_dir_name}, skipping")
            continue
        
        scenario = read_scenario(scenario_path)
        scenario_name = scenario["name"]
        if scenario_name not in OPEN_ENDED_BENCHMARKS:
            continue
        main_metric_name = BENCHMARK_METRICS.get(scenario_name)
        secondary_metric_name = "BERTScore-F"
        if not main_metric_name:
            hlog(f"WARNING: No benchmark metric defined for scenario {scenario_name}, skipping")
            continue
        
        if scenario_name not in scenario_states:
            scenario_states[scenario_name] = [{
                "model": model,
                "scenario_state": scenario_state,
                "run_dir_name": run_dir_name,
                "suffix": None
            }]
        if processed.get(f"{scenario_name},{model}", 0) > 0 and scenario_states[scenario_name][0]["model"] == model:
            scenario_states[scenario_name].append({
                "model": model,
                "scenario_state": scenario_state,
                "run_dir_name": run_dir_name,
                "suffix": f"-{processed.get(f'{scenario_name},{model}')}"
            })

        per_instance_stats_list = read_per_instance_stats(per_instance_stats_path)
        main_scores = get_metric_stats(per_instance_stats_list, main_metric_name)
        secondary_scores = get_metric_stats(per_instance_stats_list, secondary_metric_name)

        
        if f"{scenario_name},{model}" not in processed:
            processed[f"{scenario_name},{model}"] = 0
        if scenario_name not in benchmark_instance_scores_main:
            benchmark_instance_scores_main[scenario_name] = {}
        if scenario_name not in benchmark_instance_scores_secondary:
            benchmark_instance_scores_secondary[scenario_name] = {}
        for instance_id, score in main_scores.items():
            benchmark_instance_scores_main[scenario_name].setdefault(f"{instance_id}-{processed.get(f'{scenario_name},{model}')}" if processed.get(f"{scenario_name},{model}") > 0 else instance_id, []).append(score)
        for instance_id, score in secondary_scores.items():
            benchmark_instance_scores_secondary[scenario_name].setdefault(f"{instance_id}-{processed.get(f'{scenario_name},{model}')}" if processed.get(f"{scenario_name},{model}") > 0 else instance_id, []).append(score)
        processed[f"{scenario_name},{model}"] += 1
    
    # Average and normalize scores
    data_main: Dict[str, Dict[str, float]]= {}
    data_secondary: Dict[str, Dict[str, float]] = {}
    labels = []

    for scenario_name in sorted(benchmark_instance_scores_main.keys()):
        main_scores = {}
        
        for id, scores in benchmark_instance_scores_main[scenario_name].items():
            avg_score = sum(scores) / len(scores)
            min_val, max_val = MAIN_METRIC_RANGE
            norm_score = (avg_score - min_val) / (max_val - min_val)
            main_scores[id] = norm_score
        data_main[scenario_name] = main_scores

        secondary_scores = {}
        for id, scores in benchmark_instance_scores_secondary[scenario_name].items():
            avg_score = sum(scores) / len(scores)
            secondary_scores[id] = avg_score
        data_secondary[scenario_name] = secondary_scores

        label = BENCHMARK_NAME_MAPPING.get(scenario_name, scenario_name)
        labels.append(label)
    
    
    for scenario, main_scores in data_main.items():
        secondary_scores = data_secondary[scenario]
        outliers = find_outliers(main_scores)
        
        outliers_data = aggregate_outliers_data(
            outliers,
            secondary_scores,
            secondary_metric_name,
            scenario_states[scenario],
            scenario
        )
        all_outliers_data.extend(outliers_data)
    
    df = pd.DataFrame(all_outliers_data)

    # Save to CSV
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    print(f"Outliers successfully saved to {output_path}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark_path", "-b", type=str, required=True, help="Path to the directory containing run outputs")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Output directory for the figure")
    args = parser.parse_args()

    main(f"{args.benchmark_path}/runs", args.output_path)
