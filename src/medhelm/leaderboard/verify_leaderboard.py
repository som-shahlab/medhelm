import argparse
import os
import json
import pandas as pd

from typing import Dict, List, Any
from tqdm import tqdm

from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json
from helm.benchmark.metrics.metric import PerInstanceStats

from medhelm.utils.constants import (
    EXPECTED_MAX_EVAL_INSTANCES, 
    SCENARIO_FILE_NAME, 
    RUN_SPEC_FILE_NAME, 
    STATS_FILE_NAME, 
    PER_INSTANCE_STATS_FILE_NAME
)

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

def get_expected_max_eval_instances(scenario_name: str) -> bool:
    expected_max_eval_instances = EXPECTED_MAX_EVAL_INSTANCES["default"]
    for key, val in EXPECTED_MAX_EVAL_INSTANCES.items():
        if key in scenario_name:
            expected_max_eval_instances = val
            break
    return expected_max_eval_instances
    

def verify_leaderboard(
    runs_path: str,
    output_path: str
) -> None:
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
    verifications = []
    not_passed_benchmarks: List[str] =[]
    
    for run_dir_name in tqdm(run_dir_names, disable=None):
        per_instance_stats_path: str = os.path.join(runs_path, run_dir_name, PER_INSTANCE_STATS_FILE_NAME)
        scenario_path: str = os.path.join(runs_path, run_dir_name, SCENARIO_FILE_NAME)
        run_spec_path: str = os.path.join(runs_path, run_dir_name, RUN_SPEC_FILE_NAME)
        stats_path: str = os.path.join(runs_path, run_dir_name, STATS_FILE_NAME)
        run_spec = None
        scenario = None
        if os.path.exists(run_spec_path):
            run_spec = read_run_spec(run_spec_path)
        if os.path.exists(scenario_path):
            scenario = read_scenario(scenario_path)
        expected_max_eval_instances = get_expected_max_eval_instances(scenario["name"] if scenario else "default")
        verification: Dict[str, bool] = {
            "scenario": scenario["name"] if scenario else run_dir_name,
            "model": run_spec.adapter_spec.model.split("/")[-1].replace("-", "_") if run_spec else None,
            "run_spec_exists": os.path.exists(run_spec_path),
            "scenario_exists": os.path.exists(scenario_path),
            "stats_exists": os.path.exists(stats_path),
            "per_instance_stats_exists": os.path.exists(per_instance_stats_path),
            "max_eval_instances_ok": run_spec and run_spec.adapter_spec.max_eval_instances == expected_max_eval_instances,
            "max_eval_instances": run_spec.adapter_spec.max_eval_instances if run_spec else None,
            "expected_max_eval_instances": expected_max_eval_instances
        }
        
        verifications.append(verification)
        if not all(verification.values()):
            not_passed_benchmarks.append(verification)
    
    if not_passed_benchmarks:
        print("Verifications Failed!")
        for benchmark in not_passed_benchmarks:
            print(benchmark)
        raise Exception(f"{len(not_passed_benchmarks)}/{len(run_dir_names)} Verifications failed!")
    else:
        print(f"All {len(run_dir_names)} verifications passed!")
    
    df = pd.DataFrame(verifications)
    df.to_csv(output_path, index=False)
    
    print(f"Verification successfully saved at {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-path", type=str, help="Where the benchmarking output lives", default="benchmark_output"
    )
    parser.add_argument(
        "-o", "--output-path", type=str, help="Where to save the verification results", default="results"
    )
    args = parser.parse_args()
    benchmark_path = args.input_path
    output_path = args.output_path
    runs_path = os.path.join(benchmark_path, "runs")
    verify_leaderboard(runs_path, output_path)


if __name__ == "__main__":
    main()
