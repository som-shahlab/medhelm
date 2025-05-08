
import argparse
import json
import os

from tqdm import tqdm
from typing import List

from helm.benchmark.metrics.metric import PerInstanceStats
from helm.common.general import write, asdict_without_nones

from medhelm.utils import (
    read_per_instance_stats,
    read_scenario
)
from medhelm.utils.constants import (
    PER_INSTANCE_STATS_FILE_NAME,
    SCENARIO_FILE_NAME,
    OPEN_ENDED_BENCHMARKS,
)


def recompute_per_instance_stats(
    per_instance_stats: List[PerInstanceStats],
    metric_name: str,
    min_value: float = 1,
) -> None:
    """
    Recomputes all per instance stats for the leaderboard.
    """
    for instance_stats in per_instance_stats:
        # Recompute the stats here
        # This is a placeholder for the actual recomputation logic
        for stat in instance_stats.stats:
            if stat.name.name != metric_name:
                continue
            if stat.sum < min_value:
                stat.sum = min_value
                stat.min = stat.sum
                stat.max = stat.sum
                stat.sum_squared = stat.sum * stat.sum
                stat._update_mean_variance_stddev()


def process_run_spec(run_spec_path: str, min_value: float = 1):
    """
    Process a single run spec to replace jury scores.
    """
    per_instance_stats_path = os.path.join(run_spec_path, PER_INSTANCE_STATS_FILE_NAME)
    per_instance_stats = read_per_instance_stats(per_instance_stats_path)
    scenario = read_scenario(os.path.join(run_spec_path, SCENARIO_FILE_NAME))
    if scenario["name"] not in OPEN_ENDED_BENCHMARKS:
        # If the benchmark is not open ended, we don't need to recompute the stats
        # because those don't have any jury scores
        return
    metric_name = f"{scenario['name']}_accuracy"
    recompute_per_instance_stats(per_instance_stats, metric_name, min_value)
    
    write(
        per_instance_stats_path,
        json.dumps(list(map(asdict_without_nones, per_instance_stats)), indent=2),
    )


def process_run(run_path: str, min_value: float = 1):
    """
    Process a single run directory to recompute the leaderboard metrics.
    """
    for run_spec in tqdm(sorted(os.listdir(run_path)), desc="Processing run specs"):
        if "eval_cache" in run_spec or "groups" in run_spec:
            continue
        process_run_spec(os.path.join(run_path, run_spec), min_value)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--benchmark_output", type=str, help="Where the benchmark output lives", default="benchmark_output"
    )
    parser.add_argument(
        "-m", "--min_value", type=float, help="Min jury value", default=1
    )
    args = parser.parse_args()
    benchmark_path = args.benchmark_output
    min_value = args.min_value
    runs_path = os.path.join(benchmark_path, "runs")
    assert os.path.exists(benchmark_path), f"Benchmark path {benchmark_path} does not exist"
    assert os.path.exists(runs_path), f"Runs path {runs_path} does not exist"
    
    for run in tqdm(sorted(os.listdir(runs_path)), desc="Processing suites"):
        run_path = os.path.join(runs_path, run)
        process_run(run_path, min_value)


if __name__ == "__main__":
    main()
