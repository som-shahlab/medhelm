
import argparse
import json
import os

from collections import defaultdict
from dataclasses import replace
from tqdm import tqdm
from typing import List

from helm.benchmark.augmentations.perturbation_description import (
    PerturbationDescription,
)
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.metrics.statistic import merge_stat, Stat
from helm.benchmark.runner import remove_per_instance_stats_nans
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


def recomupte_per_instance_stats(
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


def recompute_stats(per_instance_stats_list: List[PerInstanceStats]) -> List[Stat]:
    """
    Recomputes all stats for the leaderboard.
    NOTE: For MedHELM, we didn't add any fairness or robustness perturbations,
    so the robustness and fairness stats are just copies of the original stats.
    """
    aggregated_stats = defaultdict(Stat)
    for per_instance_stats in per_instance_stats_list:
        for stat in per_instance_stats.stats:
            merge_stat(aggregated_stats, stat)

    stats_list = []
    fairness_stats_list = []
    robustness_stats_list = []

    for metric_name, stat in aggregated_stats.items():
        stat = stat.take_mean()
        stats_list.append(stat)
        # Add robustness and fairness stats
        robustness_stats_list.append(
            Stat(
                replace(metric_name, perturbation=PerturbationDescription(name="robustness", robustness=True))
            )
        )
        fairness_stats_list.append(
            Stat(
                replace(metric_name, perturbation=PerturbationDescription(name="fairness", fairness=True))
            )
        )
    stats_list.extend(robustness_stats_list)
    stats_list.extend(fairness_stats_list)
    return stats_list


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
        pass
    metric_name = f"{scenario['name']}_accuracy"
    recomupte_per_instance_stats(per_instance_stats, metric_name, min_value)
    
    write(
        per_instance_stats_path,
        json.dumps(list(map(asdict_without_nones, remove_per_instance_stats_nans(per_instance_stats))), indent=2),
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
