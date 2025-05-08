
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
from helm.benchmark.metrics.metric_name import MetricName
from helm.benchmark.metrics.statistic import merge_stat, Stat
from helm.benchmark.runner import remove_stats_nans, remove_per_instance_stats_nans
from helm.common.general import write, asdict_without_nones

from medhelm.utils import read_per_instance_stats, read_stats
from medhelm.utils.constants import (
    PER_INSTANCE_STATS_FILE_NAME,
    STATS_FILE_NAME,
)

# These metrics don't have their perturbation copies
NO_PERTURBATION_METRICS = [
    "training_co2_cost",
    "training_energy_cost",
    "num_instances",
    "logprob_per_byte",
    "bits_per_byte",
    "perplexity"
]


def recomupte_per_instance_stats(
    per_instance_stats: List[PerInstanceStats]
) -> None:
    """
    Recomputes all per instance stats for the leaderboard.
    """
    for instance_stats in per_instance_stats:
        # Recompute the stats here
        # This is a placeholder for the actual recomputation logic
        for stat in instance_stats.stats:
            stat.min = stat.sum
            stat.max = stat.sum
            stat.sum_squared = stat.sum * stat.sum
            stat._update_mean_variance_stddev()


def recompute_stats(
    per_instance_stats_list: List[PerInstanceStats],
    old_stats: List[Stat],
) -> List[Stat]:
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
    
    # Retrieve any value from the aggregated_stats dictionary
    count = next(iter(aggregated_stats.values()), None).count

    # Check if all values in aggregated_stats have the same count
    # In some cases it will not, as some metrics are repeated
    for metric_name, stat in aggregated_stats.items():
        if metric_name.name in NO_PERTURBATION_METRICS:
            continue
        if stat.count != count:
            print(f"WARNING: Count mismatch for metric {metric_name.name}: expected {count}, got {stat.count}")
    
    merge_stat(
        aggregated_stats,
        Stat(
            name=MetricName("num_instances", split="test"),
        ).add(count)
    )
    
    for old_stat in old_stats:
        if old_stat.name not in aggregated_stats and old_stat.name.perturbation is None:
            print(f"Adding missing {old_stat.name} to aggregated stats")
            aggregated_stats[old_stat.name] = old_stat    

    for metric_name, stat in aggregated_stats.items():
        stat = stat.take_mean()
        stats_list.append(stat)
        if metric_name.name in NO_PERTURBATION_METRICS:
            print(f"Skipping {stat} because it is in the no perturbation metrics")  
            continue
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


def get_unique_stats(
    stats: List[Stat],
) -> List[Stat]:
    """
    Get the unique stats from a list of stats.
    This method is necessary as some metric_specs used for the scenarios
    have overlapping metrics. For inatance, the metric inference_runtime
    appears in 2 of the metrics_specs, so it appears twice in the list of stats.
    """
    unique_stats = []
    seen = set()
    for stat in stats:
        if stat.name not in seen:
            seen.add(stat.name)
            unique_stats.append(stat)
    return unique_stats


def process_run_spec(run_spec_path: str):
    """
    Process a single run spec to recompute the leaderboard metrics.
    """
    per_instance_stats_path = os.path.join(run_spec_path, PER_INSTANCE_STATS_FILE_NAME)
    per_instance_stats = read_per_instance_stats(per_instance_stats_path)
    stats_path = os.path.join(run_spec_path, STATS_FILE_NAME)
    stats = get_unique_stats(read_stats(stats_path))
    recomupte_per_instance_stats(per_instance_stats)
    new_stats = recompute_stats(per_instance_stats, stats)
    # Even if we remove instances, we expect the same number of metrics to be
    # the same. If we remove metrics, this code does not support that feature.
    assert(
        len(new_stats) == len(stats)
    ), f"Expected {len(stats)} stats, got {len(new_stats)}"
    write(
        stats_path,
        json.dumps([asdict_without_nones(stat) for stat in remove_stats_nans(new_stats)], indent=2),
    )
    write(
        per_instance_stats_path,
        json.dumps(list(map(asdict_without_nones, remove_per_instance_stats_nans(per_instance_stats))), indent=2),
    )


def process_run(run_path: str):
    """
    Process a single run directory to recompute the leaderboard metrics.
    """
    for run_spec in tqdm(sorted(os.listdir(run_path)), desc="Processing run specs"):
        if "eval_cache" in run_spec or "groups" in run_spec:
            continue
        print(f"Processing run spec {os.path.join(run_path, run_spec)}")
        process_run_spec(os.path.join(run_path, run_spec))
        # NOTE: If the number of instances changed because of manual removal,
        # we need to update this function to also ensure the number of instance
        # is updated in the run spec.
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--benchmark_output", type=str, help="Where the benchmark output lives", default="benchmark_output"
    )
    args = parser.parse_args()
    benchmark_path = args.benchmark_output
    runs_path = os.path.join(benchmark_path, "runs")
    assert os.path.exists(benchmark_path), f"Benchmark path {benchmark_path} does not exist"
    assert os.path.exists(runs_path), f"Runs path {runs_path} does not exist"
    
    for run in tqdm(sorted(os.listdir(runs_path)), desc="Processing suites"):
        run_path = os.path.join(runs_path, run)
        process_run(run_path)


if __name__ == "__main__":
    main()
