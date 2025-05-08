import json
import os

from typing import List, Dict, Any

from helm.benchmark.run_spec import RunSpec
from helm.common.codec import from_json
from helm.benchmark.metrics.metric import PerInstanceStats
from helm.benchmark.metrics.statistic import Stat
from helm.benchmark.adaptation.scenario_state import ScenarioState


def read_per_instance_stats(per_instance_stats_path: str) -> List[PerInstanceStats]:
    if not os.path.exists(per_instance_stats_path):
        raise ValueError(f"Could not load [PerInstanceStats] from {per_instance_stats_path}")
    with open(per_instance_stats_path) as f:
        return from_json(f.read(), List[PerInstanceStats])
   
   
def read_stats(stats_path: str) -> List[Stat]:
    if not os.path.exists(stats_path):
        raise ValueError(f"Could not load [Stats] from {stats_path}")
    with open(stats_path) as f:
        return from_json(f.read(), List[Stat])
    

def read_scenario_state(scenario_state_path: str) -> ScenarioState:
    if not os.path.exists(scenario_state_path):
        raise ValueError(f"Could not load Scenario from {scenario_state_path}")
    with open(scenario_state_path) as f:
        return from_json(f.read(), ScenarioState)
    
 
def read_run_spec(run_spec_path: str) -> RunSpec:
    if not os.path.exists(run_spec_path):
        raise ValueError(f"Could not load RunSpec from {run_spec_path}")
    with open(run_spec_path) as f:
        return from_json(f.read(), RunSpec)

    
def read_scenario(scenario_path: str) -> Dict[str, Any]:
    if not os.path.exists(scenario_path):
        raise ValueError(f"Could not load Scenario from {scenario_path}")
    with open(scenario_path) as scenario_file:
        scenario = json.load(scenario_file)
    return scenario
