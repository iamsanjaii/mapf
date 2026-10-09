"""The forecast-guard work must not change any existing arm. These costs were recorded before it started."""
import pytest
from src.doi.config import SimConfig
from src.doi.runner import run_arms

PINNED = {
    "single_block": {"rof": 519.0, "rof_p": 519.0, "central": 519.0},
    "multi_block_wall": {"rof": 507.0, "rof_p": 503.0, "central": 498.0},
    "shift": {"rof": 501.0, "rof_p": 501.0, "central": 501.0},
}


@pytest.mark.parametrize("scenario", sorted(PINNED))
def test_existing_arms_cost_what_they_did(scenario):
    res = run_arms(SimConfig(scenario=scenario, n_robots=6, tasks_per_robot=6, seed=2), ["rof", "rof_p", "central"])
    assert {name: r.J for name, r in res.items()} == PINNED[scenario]
