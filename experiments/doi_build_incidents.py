"""Write the incident report datasets: dev (seeds 0..19) and test (seeds 100..149).

Seeds 200..229 are reserved for the simulation experiments E7 and E8 and must never appear here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.doi.config import SimConfig
from src.doi.incidents import build_items, write_jsonl
from src.doi.scenarios import build_scenario

SPLITS = {"dev": range(0, 20), "test": range(100, 150)}
SCENARIOS = ("incidents_aisles", "incidents_room")


def build(split: str, seeds) -> list:
    items = []
    for name in SCENARIOS:
        for seed in seeds:
            cfg = SimConfig(scenario=name, n_robots=2, tasks_per_robot=2, seed=seed,
                            scenario_params={"p_false": 0.2})
            items += build_items(build_scenario(cfg), split)
    return items


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "incidents")
    os.makedirs(out_dir, exist_ok=True)
    for split, seeds in SPLITS.items():
        assert not any(200 <= s <= 229 for s in seeds)
        items = build(split, seeds)
        write_jsonl(items, os.path.join(out_dir, f"{split}.jsonl"))
        print(f"{split}: {len(items)} items")


if __name__ == "__main__":
    main()
