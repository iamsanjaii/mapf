"""Stage 2 story and window: captions, rent meters for carries and fills, slots, stored items and the load marker."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")

from src.doi.animate import COLORS, Scene, _background, animate_runs
from src.doi.config import SimConfig
from src.doi.metrics import carried_kind
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii
from src.doi.story import obstacle_prices, plain_events, rent_meters

DOOR_CRATE = ["...T#....", "....#....", "....C....", "....#....", "....#...."]
PIT_ROOM = ["....#....", "....#....", "....P....", "..R.#....", "....#...."]


def run(rows):
    s = scenario_from_ascii(rows, [(2, 1)], [[(2, 7)]])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1, policy="myopic", kappa=50.0)
    return run_episode(cfg, scenario=s), s, cfg


def test_captions_describe_a_carry():
    r, s, _ = run(DOOR_CRATE)
    texts = [t for _, t in plain_events(r, s)]
    assert ("Detours so far cost 50, carrying the crate at (2,4) to the rack at (0,3) costs 4: "
            "robot 0 decides to carry it") in texts
    assert "Robot 0 carried the crate from (2,4) to the rack at (0,3)" in texts


def test_captions_describe_a_fill():
    r, s, _ = run(PIT_ROOM)
    texts = [t for _, t in plain_events(r, s)]
    assert ("Detours so far cost 50, filling the pit at (2,4) with the debris at (3,2) costs 2: "
            "robot 0 decides to fill it") in texts
    assert "Robot 0 filled the pit at (2,4) with debris from (3,2)" in texts


def test_the_meter_for_a_carry_sits_on_the_obstacle_and_says_carried_away():
    r, s, _ = run(DOOR_CRATE)
    prices = obstacle_prices({"myopic": r})
    assert prices == {(2, 4): (4.0, "crate")}
    meter, = rent_meters(r, prices)
    assert meter.cumulative[-1] == 50 and meter.cell == (2, 4)
    assert meter.status(0) == "carrying"                       # decided at tick 0, not yet done
    assert meter.status(r.ticks) == "carried away"


def test_the_meter_for_a_fill_sits_on_the_pit_and_says_filled():
    r, s, _ = run(PIT_ROOM)
    prices = obstacle_prices({"myopic": r})
    assert prices == {(2, 4): (2.0, "pit")}
    meter, = rent_meters(r, prices)
    assert meter.cell == (2, 4) and meter.status(r.ticks) == "filled"


def test_carried_kind_follows_the_load():
    r, s, _ = run(DOOR_CRATE)
    assert carried_kind(r, 0, 2) is None
    assert carried_kind(r, 0, 3) == "crate" and carried_kind(r, 0, 4) == "crate"
    assert carried_kind(r, 0, 5) is None


def test_background_tells_racks_and_dump_cells_from_walls_and_floor():
    s = scenario_from_ascii(["T.D#"], [(0, 1)], [[(0, 1)]])
    img = _background(s, None)
    assert tuple(img[0, 0]) == COLORS["rack"] and tuple(img[0, 2]) == COLORS["dump"]
    assert tuple(img[0, 3]) == COLORS["wall"] and tuple(img[0, 1]) == COLORS["free"]


def test_the_scene_draws_every_frame_of_a_carry_and_a_fill(tmp_path):
    for rows in (DOOR_CRATE, PIT_ROOM):
        r, s, cfg = run(rows)
        scene = Scene(s, {"myopic": r}, cfg)
        for t in range(scene.horizon + 1):
            scene.draw(t)
        scene.plt.close(scene.fig)
    r, s, cfg = run(DOOR_CRATE)
    out = animate_runs(s, {"myopic": r}, cfg, path=str(tmp_path / "carry.gif"), fps=4)
    assert os.path.getsize(out) > 3000
