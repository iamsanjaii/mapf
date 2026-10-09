import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_doi
from src.doi.animate import Scene
from src.doi.config import SimConfig
from src.doi.runner import run_episode
from src.doi.scenarios import scenario_from_ascii
from src.doi.story import caption_at, obstacle_prices, plain_events, rent_meters, verdict


def toy_runs():
    s = scenario_from_ascii(run_doi.TOY_ROWS, [(1, 2)], [run_doi.TOY_TASKS], name="toy")
    cfg = SimConfig(n_robots=1, tasks_per_robot=4)
    return s, cfg, {p: run_episode(cfg.replace(policy=p), scenario=s) for p in ("never", "rof")}


def test_rent_meter_crosses_the_price_for_never_and_stops_at_the_push_for_rof():
    s, cfg, runs = toy_runs()
    prices = obstacle_prices(runs)
    assert prices == {(1, 3): (15.0, "shelf_unit")}
    never, = rent_meters(runs["never"], prices)
    rof, = rent_meters(runs["rof"], prices)
    assert never.cumulative[-1] == 32 and never.pushed_tick is None
    assert never.status(never.cumulative.size) == "over the price, not pushed"
    assert rof.pushed_tick == 10 and rof.decided_tick == 10 and rof.status(10) == "pushed away"
    assert rof.status(5) == ""


def test_captions_follow_the_events_in_plain_words():
    s, cfg, runs = toy_runs()
    ev = plain_events(runs["rof"], s)
    assert ev[0] == (10, "Detours so far cost 16, pushing the shelf unit at (1,3) costs 15: robot 0 decides "
                         "to push it")
    assert ev[1][1] == "Robot 0 pushes the shelf unit at (1,3) 2 cells left, to (1,1)"
    assert caption_at(ev, 0, "idle") == ("idle", "")
    now, before = caption_at(ev, 12, "idle")
    assert now.startswith("Robot 0 pushes") and before.startswith("Detours so far cost 16")


def test_verdict_names_the_saving():
    s, cfg, runs = toy_runs()
    assert verdict(runs) == "rof saves 9 (22%) against never: it pushed 1 obstacle aside once the detours added up."


def test_window_widgets_play_scrub_and_restart(monkeypatch):
    s, cfg, runs = toy_runs()
    scene = Scene(s, runs, cfg, controls=True)
    timers = []
    real = scene.fig.canvas.new_timer
    monkeypatch.setattr(scene.fig.canvas, "new_timer", lambda **kw: timers.append(real(**kw)) or timers[-1])
    monkeypatch.setattr(plt, "show", lambda: None)
    scene.show()
    play, restart, speed, scrub = scene._widgets
    step = timers[0].callbacks[0][0]
    for _ in range(5):
        step()
    assert scrub.val == 5
    scrub.set_val(10)                      # scrubbing pauses and jumps
    assert play.label.get_text() == "Play"
    step()
    assert scrub.val == 10                 # paused: the timer does not advance it
    restart._observers.process("clicked", None)
    assert scrub.val == 0 and play.label.get_text() == "Pause"
    speed.set_val(4)
    step()
    assert scrub.val == 4
    scrub.set_val(scene.horizon)
    play._observers.process("clicked", None)   # at the end, Play replays from the start
    assert scrub.val == 0 and play.label.get_text() == "Pause"


def test_verdict_says_why_when_nothing_was_pushed():
    from src.doi.story import detour_paid
    open_floor = scenario_from_ascii(["...L....", "........", "........"], [(2, 0)], [[(2, 7)]])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1)
    runs = {p: run_episode(cfg.replace(policy=p), scenario=open_floor) for p in ("never", "rof")}
    assert detour_paid(runs["never"]) == 0
    assert "no robot's route ran through an obstacle" in verdict(runs)
    # a small detour that no push is cheaper than: rent exists, nothing pushed
    s = scenario_from_ascii([".S...", "....."], [(0, 0)], [[(0, 4)]])
    runs = {p: run_episode(cfg.replace(policy=p), scenario=s) for p in ("never", "rof")}
    assert detour_paid(runs["never"]) == 2 and runs["rof"].removals == 0
    assert "cost the robots 2 steps of detour" in verdict(runs) and "did not push" in verdict(runs)


def test_the_window_explains_an_empty_meter_panel_and_marks_the_final_cost():
    s = scenario_from_ascii(["...L....", "........", "........"], [(2, 0)], [[(2, 7)]])
    cfg = SimConfig(n_robots=1, tasks_per_robot=1)
    runs = {p: run_episode(cfg.replace(policy=p), scenario=s) for p in ("never", "rof")}
    sc = Scene(s, runs, cfg)
    sc.draw(sc.horizon)
    texts = [t.get_text() for ax in sc.meter_axes for t in ax.texts]
    assert all("nothing to push" in t for t in texts) and len(texts) == 2
    assert all(ax.get_title().split("|")[2].strip().startswith("FINAL cost") for ax in sc.map_axes)
