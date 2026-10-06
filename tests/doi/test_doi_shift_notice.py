import pytest
from src.doi.config import SimConfig
from src.doi.notices import NoticeFeed
from src.doi.runner import run_episode
from src.doi.scenarios import build_scenario


def _build(mode, seed=3, tasks=12, **params):
    cfg = SimConfig(scenario="shift_notice", n_robots=4, tasks_per_robot=tasks, seed=seed,
                    scenario_params={"notice_mode": mode, **params})
    return cfg, build_scenario(cfg)


def _about_the_band(s):
    return sorted((n.kind, n.group, n.is_true) for n in s.notices if n.kind != "distractor")


def test_modes_decide_the_tasks_and_the_notices():
    (_, t), (_, m), (_, f), (_, q) = (_build(x) for x in ("true", "missing", "false", "quiet"))
    assert t.tasks == m.tasks and f.tasks == q.tasks and t.tasks != f.tasks
    assert [g[:10] for g in t.tasks] == [g[:10] for g in f.tasks] and t.starts == f.starts
    plain = build_scenario(SimConfig(scenario="shift", n_robots=4, tasks_per_robot=12, seed=3))
    assert plain.tasks == m.tasks and plain.obstacles == m.obstacles
    assert _about_the_band(t) == [("drop", "north bays", True), ("surge", "south bays", True)]
    assert _about_the_band(f) == [("drop", "north bays", False), ("surge", "south bays", False)]
    assert _about_the_band(m) == [] and _about_the_band(q) == []
    for s in (t, m, f, q):
        distractors = [n for n in s.notices if n.kind == "distractor"]
        assert len(distractors) == 1 and distractors[0].group is None and 0 <= distractors[0].emit_tick <= 10


def test_zones_are_the_four_bay_blocks():
    _, s = _build("true")
    assert sorted(s.zones) == ["east north bays", "east south bays", "west north bays", "west south bays"]
    assert all(c < 10 and 0 <= r <= 6 for r, c in s.zones["west north bays"])
    assert all(c > 10 and 8 <= r <= 14 for r, c in s.zones["east south bays"])
    assert not any(cell in s.obstacles for cells in s.zones.values() for cell in cells)


def test_notice_text_is_deterministic_and_follows_the_bank():
    _, a = _build("true")
    _, b = _build("true")
    _, dev = _build("true", notice_bank="dev")
    assert [n.text for n in a.notices] == [n.text for n in b.notices]
    assert [n.text for n in a.notices] != [n.text for n in dev.notices]


def test_bad_mode_and_missing_human_file_are_errors(tmp_path):
    with pytest.raises(ValueError, match="notice_mode"):
        _build("sometimes")
    with pytest.raises(ValueError, match="data/forecasts/README.md"):
        _build("true", notice_bank="human", human_notices=str(tmp_path / "absent.jsonl"))


def test_too_few_tasks_for_the_band_to_move_is_an_error():
    with pytest.raises(ValueError, match="shift_after_task"):
        _build("true", tasks=10)
    with pytest.raises(ValueError, match="shift_after_task"):
        _build("missing", tasks=6)
    _build("quiet", tasks=6)                                   # the band is not meant to move: allowed


class _Stub:
    def __init__(self, rid, pos, finished=False):
        self.id, self.pos, self.finished, self.got = rid, pos, finished, []

    def ingest_notice(self, rec):
        self.got.append(rec)


def test_feed_delivers_each_notice_once_to_the_robot_nearest_the_centre():
    _, s = _build("true")
    feed = NoticeFeed(s)
    far, near = _Stub(0, (0, 0)), _Stub(1, (7, 10))
    for t in range(20):
        feed.emit(t, [far, near])
    assert far.got == [] and feed.delivered == len(s.notices) == 3
    assert sorted((r.notice_id, r.tick, r.text) for r in near.got) == \
        sorted((n.notice_id, n.emit_tick, n.text) for n in s.notices)


def test_feed_with_no_live_robot_does_nothing():
    _, s = _build("true")
    feed = NoticeFeed(s)
    done = _Stub(0, (7, 10), finished=True)
    for t in range(20):
        feed.emit(t, [done])
    assert done.got == [] and feed.delivered == 0


def test_notices_spread_by_gossip_and_not_without_it():
    cfg, s = _build("true")
    shared = run_episode(cfg.replace(policy="rof"), scenario=s)
    alone = run_episode(cfg.replace(policy="rof", r_comm=0.0), scenario=s)
    assert sorted(shared.notice_reach) == sorted(n.notice_id for n in s.notices)
    assert all(v == 1 for v in alone.notice_reach.values())
    assert max(shared.notice_reach.values()) >= 2
    assert not shared.stalled and shared.unfinished_tasks == 0
